"""Model router: one OpenAI-compatible endpoint that swaps model recipes on demand.

Engine-agnostic: everything the router needs beyond the OpenAI API (is the server still
busy, which context length does it serve) comes from a small per-engine adapter, chosen
with "engine" in models.json (vllm, tensorfold, or openai for anything else).

- Requests name a model ("qwen3.6-35b", "qwen3.8-flash-next", an alias, or "any").
- A specific model that isn't loaded triggers a switch: requests for the old model
  are held, in-flight ones finish (drain), the old recipe is stopped, the new one
  started, and waiting requests are then forwarded.
- "any" uses whichever model is loaded (or starting); if none, the last-used one.
- Recipes run on the host as the normal user via nsenter, exactly like ./start.sh.
"""
import asyncio
import hashlib
import json
import logging
import os
import re
import sqlite3
import time

from aiohttp import ClientSession, ClientTimeout, TCPConnector, web
from aiohttp.client_exceptions import ClientConnectionResetError, ClientConnectorError

import thinking

CFG = json.load(open(os.environ.get("ROUTER_CONFIG", "/app/models.json")))
STATE_DIR = os.environ.get("STATE_DIR", "/state")
HOST_UID = os.environ.get("HOST_UID", "1000")
HOST_GID = os.environ.get("HOST_GID", "1000")
BACKEND = CFG["backend_url"].rstrip("/")
MODELS = CFG["models"]
ANY_NAMES = {"any", "auto", "*", "default", ""}
HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te",
       "trailers", "transfer-encoding", "upgrade", "host", "content-length"}

os.makedirs(f"{STATE_DIR}/logs", exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("router")

ALIASES = {}
for _name in MODELS:            # canonical names first: an alias can never shadow a model id
    ALIASES[_name.lower()] = _name
for _name, _m in MODELS.items():
    for a in [_m["served_name"], *_m.get("aliases", [])]:
        if ALIASES.setdefault(a.lower(), _name) != _name:
            log.warning("alias %r of %s already names %s: ignored", a, _name, ALIASES[a.lower()])


class _QuietDisconnects(logging.Filter):
    """Client hang-ups are normal (e.g. a user cancels a stream); don't dump tracebacks."""
    def filter(self, record):
        exc = record.exc_info[1] if record.exc_info else None
        return not isinstance(exc, (ConnectionResetError, ClientConnectionResetError))


logging.getLogger("aiohttp.server").addFilter(_QuietDisconnects())


def _png_data_url(w=256, h=256):
    """Small gradient PNG (stdlib only) for the background image warmup."""
    import base64
    import struct
    import zlib
    raw = b"".join(b"\x00" + b"".join(bytes((x * 255 // w, y * 255 // h, 128)) for x in range(w))
                   for y in range(h))

    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))
    return "data:image/png;base64," + base64.b64encode(png).decode()


_WARMUP_IMAGE = _png_data_url()


CTX_FIELDS = ("max_model_len", "context_length", "context_window", "n_ctx")


class Engine:
    """What the router needs from an inference server beyond the OpenAI API.
    This base class assumes nothing but /v1/models ("engine": "openai")."""
    name = "openai"

    async def _json(self, session, path):
        async with session.get(f"{BACKEND}{path}", timeout=ClientTimeout(total=5)) as r:
            return await r.json(content_type=None) if r.status == 200 else {}

    async def busy(self, session):
        """Requests the server itself still has running or queued (0 if it cannot tell)."""
        return 0

    async def context_length(self, session, served_name):
        """The context length the running server actually serves, or None."""
        for m in (await self._json(session, "/v1/models")).get("data", []):
            if m.get("id") == served_name:
                for k in CTX_FIELDS:
                    if m.get(k):
                        return int(m[k])
                meta = m.get("meta") or {}          # llama.cpp
                if meta.get("n_ctx_train"):
                    return int(meta["n_ctx_train"])
        return None


class VllmEngine(Engine):
    name = "vllm"

    async def busy(self, session):
        async with session.get(f"{BACKEND}/metrics", timeout=ClientTimeout(total=5)) as r:
            text = await r.text()
        return sum(float(v) for v in re.findall(
            r'^vllm:num_requests_(?:running|waiting)\{[^}]*\}\s+([0-9.eE+-]+)', text, re.M))


class TensorFoldEngine(Engine):
    """TensorFold's Python engine has no /metrics; /health carries requests_running and context_length.
    The Zig engine (tensorfold-native) moved them: tensorfold:requests_running/waiting in /metrics,
    context_length in /v1/models."""
    name = "tensorfold"

    async def busy(self, session):
        h = await self._json(session, "/health")
        if "requests_running" in h:
            return float(h["requests_running"] or 0)
        async with session.get(f"{BACKEND}/metrics", timeout=ClientTimeout(total=5)) as r:
            text = await r.text() if r.status == 200 else ""
        return sum(float(v) for v in re.findall(
            r'^tensorfold:requests_(?:running|waiting)(?:\{[^}]*\})?\s+([0-9.eE+-]+)', text, re.M))

    async def context_length(self, session, served_name):
        n = (await self._json(session, "/health")).get("context_length")
        return int(n) if n else await super().context_length(session, served_name)


ENGINES = {e.name: e() for e in (Engine, VllmEngine, TensorFoldEngine)}


def engine_of(name):
    """Adapter for a model; entries without "engine" are vLLM (the first recipes)."""
    kind = MODELS[name].get("engine", "vllm")
    if kind not in ENGINES:
        log.warning("%s: unknown engine %r, treating it as a plain OpenAI server", name, kind)
        return ENGINES["openai"]
    return ENGINES[kind]


class UnknownModel(Exception):
    pass


class SwitchFailed(Exception):
    pass


def resolve(raw):
    """Model field -> canonical model name, or None for 'any'."""
    if raw is None or str(raw).strip().lower() in ANY_NAMES:
        return None
    name = ALIASES.get(str(raw).strip().lower())
    if not name:
        raise UnknownModel(raw)
    return name


def load_state():
    try:
        return json.load(open(f"{STATE_DIR}/state.json"))
    except Exception:
        return {}


def save_state(**kw):
    st = load_state()
    st.update(kw)
    with open(f"{STATE_DIR}/state.json", "w") as f:
        json.dump(st, f)


async def host_run(cmd, logfile=None, timeout=None, capture=False):
    """Run a shell command on the host, as the normal user, in host namespaces."""
    argv = ["nsenter", "-t", "1", "-m", "-u", "-i", "-n", "-p", "--",
            "setpriv", f"--reuid={HOST_UID}", f"--regid={HOST_GID}", "--init-groups",
            "--reset-env", "bash", "-lc", cmd]
    if capture:
        p = await asyncio.create_subprocess_exec(*argv, stdout=asyncio.subprocess.PIPE,
                                                 stderr=asyncio.subprocess.STDOUT,
                                                 stdin=asyncio.subprocess.DEVNULL)
        out, _ = await asyncio.wait_for(p.communicate(), timeout)
        return p.returncode, out.decode(errors="replace")
    with open(logfile, "ab") as lf:
        lf.write(f"\n===== {time.strftime('%F %T')} $ {cmd}\n".encode())
        lf.flush()
        p = await asyncio.create_subprocess_exec(*argv, stdout=lf, stderr=asyncio.subprocess.STDOUT,
                                                 stdin=asyncio.subprocess.DEVNULL,
                                                 start_new_session=True)
        try:
            return await asyncio.wait_for(p.wait(), timeout), None
        except asyncio.TimeoutError:
            p.kill()
            raise


def tail(path, n=15):
    try:
        return "".join(open(path, errors="replace").readlines()[-n:])
    except Exception:
        return ""


class Manager:
    def __init__(self):
        self.current = None        # loaded, ready model
        self.state = "idle"        # idle | ready | draining | stopping | starting | error
        self.target = None         # model being switched to (None = unload)
        self.inflight = 0          # requests being served by the current model
        self.waiting = 0           # requests held while a switch runs
        self.switch_task = None
        self.failures = {}         # model -> (time, message)
        self.last_error = None
        self.last_used = load_state().get("last_used") or CFG["default_model"]
        self.loaded_since = None
        self.cond = asyncio.Condition()
        self.session = None
        # context length per model: {"max_model_len": n, "source": <engine name> | "config.json"}
        self.ctx = load_state().get("ctx", {})
        self.localpatch = {}   # model -> {patch name: "active" | "inactive"} from the last start
        self._bg = set()       # background tasks (start.sh finishing after the model is ready)
        self.last_activity = 0.0   # when the last proxied request finished

    # ---------- backend helpers ----------
    async def backend_model_ids(self):
        try:
            async with self.session.get(f"{BACKEND}/v1/models", timeout=ClientTimeout(total=5)) as r:
                if r.status == 200:
                    return [m["id"] for m in (await r.json()).get("data", [])]
        except Exception:
            pass
        return []

    async def backend_busy(self, name):
        """Running + waiting requests according to the engine's own counters."""
        try:
            return await engine_of(name).busy(self.session)
        except Exception:
            return 0

    async def running_containers(self):
        rc, out = await host_run("docker ps --format '{{.Names}}'", capture=True, timeout=30)
        return set(out.split()) if rc == 0 else set()

    async def learn_ctx(self, name):
        """Record the context length the engine actually serves for `name` (not hard-coded)."""
        eng = engine_of(name)
        try:
            n = await eng.context_length(self.session, MODELS[name]["served_name"])
            if n:
                self.ctx[name] = {"max_model_len": n, "source": eng.name}
                save_state(ctx=self.ctx)
                log.info("%s serves a context of %s tokens (%s)", name, n, eng.name)
            else:
                log.info("%s (%s) does not report its context length; keeping %s", name, eng.name,
                         self.ctx.get(name))
        except Exception as e:
            log.warning("could not read the context length of %s: %s", name, e)

    async def mm_warmup(self, name):
        """One tiny image request once the model is ready and idle, in the background
        (models.json "mm_warmup": true; any engine that takes OpenAI image_url parts).
        For the vLLM recipes its own start-up image warmup is skipped (localpatch
        mm-warmup-skip) so a text request that triggered the switch isn't delayed ~12 s;
        this pays the one-time image-processing setup through the normal request path."""
        body = {"model": MODELS[name]["served_name"], "max_tokens": 1,
                "chat_template_kwargs": {"enable_thinking": False},
                "messages": [{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": _WARMUP_IMAGE}},
                    {"type": "text", "text": "ok"}]}]}
        # Wait for an idle moment: the warmup runs ahead of anything queued behind it,
        # so it must never delay the request that triggered the switch.
        deadline = time.time() + 1800
        while not (self.inflight == 0 and self.waiting == 0 and time.time() - self.last_activity > 3):
            if self.current != name or self.state != "ready" or time.time() > deadline:
                return
            await asyncio.sleep(0.5)
        t0 = time.time()
        try:
            async with self.session.post(f"{BACKEND}/v1/chat/completions", json=body,
                                         timeout=ClientTimeout(total=300)) as r:
                await r.read()
                log.info("%s image warmup: HTTP %s in %.1fs", name, r.status, time.time() - t0)
        except Exception as e:
            log.info("%s image warmup skipped: %s", name, e)

    async def learn_localpatch(self, name):
        """Read which update-safe local speed patches (~/recipes/localpatch) activated."""
        rc, out = await host_run(
            f"docker logs {MODELS[name]['container']} 2>&1 | grep -o '\\[localpatch\\] [a-z-]* [a-z]*' | sort -u",
            capture=True, timeout=60)
        st = {}
        for line in out.splitlines():
            parts = line.split()
            if len(parts) >= 3 and parts[2] in ("active", "inactive"):
                # any "inactive" wins: one process without the patch means it isn't fully applied
                st[parts[1]] = "inactive" if "inactive" in (st.get(parts[1]), parts[2]) else "active"
        self.localpatch[name] = st
        if any(v == "inactive" for v in st.values()):
            log.warning("%s: local patch(es) inactive after update: %s", name, st)

    async def ctx_from_config(self):
        """Fallback for never-loaded models: max_position_embeddings from the HF config.json."""
        for name, m in MODELS.items():
            if name in self.ctx:
                continue
            repo = m.get("hf_repo") or next(
                (a for a in [m["served_name"], *m.get("aliases", [])] if "/" in a), None)
            if not repo:
                continue
            py = ("import json,glob,os;"
                  f"p=sorted(glob.glob(os.path.expanduser('~/.cache/huggingface/hub/models--{repo.replace('/', '--')}/snapshots/*/config.json')));"
                  "c=json.load(open(p[0])) if p else {};"
                  "print(c.get('max_position_embeddings') or c.get('text_config',{}).get('max_position_embeddings') or '')")
            rc, out = await host_run(f"python3 -c \"{py}\"", capture=True, timeout=30)
            if rc == 0 and out.strip().isdigit():
                self.ctx[name] = {"max_model_len": int(out.strip()), "source": "config.json"}
                log.info("%s context from config.json: %s", name, out.strip())

    async def adopt(self):
        """On router start, adopt a model that is already serving on the backend port."""
        ids = await self.backend_model_ids()
        for name, m in MODELS.items():
            if m["served_name"] in ids:
                self.current, self.state, self.loaded_since = name, "ready", time.time()
                log.info("adopted running model %s", name)
                await self.learn_ctx(name)
                await self.learn_localpatch(name)
                break
        await self.ctx_from_config()
        for name in MODELS:        # stopped containers keep their logs
            if name not in self.localpatch:
                await self.learn_localpatch(name)

    # ---------- request admission ----------
    async def acquire(self, want, start_if_idle=True):
        """Block until `want` (or any model if None) is ready; count the request in-flight."""
        t0 = time.time()
        async with self.cond:
            self.waiting += 1
            try:
                while True:
                    w = want
                    if w is None:
                        if self.state == "ready":
                            self.inflight += 1
                            return self.current
                        if not start_if_idle:   # probes/metrics: answer now, never wait
                            raise SwitchFailed(f"no model is ready (router state: {self.state})")
                        if self.switch_task is None:
                            w = self.last_used if self.last_used in MODELS else CFG["default_model"]
                    if w is not None and self.state == "ready" and self.current == w:
                        self.inflight += 1
                        self._mark_used(w)
                        return w
                    f = self.failures.get(w)
                    if w is not None and f and f[0] >= t0:
                        raise SwitchFailed(f"{w} failed to start: {f[1]}")
                    if self.switch_task is None and w is not None:
                        self.switch_task = asyncio.create_task(self._switch(w))
                    await self.cond.wait()
            finally:
                self.waiting -= 1

    def needs_wait(self, want):
        """True if serving `want` would have to wait for a load/switch."""
        return not (self.state == "ready" and (want is None or self.current == want))

    async def release(self):
        async with self.cond:
            self.inflight -= 1
            self.last_activity = time.time()
            self.cond.notify_all()

    def _mark_used(self, name):
        if self.last_used != name:
            self.last_used = name
            save_state(last_used=name)

    # ---------- switching ----------
    async def _set(self, **kw):
        async with self.cond:
            for k, v in kw.items():
                setattr(self, k, v)
            self.cond.notify_all()

    async def _drain(self):
        log.info("draining %s (%d in flight)", self.current, self.inflight)
        timeout = CFG.get("drain_timeout_s", 1800)
        try:
            async with self.cond:
                await asyncio.wait_for(self.cond.wait_for(lambda: self.inflight == 0), timeout)
            for _ in range(60):   # also let the engine finish anything queued internally
                if await self.backend_busy(self.current) == 0:
                    break
                await asyncio.sleep(1)
        except asyncio.TimeoutError:
            log.warning("drain timed out after %ss; stopping anyway", timeout)

    async def _stop_model(self, name):
        m = MODELS[name]
        logf = f"{STATE_DIR}/logs/{name}.log"
        log.info("stopping %s", name)
        rc, _ = await host_run(f"cd {m['dir']} && {m['stop']}", logf, timeout=300)
        if rc != 0:
            log.warning("stop of %s exited %s", name, rc)

    async def _switch(self, target):
        """Drain + stop the current model, then start `target` (None = just unload)."""
        old = self.current
        try:
            await self._set(target=target, state="draining" if old else "stopping")
            if old:
                await self._drain()
            await self._set(state="stopping")
            running = await self.running_containers()
            for name, m in MODELS.items():   # stop whatever holds the GPU / port
                if name == old or m["container"] in running:
                    await self._stop_model(name)
            await self._set(current=None, loaded_since=None)
            if target is None:
                await self._set(state="idle")
                return
            await self._set(state="starting")
            m = MODELS[target]
            logf = f"{STATE_DIR}/logs/{target}.log"
            log.info("starting %s", target)
            t0 = time.time()
            timeout = CFG.get("ready_timeout_s", 2700)
            # Don't wait for start.sh's own (slower) readiness loop: the model is ready as
            # soon as it serves its name. The old model is already stopped, so nothing
            # else can answer on the backend port. start.sh finishes in the background.
            start = asyncio.create_task(host_run(f"cd {m['dir']} && {m['start']}", logf, timeout=timeout))
            self._bg.add(start)
            start.add_done_callback(self._bg.discard)
            while m["served_name"] not in await self.backend_model_ids():
                if start.done():
                    rc, _ = start.result()
                    if rc != 0:
                        raise RuntimeError(f"start.sh exited {rc}; log tail:\n{tail(logf)}")
                if time.time() - t0 > timeout:
                    raise RuntimeError("timed out waiting for the model to answer")
                await asyncio.sleep(0.5)
            log.info("%s ready after %.0fs", target, time.time() - t0)
            if m.get("mm_warmup"):
                warm = asyncio.create_task(self.mm_warmup(target))
                self._bg.add(warm)
                warm.add_done_callback(self._bg.discard)
            await self.learn_ctx(target)
            await self.learn_localpatch(target)
            self._mark_used(target)
            await self._set(current=target, state="ready", loaded_since=time.time(), last_error=None)
        except Exception as e:
            log.error("switch to %s failed: %s", target, e)
            async with self.cond:
                self.state, self.current = "error", None
                self.last_error = f"{target}: {e}"
                if target:
                    self.failures[target] = (time.time(), str(e)[:2000])
                self.cond.notify_all()
        finally:
            await self._set(switch_task=None, target=None)

    async def mark_backend_dead(self, name):
        async with self.cond:
            if self.current == name and self.state == "ready" and not await self.backend_model_ids():
                log.warning("%s stopped answering; will restart on next request", name)
                self.state, self.current = "idle", None
                self.cond.notify_all()

    async def restart(self, name):
        """Drain and restart `name` (used after an update)."""
        async with self.cond:
            if self.switch_task is None:
                self.switch_task = asyncio.create_task(self._switch(name))
            task = self.switch_task
        await task

    def status(self):
        return {
            "state": self.state,
            "loaded": self.current,
            "switching_to": self.target,
            "in_flight": self.inflight,
            "waiting": self.waiting,
            "last_used": self.last_used,
            "loaded_for_s": round(time.time() - self.loaded_since) if self.loaded_since else None,
            "last_error": self.last_error,
            "models": {n: {"aliases": m.get("aliases", []), "served_name": m["served_name"],
                           "container": m["container"], "engine": engine_of(n).name}
                       for n, m in MODELS.items()},
        }


mgr = Manager()


# ---------------------------------------------------------------- updates
class Updater:
    """Checks each configured model for recipe (git), Docker image and weights (HF) updates,
    and applies them as background jobs. Entirely driven by models.json."""

    def __init__(self):
        self.info = {}          # model -> {"recipe": {...}, "image": {...}, "weights": {...}}
        self.checked_at = None
        self.jobs = {}          # model -> job dict
        self.lock = asyncio.Lock()
        self.web = None         # session for internet APIs (with decompression)

    async def image_of(self, name):
        """(registry ref, local image name) of a model's Docker image.
        Default: the image its container was created from, under the same name on both
        sides. models.json can name it ("image"), or give "image_cmd": a command run in
        the recipe dir that prints "REGISTRY_REF [LOCAL_NAME]" — for recipes that pull a
        tag derived from their own files and re-tag it locally."""
        m = MODELS[name]
        if m.get("image_cmd"):
            rc, out = await host_run(f"cd {m['dir']} && {m['image_cmd']}", capture=True, timeout=60)
            parts = out.split()[-2:] if rc == 0 else []
            if not parts:
                return None, None
            return parts[0], parts[-1]
        if m.get("image"):
            return m["image"], m["image"]
        rc, out = await host_run(f"docker inspect -f '{{{{.Config.Image}}}}' {m['container']}",
                                 capture=True, timeout=30)
        img = out.strip() if rc == 0 and out.strip() else None
        return img, img

    async def check_recipe(self, name):
        script = ("git fetch -q origin 2>&1 | tail -1; "
                  "up=$(git rev-parse --abbrev-ref origin/HEAD 2>/dev/null || echo origin/main); "
                  "base=$(git merge-base HEAD $up); "
                  "echo BEHIND=$(git rev-list --count $base..$up); "
                  "echo LOCAL=$(git rev-list --count $up..HEAD); "
                  "echo BRANCH=$(git rev-parse --abbrev-ref HEAD); "
                  "git log --format='C %h %cs %s' $base..$up | head -20")
        rc, out = await host_run(f"cd {MODELS[name]['dir']} && {script}", capture=True, timeout=120)
        r = {"behind": None, "local_patches": None, "commits": []}
        for line in out.splitlines():
            if line.startswith("BEHIND="):
                r["behind"] = int(line[7:] or 0)
            elif line.startswith("LOCAL="):
                r["local_patches"] = int(line[6:] or 0)
            elif line.startswith("BRANCH="):
                r["branch"] = line[7:]
            elif line.startswith("C "):
                r["commits"].append(line[2:])
        if r["behind"] is None:
            r["error"] = out.strip()[-300:] or "git check failed"
        r["update_available"] = bool(r["behind"])
        return r

    async def check_image(self, name):
        img, local_name = await self.image_of(name)
        if not img:
            return {"error": "image unknown (model never started)"}
        if MODELS[name].get("image_build"):
            # built locally from the recipe (no registry): an update is due when the image the current
            # recipe + patches would run (image_cmd) has not been built yet
            rc, out = await host_run(f"docker image inspect {local_name} --format '{{{{.Id}}}}'",
                                     capture=True, timeout=30)
            have = rc == 0 and out.strip().startswith("sha256:")
            return {"image": f"{local_name} (built locally)", "local_digest": out.strip() if have else None,
                    "remote_digest": None, "update_available": not have}
        rc1, local = await host_run(
            f"docker image inspect {local_name} --format '{{{{json .RepoDigests}}}}'; "
            f"docker image inspect {local_name} --format '{{{{.Id}}}}'", capture=True, timeout=30)
        local, _, local_id = local.strip().partition("\n")
        rc2, remote = await host_run(
            f"timeout 60 docker buildx imagetools inspect {img} --format '{{{{json .Manifest.Digest}}}}'",
            capture=True, timeout=90)
        try:
            digests = [d.split("@")[-1] for d in json.loads(local.strip()) or []] if rc1 == 0 else []
        except ValueError:
            digests = []
        # with the containerd image store a pulled image's id is its manifest digest: accept it too
        if rc1 == 0 and local_id.strip().startswith("sha256:") and local_id.strip() not in digests:
            digests.append(local_id.strip())
        rd = remote.strip().strip('"') if rc2 == 0 else ""
        if local_name != img:
            img = f"{img} (local: {local_name})"
        if rc1 != 0:
            return {"image": img, "local_digest": None,
                    "remote_digest": rd if rd.startswith("sha256:") else None,
                    "update_available": rd.startswith("sha256:"), "error": "image not present locally"}
        r = {"image": img, "local_digest": rd if rd in digests else digests[0] if digests else None,
             "remote_digest": rd if rd.startswith("sha256:") else None}
        if not r["remote_digest"]:
            r["error"] = "registry check failed"
        r["update_available"] = bool(r["remote_digest"] and digests and r["remote_digest"] not in digests)
        return r

    async def check_weights(self, name):
        repo = MODELS[name].get("hf_repo")
        if not repo:
            return {"error": "no hf_repo configured"}
        branch = MODELS[name].get("hf_branch", "main")     # repos that keep each quant on its own branch
        rc, local = await host_run(
            f"cat ~/.cache/huggingface/hub/models--{repo.replace('/', '--')}/refs/{branch}",
            capture=True, timeout=15)
        r = {"repo": repo if branch == "main" else f"{repo}@{branch}",
             "local_rev": local.strip() if rc == 0 else None, "remote_rev": None}
        try:
            async with self.web.get(f"https://huggingface.co/api/models/{repo}/revision/{branch}",
                                    timeout=ClientTimeout(total=30)) as resp:
                if resp.status == 200:
                    r["remote_rev"] = (await resp.json()).get("sha")
                else:
                    r["error"] = f"Hugging Face API HTTP {resp.status}"
        except Exception as e:
            r["error"] = f"Hugging Face API: {e}"
        r["update_available"] = bool(r["local_rev"] and r["remote_rev"] and r["local_rev"] != r["remote_rev"])
        return r

    async def check_all(self, force=False):
        async with self.lock:
            if not force and self.checked_at and time.time() - self.checked_at < 60:
                return
            for name in MODELS:
                res = {}
                for part, fn in (("recipe", self.check_recipe), ("image", self.check_image),
                                 ("weights", self.check_weights)):
                    try:
                        res[part] = await fn(name)
                    except Exception as e:
                        res[part] = {"error": str(e)[:300]}
                self.info[name] = res
            self.checked_at = time.time()
            log.info("update check done: %s", {n: [p for p in ("recipe", "image", "weights")
                                                   if r[p].get("update_available")]
                                               for n, r in self.info.items()})

    async def periodic(self):
        while True:
            try:
                await self.check_all(force=True)
            except Exception as e:
                log.warning("update check failed: %s", e)
            await asyncio.sleep(CFG.get("update_check_interval_s", 21600))

    def busy(self, name):
        j = self.jobs.get(name)
        return bool(j and j["state"] == "running")

    async def run_job(self, name, parts, restart):
        m = MODELS[name]
        logf = f"{STATE_DIR}/logs/update-{name}.log"
        job = self.jobs[name]
        ok = True
        try:
            for part in parts:
                job["step"] = part
                if part == "recipe" and m.get("update_check"):
                    # the recipe's own compatibility check with our patches comes first (exit 3: incompatible)
                    rc, _ = await host_run(f"cd {m['dir']} && {m['update_check']}", logf, timeout=1800)
                    if rc != 0:
                        job["results"][part] = ("incompatible with our patches: left unchanged (see log)"
                                                if rc == 3 else f"compatibility check failed (exit {rc})")
                        ok = False
                        break
                if part == "recipe":
                    cmd = ("git fetch -q origin && "
                           "up=$(git rev-parse --abbrev-ref origin/HEAD 2>/dev/null || echo origin/main) && "
                           "if git rebase --autostash $up; then echo REBASE_OK; "
                           "else git rebase --abort; echo REBASE_CONFLICT; exit 3; fi")
                    rc, _ = await host_run(f"cd {m['dir']} && {cmd}", logf, timeout=600)
                    job["results"][part] = ("updated" if rc == 0 else
                                            "conflict with local patches: left unchanged" if rc == 3
                                            else f"failed (exit {rc})")
                elif part == "image":
                    if m.get("image_update"):       # the recipe's own way to get its image
                        cmd = f"cd {m['dir']} && {m['image_update']}"
                    else:
                        img, _ = await self.image_of(name)
                        cmd = f"for i in $(seq 1 20); do docker pull {img} && exit 0; sleep 20; done; exit 1"
                    rc, _ = await host_run(cmd, logf, timeout=6 * 3600)
                    job["results"][part] = "updated" if rc == 0 else f"failed (exit {rc})"
                elif part == "weights":
                    rc, _ = await host_run(f"cd {m['dir']} && {m['weights_update']}", logf, timeout=12 * 3600)
                    job["results"][part] = "updated" if rc == 0 else f"failed (exit {rc})"
                    if rc == 0 and m.get("hf_repo"):
                        md = f"~/.cache/huggingface/hub/models--{m['hf_repo'].replace('/', '--')}"
                        # every file link must resolve inside what the recipe mounts: the model folder
                        # (default), or the whole HF cache ("weights_links": "cache" — newer hf CLIs
                        # keep blobs in a store shared by all models, hub/blobs)
                        root = "~/.cache/huggingface" if m.get("weights_links") == "cache" else md
                        chk = (f"d=$(readlink -f {root}); n=0; for l in {md}/snapshots/*/*; do "
                               "t=$(readlink -f \"$l\") || { n=$((n+1)); continue; }; "
                               "[ -e \"$t\" ] && [[ \"$t\" == \"$d\"/* ]] || n=$((n+1)); done; echo $n")
                        _, bad = await host_run(chk, capture=True, timeout=60)
                        if bad.strip() not in ("", "0"):
                            job["results"][part] = (f"downloaded, but {bad.strip()} file link(s) are broken or "
                                                    f"point outside {root}: check before starting")
                            rc = 1
                ok = ok and rc == 0
                if rc != 0:
                    break
            if ok and restart and mgr.current == name:
                job["step"] = "restart"
                await mgr.restart(name)
                job["results"]["restart"] = "restarted" if mgr.current == name else f"failed: {mgr.last_error}"
            final = "done" if ok else "failed"
        except Exception as e:
            final, job["error"] = "failed", str(e)[:500]
        # re-check before reporting completion, so the UI never shows stale "update available"
        job["step"] = "re-checking"
        try:
            await self.check_all(force=True)
        except Exception as e:
            log.warning("post-update check failed: %s", e)
        job["log_tail"] = tail(logf, 15)
        job["step"], job["finished"], job["state"] = None, time.time(), final

    def start_job(self, name, parts, restart):
        self.jobs[name] = {"state": "running", "parts": parts, "restart": restart, "step": None,
                           "started": time.time(), "finished": None, "results": {}}
        asyncio.create_task(self.run_job(name, parts, restart))
        return self.jobs[name]

    def report(self):
        out = {}
        for name in MODELS:
            info = self.info.get(name, {})
            job = self.jobs.get(name)
            loaded = mgr.current == name
            restart_required = bool(loaded and job and job.get("finished") and mgr.loaded_since
                                    and job["finished"] > mgr.loaded_since
                                    and any(v == "updated" for v in job["results"].values()))
            out[name] = {**info, "loaded": loaded, "job": job, "restart_required": restart_required,
                         "localpatch": mgr.localpatch.get(name),
                         "update_available": any((info.get(p) or {}).get("update_available")
                                                 for p in ("recipe", "image", "weights"))}
        return {"checked_at": self.checked_at, "models": out}


updater = Updater()


def err(status, message, **extra):
    return web.json_response({"error": {"message": message, "type": "router_error", **extra}},
                             status=status)


async def list_models(request):
    def ctx_fields(name):
        n = (mgr.ctx.get(name) or {}).get("max_model_len")
        # different clients read different field names
        return {"max_model_len": n, "context_length": n, "context_window": n} if n else {}

    # "any" first: clients that default to the first listed model then never force a switch
    any_target = mgr.current or mgr.target or mgr.last_used
    data = [{"id": "any", "object": "model", "owned_by": "model-router",
             "loaded": mgr.current is not None, "resolves_to": any_target,
             "description": "whichever model is loaded; no switch", **ctx_fields(any_target)}]
    data += [{"id": n, "object": "model", "owned_by": "model-router",
              "loaded": mgr.current == n, "aliases": m.get("aliases", []),
              "context_source": (mgr.ctx.get(n) or {}).get("source"), **ctx_fields(n)}
             for n, m in MODELS.items()]
    return web.json_response({"object": "list", "data": data})


async def router_status(request):
    return web.json_response(mgr.status())


async def router_switch(request):
    body = await request.json() if request.can_read_body else {}
    try:
        want = resolve(body.get("model"))
    except UnknownModel as e:
        return err(404, f"unknown model {e}", available=list(MODELS))
    try:
        name = await mgr.acquire(want)
    except SwitchFailed as e:
        return err(503, str(e))
    await mgr.release()
    return web.json_response({"loaded": name, **mgr.status()})


async def router_stop(request):
    async with mgr.cond:
        if mgr.switch_task is None and mgr.current:
            mgr.switch_task = asyncio.create_task(mgr._switch(None))
        task = mgr.switch_task
    if task:
        await task
    return web.json_response(mgr.status())


async def router_updates(request):
    if request.query.get("refresh") in ("1", "true"):
        await updater.check_all(force=True)
    elif updater.checked_at is None:
        await updater.check_all()
    return web.json_response(updater.report())


async def router_update(request):
    body = await request.json() if request.can_read_body else {}
    try:
        name = resolve(body.get("model"))
    except UnknownModel as e:
        return err(404, f"unknown model {e}", available=list(MODELS))
    if name is None:
        return err(400, "name a specific model to update")
    if updater.busy(name):
        return err(409, f"an update for {name} is already running")
    if mgr.switch_task is not None and name in (mgr.current, mgr.target):
        return err(409, f"{name} is starting/stopping right now; try again when it is ready")
    parts = body.get("parts") or [p for p in ("recipe", "image", "weights")
                                  if (updater.info.get(name, {}).get(p) or {}).get("update_available")]
    parts = [p for p in ("recipe", "image", "weights") if p in parts]   # fixed order
    if not parts:
        return err(400, "nothing to update (run a check first, or pass parts)")
    job = updater.start_job(name, parts, bool(body.get("restart")))
    return web.json_response({"model": name, "job": job})


async def router_restart(request):
    body = await request.json() if request.can_read_body else {}
    try:
        name = resolve(body.get("model"))
    except UnknownModel as e:
        return err(404, f"unknown model {e}", available=list(MODELS))
    if name is None or mgr.current != name:
        return err(409, "only the loaded model can be restarted")
    await mgr.restart(name)
    return web.json_response(mgr.status())


async def ui(request):
    return web.FileResponse(os.environ.get("ROUTER_UI", "/app/ui.html"))


async def health(request):
    return web.json_response({"router": "ok", **mgr.status()})


REQ_LOG = f"{STATE_DIR}/requests.jsonl"     # one line a completion: tokens, cached tokens, time to first token
REQ_LOG_MAX = 64 << 20                       # then renamed to requests.jsonl.1 (the one before is dropped)
STORE_DB = os.environ.get("PROMPT_CACHE_DB", "/prompt-cache/index.sqlite")


class Meter:
    """Watches a relayed completion for its first token and its usage, then appends one line to REQ_LOG."""

    def __init__(self, request, data, stream):
        msgs = data.get("messages") if isinstance(data.get("messages"), list) else None
        head = json.dumps(msgs[:2] if msgs else data.get("prompt", "")[:2000], sort_keys=True, default=str)
        self.rec = {"t": round(time.time(), 3), "path": request.path, "client": request.remote, "stream": stream,
                    "conv": hashlib.blake2b(head.encode(), digest_size=6).hexdigest(),
                    "messages": len(msgs) if msgs else None}
        self.stream, self.t0, self.ttft, self.usage, self.buf, self.body = stream, time.monotonic(), None, None, "", b""

    def feed(self, chunk):
        if not self.stream:
            if len(self.body) < (8 << 20):
                self.body += chunk
            return
        self.buf += chunk.decode("utf-8", "replace")
        *lines, self.buf = self.buf.split("\n")
        for line in lines:
            if not line.startswith("data:") or line.startswith("data: [DONE]"):
                continue
            try:
                d = json.loads(line[5:])
            except ValueError:
                continue
            if not isinstance(d, dict):
                continue
            self.usage = d.get("usage") or self.usage
            if self.ttft is None:
                for ch in d.get("choices") or []:
                    dl = ch.get("delta") or {}
                    if (dl.get("content") or dl.get("reasoning_content") or dl.get("reasoning")
                            or dl.get("tool_calls") or ch.get("text")):
                        self.ttft = time.monotonic() - self.t0
                        break

    def done(self, name, status):
        total = time.monotonic() - self.t0
        if not self.stream:
            try:
                self.usage = json.loads(self.body).get("usage")
            except (ValueError, AttributeError):
                pass
        u = self.usage or {}
        rec = {**self.rec, "model": name, "status": status, "prompt": u.get("prompt_tokens"),
               "cached": (u.get("prompt_tokens_details") or {}).get("cached_tokens"),
               "completion": u.get("completion_tokens"),
               "ttft_s": round(self.ttft if self.ttft is not None else total, 3), "total_s": round(total, 3)}
        try:
            if os.path.exists(REQ_LOG) and os.path.getsize(REQ_LOG) > REQ_LOG_MAX:
                os.replace(REQ_LOG, REQ_LOG + ".1")
            with open(REQ_LOG, "a") as f:
                f.write(json.dumps(rec) + "\n")
        except OSError as e:
            log.warning("request log: %s", e)


async def router_cache(request):
    """Prompt-cache report: per model and per conversation from REQ_LOG, and the TensorFold disk store."""
    hours = float(request.query.get("hours", 24))
    since = time.time() - hours * 3600
    rows = []
    for path in (REQ_LOG + ".1", REQ_LOG):
        try:
            with open(path) as f:
                for line in f:
                    try:
                        r = json.loads(line)
                    except ValueError:
                        continue
                    if r.get("t", 0) >= since and r.get("prompt"):
                        rows.append(r)
        except OSError:
            pass

    def summary(rs):
        p = sum(r["prompt"] for r in rs)
        c = sum(r.get("cached") or 0 for r in rs)
        tt = sorted(r["ttft_s"] for r in rs)
        return {"requests": len(rs), "prompt_tokens": p, "cached_tokens": c,
                "hit_pct": round(100 * c / p, 1) if p else None,
                "uncached_tokens": p - c, "ttft_median_s": tt[len(tt) // 2] if tt else None}

    per_model, per_conv = {}, {}
    for r in rows:
        per_model.setdefault(r.get("model"), []).append(r)
        per_conv.setdefault((r.get("model"), r.get("conv")), []).append(r)
    convs = sorted(per_conv.items(), key=lambda kv: -sum(r["prompt"] - (r.get("cached") or 0) for r in kv[1]))
    out = {"hours": hours, "all": summary(rows), "models": {m: summary(rs) for m, rs in per_model.items()},
           "top_conversations": [{"model": m, "conv": c, "last": max(r["t"] for r in rs), **summary(rs)}
                                 for (m, c), rs in convs[:15]],
           "recent": rows[-20:]}
    try:
        db = sqlite3.connect(f"file:{STORE_DB}?mode=ro", uri=True, timeout=5)
        out["disk_store"] = [{"label": label, "key": model, "pinned": bool(pinned), "points": n,
                              "point_gib": round(b / 2**30, 2), "hits": hits}
                             for label, model, pinned, n, b, hits in db.execute(
                                 "SELECT label, model, pinned, COUNT(*), SUM(bytes), SUM(hits) FROM points "
                                 "GROUP BY label, model, pinned")]
        out["disk_store_block_gib"] = round((db.execute("SELECT COALESCE(SUM(bytes),0) FROM blocks")
                                             .fetchone()[0]) / 2**30, 2)
        db.close()
    except sqlite3.Error as e:
        out["disk_store"] = f"unavailable: {e}"
    return web.json_response(out)


async def proxy(request):
    body = await request.read()
    data = None
    if body:
        try:
            data = json.loads(body)
        except ValueError:
            data = None
    raw = data.get("model") if isinstance(data, dict) else None
    try:
        want = resolve(raw)
    except UnknownModel:
        return err(404, f"unknown model '{raw}'", available=[*MODELS, "any"])
    # Inference calls (POST) may start a model; GETs like /metrics never do.
    start_if_idle = request.method == "POST"
    stream = isinstance(data, dict) and bool(data.get("stream"))
    early = None   # response started before a model was ready (keep-alive while switching)
    try:
        for attempt in range(2):
            if start_if_idle and early is None and mgr.needs_wait(want):
                early = await start_early(request, want, stream)
            try:
                name = await (wait_keepalive(early, want, stream) if early
                              else mgr.acquire(want, start_if_idle=start_if_idle))
            except SwitchFailed as e:
                if early:
                    return await finish_early_error(early, str(e), stream)
                return err(503, str(e))
            meter = (Meter(request, data, stream) if request.method == "POST" and isinstance(data, dict)
                     and request.path.rstrip("/").endswith("completions") else None)
            status = None
            try:
                fwd, out, added_budget = body, data, None
                if isinstance(data, dict) and "model" in data:
                    data["model"] = MODELS[name]["served_name"]
                    fwd = json.dumps(data).encode()
                eng = engine_of(name).name
                if (isinstance(data, dict) and request.method == "POST"
                        and request.path.rstrip("/").endswith("completions")):
                    # thinking on/off / effort / budget -> the engine's own dialect (thinking.py)
                    cfg = {**MODELS[name], **({"thinking_budget": False} if BUDGET_REFUSED.get(name) else {})}
                    out, intent = thinking.normalize(data, eng, cfg)
                    bf = thinking.adapter_for(eng).budget_field
                    if bf and bf in out and bf not in data:
                        added_budget = bf
                    if meter and not intent.empty:   # field names and small values only, never content
                        meter.rec["think"] = {"in": [f"{k}={v}" for k, v in intent.sources],
                                              "added": thinking.added_fields(data, out)}
                    if out is not data:
                        fwd = json.dumps(out).encode()
                headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP}
                r = await mgr.session.request(request.method, BACKEND + request.path_qs,
                                              data=fwd if fwd else None, headers=headers)
                if r.status == 400 and added_budget and not BUDGET_REFUSED.get(name):
                    # the engine refuses a budget the router added (e.g. vLLM without a reasoning config):
                    # remember that for this model and send the request again without it
                    text = await r.text()
                    r.release()
                    if added_budget in text:
                        BUDGET_REFUSED[name] = True
                        log.warning("%s refused %s; not sending it again: %s", name, added_budget, text[:200])
                        fwd = json.dumps(thinking.strip_budget(out, eng)).encode()
                        r = await mgr.session.request(request.method, BACKEND + request.path_qs,
                                                      data=fwd, headers=headers)
                    else:
                        r = _Replayed(r.status, r.headers, text.encode())
                async with r:
                    status = r.status
                    if early:   # status/headers already sent; just relay the body
                        if r.status != 200 and stream:
                            await early.write(b"data: " + await r.read() + b"\n\ndata: [DONE]\n\n")
                        else:
                            async for chunk in r.content.iter_any():
                                if meter:
                                    meter.feed(chunk)
                                await early.write(chunk)
                        await early.write_eof()
                        return early
                    resp = web.StreamResponse(status=r.status, headers={
                        k: v for k, v in r.headers.items() if k.lower() not in HOP})
                    resp.headers["X-Router-Model"] = name
                    await resp.prepare(request)
                    async for chunk in r.content.iter_any():
                        if meter:
                            meter.feed(chunk)
                        await resp.write(chunk)
                    await resp.write_eof()
                    return resp
            except ClientConnectorError:
                await mgr.mark_backend_dead(name)
                if attempt == 1:
                    msg = f"{name} is not answering"
                    return await finish_early_error(early, msg, stream) if early else err(502, msg)
            finally:
                if meter and status is not None:
                    meter.done(name, status)
                await mgr.release()
    except (ConnectionResetError, ClientConnectionResetError, asyncio.CancelledError):
        # client went away; closing the backend request makes the engine abort it
        log.info("client disconnected (%s %s)", request.method, request.path)
        raise


BUDGET_REFUSED = {}   # model -> True once its engine refused a router-added thinking budget (until router restart)


class _Replayed:
    """A backend response whose body was already read (400 inspected for the budget retry)."""

    def __init__(self, status, headers, body):
        self.status, self.headers, self._body = status, headers, body
        self.content = self

    async def read(self):
        return self._body

    async def iter_any(self):
        yield self._body

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False


KEEPALIVE_S = 2   # browsers drop a stream that shows no activity ~5 s after a "network change"


async def start_early(request, want, stream):
    """A switch is needed: send headers now so clients don't time out during the load."""
    guess = want or mgr.target or mgr.current or mgr.last_used
    resp = web.StreamResponse(status=200, headers={
        "Content-Type": "text/event-stream" if stream else "application/json",
        "Cache-Control": "no-cache", "X-Router-Model": guess or "any"})
    await resp.prepare(request)
    return resp


async def wait_keepalive(resp, want, stream):
    """acquire() while writing keep-alive bytes: SSE comments for streams, spaces for JSON."""
    task = asyncio.create_task(mgr.acquire(want))
    try:
        while True:
            done, _ = await asyncio.wait({task}, timeout=KEEPALIVE_S)
            if done:
                return task.result()
            who = mgr.target or want or "a model"
            await resp.write(f": waiting for {who} (router: {mgr.state})\n\n".encode()
                             if stream else b" ")
    except BaseException:
        if task.done() and not task.cancelled() and task.exception() is None:
            await mgr.release()          # acquired just as the client left
        else:
            task.cancel()
        raise


async def finish_early_error(resp, message, stream):
    payload = json.dumps({"error": {"message": message, "type": "router_error"}})
    await resp.write((f"data: {payload}\n\ndata: [DONE]\n\n" if stream else payload).encode())
    await resp.write_eof()
    return resp


async def on_startup(app):
    mgr.session = ClientSession(timeout=ClientTimeout(total=None, sock_connect=10),
                                connector=TCPConnector(limit=0), auto_decompress=False)
    await mgr.adopt()
    updater.web = ClientSession()
    app["update_task"] = asyncio.create_task(updater.periodic())


async def on_cleanup(app):
    app["update_task"].cancel()
    await updater.web.close()
    await mgr.session.close()


def main():
    app = web.Application(client_max_size=512 * 1024 ** 2)
    app.router.add_get("/", ui)
    app.router.add_get("/v1/models", list_models)
    app.router.add_get("/router/status", router_status)
    app.router.add_post("/router/switch", router_switch)
    app.router.add_post("/router/stop", router_stop)
    app.router.add_get("/router/updates", router_updates)
    app.router.add_post("/router/update", router_update)
    app.router.add_post("/router/restart", router_restart)
    app.router.add_get("/health", health)
    app.router.add_get("/router/cache", router_cache)
    app.router.add_route("*", "/{tail:.*}", proxy)
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    web.run_app(app, host=CFG.get("listen_host", "0.0.0.0"), port=CFG.get("listen_port", 8888),
                access_log=None)


if __name__ == "__main__":
    main()
