"""Install job tests (CPU, no host, host_run mocked): python3 test_install.py  (needs aiohttp)"""

import asyncio
import json
import os
import re
import tempfile
import unittest

TMP = tempfile.mkdtemp()
BASE, DEP = "Base-Model", "Dep-Model"
CFG = {"listen_host": "127.0.0.1", "listen_port": 0, "backend_url": "http://127.0.0.1:9",
       "default_model": DEP, "models": {
           BASE: {"engine": "tensorfold", "aliases": ["base"], "dir": f"{TMP}/base",
                  "repo": "https://example.invalid/base", "branch": "local-patches",
                  "start": "./start", "stop": "./stop", "served_name": BASE, "container": "t-base",
                  "hf_repo": "org/base", "hf_branch": "4bpw", "weights_update": "./dl",
                  "image_cmd": "./img ref", "image_update": "./img update"},
           DEP: {"engine": "tensorfold", "aliases": ["dep"], "dir": f"{TMP}/dep",
                 "repo": "https://example.invalid/dep", "requires": ["base"],
                 "start": "./start", "stop": "./stop", "served_name": DEP, "container": "t-dep",
                 "hf_repo": "org/dep", "weights_install": "./dl", "weights_update": "./dl --follow",
                 "image_cmd": "./img ref", "image_update": "./img update", "prebuild": "./prebuild"}}}
json.dump(CFG, open(f"{TMP}/models.json", "w"))
os.environ.update(ROUTER_CONFIG=f"{TMP}/models.json", STATE_DIR=f"{TMP}/state")

import router  # noqa: E402

CHECK = re.compile(r"echo RECIPE=1")


class Host:
    """Fake host: which parts each model has; install steps flip them on; records commands."""

    def __init__(self, have, fail=()):
        self.have = have
        self.fail = fail
        self.cmds = []

    def model_of(self, cmd):
        return BASE if f"{TMP}/base" in cmd else DEP

    async def run(self, cmd, logfile=None, timeout=None, capture=False):
        name = self.model_of(cmd)
        h = self.have[name]
        if CHECK.search(cmd):
            return 0, "".join(f"{p.upper()}={int(h[p])}\n" for p in ("recipe", "weights", "image"))
        step = ("recipe" if "git clone" in cmd else "prebuild" if "./prebuild" in cmd
                else "image" if "./img update" in cmd else "weights" if "./dl" in cmd else "?")
        self.cmds.append((name, step, cmd))
        if (name, step) in self.fail:
            return 1, None
        if step in h:
            h[step] = True
        return 0, None


def missing_all():
    return {"recipe": False, "image": False, "weights": False}


class Install(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        router.installer = router.Installer()
        router.mgr.__init__()

    def use(self, host):
        router.host_run = host.run
        return host

    async def install(self, name):
        job = router.installer.start_job(name)
        for _ in range(200):
            if job["state"] in ("done", "failed"):
                return job
            await asyncio.sleep(0.01)
        self.fail("install did not finish")

    def test_requires_resolved(self):
        self.assertEqual(router.MODELS[DEP]["requires"], [BASE])

    def test_chain(self):
        self.assertEqual(router.installer.chain(DEP),
                         [(BASE, ("recipe", "image")), (DEP, ("recipe", "image", "weights"))])
        self.assertEqual(router.installer.chain(BASE), [(BASE, ("recipe", "image", "weights"))])

    async def test_check(self):
        self.use(Host({BASE: {"recipe": True, "image": False, "weights": True}, DEP: missing_all()}))
        await router.installer.check_all()
        self.assertFalse(router.installer.installed(BASE))
        self.assertEqual(router.installer.missing(BASE), ["image"])
        self.assertEqual(router.installer.missing(DEP), ["recipe", "image", "weights"])
        st = router.mgr.status()["models"]
        self.assertEqual((st[BASE]["installed"], st[DEP]["missing"]), (False, ["recipe", "image", "weights"]))

    async def test_failed_check_is_not_installed_true(self):
        async def broken(cmd, **kw):
            return 1, "nsenter: permission denied"
        router.host_run = broken
        with self.assertRaises(RuntimeError):
            await router.installer.check(BASE)
        self.assertIsNone(router.installer.installed(BASE))

    async def test_install_with_requirement(self):
        host = self.use(Host({BASE: missing_all(), DEP: missing_all()}))
        job = await self.install(DEP)
        self.assertEqual(job["state"], "done", job)
        self.assertEqual([(n, s) for n, s, _ in host.cmds],
                         [(BASE, "recipe"), (BASE, "image"),            # requirement: no weights
                          (DEP, "recipe"), (DEP, "image"), (DEP, "prebuild"), (DEP, "weights")])
        clone = host.cmds[0][2]
        self.assertIn("git clone -b local-patches https://example.invalid/base", clone)
        self.assertIn("git clone -b main https://example.invalid/dep", host.cmds[2][2])
        self.assertTrue(host.cmds[-1][2].endswith("./dl"))                 # weights_install, not --follow
        self.assertTrue(router.installer.installed(DEP))

    async def test_only_missing_parts(self):
        host = self.use(Host({BASE: {"recipe": True, "image": True, "weights": False},
                              DEP: {"recipe": True, "image": True, "weights": False}}))
        job = await self.install(DEP)
        self.assertEqual(job["state"], "done", job)
        self.assertEqual([(n, s) for n, s, _ in host.cmds], [(DEP, "weights")])
        self.assertEqual(job["results"][BASE], "present")

    async def test_failure_stops(self):
        host = self.use(Host({BASE: missing_all(), DEP: missing_all()}, fail={(BASE, "image")}))
        job = await self.install(DEP)
        self.assertEqual(job["state"], "failed")
        self.assertEqual(job["results"][f"{BASE}: image"], "failed (exit 1)")
        self.assertEqual([(n, s) for n, s, _ in host.cmds], [(BASE, "recipe"), (BASE, "image")])

    async def test_prebuild_failure_is_not_fatal(self):
        self.use(Host({BASE: {"recipe": True, "image": True, "weights": True}, DEP: missing_all()},
                      fail={(DEP, "prebuild")}))
        job = await self.install(DEP)
        self.assertEqual(job["state"], "done", job)
        self.assertIn("first start compiles", job["results"][f"{DEP}: prebuild"])

    async def test_not_installed_409(self):
        self.use(Host({BASE: {"recipe": True, "image": True, "weights": True}, DEP: missing_all()}))
        await router.installer.check_all()
        r = await router.not_installed(DEP)
        self.assertEqual(r.status, 409)
        body = json.loads(r.body)
        self.assertIn("not installed: POST /router/install", body["error"]["message"])
        self.assertEqual(body["error"]["missing"], ["recipe", "image", "weights"])
        self.assertIsNone(await router.not_installed(BASE))
        # "any" with nothing loaded: last used is not installed -> the installed one
        router.mgr.last_used = DEP
        self.assertEqual(router.mgr.any_target(), BASE)
        self.assertIsNone(await router.not_installed(None))

    async def test_installed_by_hand_is_noticed(self):
        host = self.use(Host({BASE: missing_all(), DEP: missing_all()}))
        await router.installer.check_all()
        host.have[BASE] = {"recipe": True, "image": True, "weights": True}
        self.assertIsNone(await router.not_installed(BASE))


if __name__ == "__main__":
    unittest.main()
