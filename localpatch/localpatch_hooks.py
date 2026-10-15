"""Patch definitions: {module name: function(module)} (vLLM and TensorFold containers)."""
import functools
import hashlib
import inspect
import os
import sys


def _log(msg):
    # only once per process type is enough; workers are separate processes
    print(f"[localpatch] {msg}", file=sys.stderr, flush=True)


def _fingerprint(func):
    raw = func.__func__ if isinstance(func, (staticmethod, classmethod)) else func
    return hashlib.sha256(inspect.getsource(raw).encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# MoE expert mapping cache
#
# RoutedExperts.load_weights() calls get_expert_mapping() -> build_expert_params_mapping()
# on every call, and the weight loader calls load_weights() many times with small
# batches. The mapping is a pure function of its arguments (expert counts and weight
# names), so it is rebuilt identically each time: ~30% of Flash Next's weight-loading
# CPU time (py-spy, 2026-09-28). Memoize it by its arguments.
#
# Tested fingerprints of build_expert_params_mapping:
#   8a360b2ea924  vllm/vllm-openai:qwen38-flash-next
#   d0637220c2d5  <35B NVFP4 recipe vLLM image> (Qwen3.6 35B)
_MOE_KNOWN = {"8a360b2ea924", "d0637220c2d5"}


def _patch_moe_mapping(module):
    cls = getattr(module, "RoutedExperts", None)
    target = cls.__dict__.get("build_expert_params_mapping") if cls else None
    if not isinstance(target, staticmethod):
        _log("moe-mapping-cache inactive: RoutedExperts.build_expert_params_mapping not found as expected")
        return
    fp = _fingerprint(target)
    if fp not in _MOE_KNOWN:
        _log(f"moe-mapping-cache inactive: code changed (fingerprint {fp}); loading normally")
        return
    orig = target.__func__
    sig = inspect.signature(orig)
    cache = {}

    @functools.wraps(orig)
    def cached(*args, **kwargs):
        try:
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            key = tuple(bound.arguments.items())
            hash(key)
        except TypeError:                       # unhashable argument: don't cache
            return orig(*args, **kwargs)
        hit = cache.get(key)
        if hit is None:
            hit = cache[key] = orig(*args, **kwargs)
        return list(hit)                        # fresh list: callers may modify it

    cls.build_expert_params_mapping = staticmethod(cached)
    if os.environ.get("LOCALPATCH_QUIET") != "1":
        _log(f"moe-mapping-cache active (fingerprint {fp}, pid {os.getpid()})")


# ---------------------------------------------------------------------------
# Draft (MTP) model: read only the checkpoint files it actually uses
#
# The speculative-decoding draft model is loaded in a second pass that iterates *all*
# 35 checkpoint files (~106 GB, 55 s) although only 2 contain tensors it keeps (its own
# name filter, _remap_mtp_weight_name, returns None for everything else). Using that
# same filter + the checkpoint index, skip files with no wanted tensor. The model still
# receives exactly the tensors it would have kept; if the filter, the index or the
# loader's return shape is missing/unexpected, nothing is filtered.
def _flash_mtp_keep(model_module):
    # Flash Next: the model's own module-level filter decides (follows upstream changes)
    keep = getattr(model_module, "_remap_mtp_weight_name", None)
    return keep if callable(keep) else None


# Qwen3.5/3.6 MTP (35B): the filter is inline in Qwen3_5MTP.load_weights():
#   keep "mtp.*", anything with "embed_tokens" or "lm_head", drop the rest.
# Mirror it only while that method is exactly the version it was read from.
_QWEN35_MTP_KNOWN = {"8528d343ea36"}


def _qwen35_mtp_keep(model_module):
    base = getattr(model_module, "Qwen3_5MTP", None)
    lw = base.__dict__.get("load_weights") if base else None
    if lw is None or _fingerprint(lw) not in _QWEN35_MTP_KNOWN:
        _log("draft-file-filter: Qwen3_5MTP.load_weights changed; not filtering its files")
        return None
    return lambda n: n.startswith("mtp.") or "embed_tokens" in n or "lm_head" in n


_DRAFT_FILTERS = {
    # draft model class -> factory(model module) returning a "keep this tensor?" predicate
    "Qwen3_8FlashNextMTP": _flash_mtp_keep,
    "Qwen3_5MoeMTP": _qwen35_mtp_keep,
    "Qwen3_5MTP": _qwen35_mtp_keep,
}


def _patch_default_loader(module):
    import json

    DL = getattr(module, "DefaultModelLoader", None)
    if DL is None or not callable(getattr(DL, "_prepare_weights", None)) \
            or not callable(getattr(DL, "load_weights", None)):
        _log("draft-file-filter inactive: DefaultModelLoader API changed")
        return
    orig_load, orig_prepare = DL.load_weights, DL._prepare_weights

    @functools.wraps(orig_load)
    def load_weights(self, model, model_config, *args, **kwargs):
        factory = _DRAFT_FILTERS.get(type(model).__name__)
        keep = factory(sys.modules.get(type(model).__module__)) if factory else None
        self._lp_keep = keep if callable(keep) else None
        try:
            return orig_load(self, model, model_config, *args, **kwargs)
        finally:
            self._lp_keep = None

    @functools.wraps(orig_prepare)
    def _prepare_weights(self, *args, **kwargs):
        result = orig_prepare(self, *args, **kwargs)
        keep = getattr(self, "_lp_keep", None)
        if not keep or not (isinstance(result, tuple) and len(result) == 3):
            return result
        folder, files, use_safetensors = result
        index = os.path.join(folder, "model.safetensors.index.json")
        if not use_safetensors or not os.path.exists(index):
            return result
        try:
            weight_map = json.load(open(index))["weight_map"]
            wanted = {f for n, f in weight_map.items() if keep(n) not in (None, False)}
            indexed = set(weight_map.values())
            kept = [p for p in files
                    if os.path.basename(p) in wanted or os.path.basename(p) not in indexed]
        except Exception as e:
            _log(f"draft-file-filter skipped: {e}")
            return result
        if not kept:
            return result
        _log(f"draft-file-filter: reading {len(kept)} of {len(files)} checkpoint files for the draft model")
        return folder, kept, use_safetensors

    DL.load_weights, DL._prepare_weights = load_weights, _prepare_weights
    _log(f"draft-file-filter active (pid {os.getpid()})")


# ---------------------------------------------------------------------------
# Skip the multi-modal warmup at server start
#
# BaseRenderer.warmup() runs the image processor once on a *maximum-size* dummy input
# (sized for the full 262k context): ~12.7 s on every start, only to shave first-image
# latency. Running it in the background would race with real requests on the processor
# cache (vLLM notes this), so skip it instead. Re-enable with LOCALPATCH_MM_WARMUP=1.
_MM_WARMUP_KNOWN = {"f8b988352d7f"}


def _patch_mm_warmup(module):
    if os.environ.get("LOCALPATCH_MM_WARMUP") == "1":
        return
    cls = getattr(module, "BaseRenderer", None)
    target = cls.__dict__.get("_warmup_mm_processor") if cls else None
    if target is None:
        _log("mm-warmup-skip inactive: BaseRenderer._warmup_mm_processor not found")
        return
    fp = _fingerprint(target)
    if fp not in _MM_WARMUP_KNOWN:
        _log(f"mm-warmup-skip inactive: code changed (fingerprint {fp}); warming up normally")
        return

    def _warmup_mm_processor(self, processor, *, log_prefix):
        return None

    cls._warmup_mm_processor = _warmup_mm_processor
    _log(f"mm-warmup-skip active (pid {os.getpid()})")


# ---------------------------------------------------------------------------
# TensorFold (Flash Next CUDA engine): no allocator flush after every layer
#
# tensorfold.families.qwen4_exp.cuda.weights.load() calls torch.cuda.empty_cache()
# after each of the 48 layers. On GB10's unified memory that hands every freed staging
# block back to the driver and the next layer maps it again; a stack-sampling profile of
# a start (2026-10-01) put ~23% of the load in that call. Here the flushes during load()
# are skipped and the cache is emptied once when load() returns, so the weights, the
# memory the server keeps and the output are unchanged; only the transient cache during
# loading differs (min MemAvailable during load unchanged, ~10-11 GiB).
# Measured 2026-10-01 (3 restarts each, other patches on): 50-53 s -> 43-44 s.
# Disable: LOCALPATCH_TF_FLUSH=1.
#
# Tested fingerprints of weights.load:
#   10c8d8d6f0cc  tensorfold 0.3.6.3 + recipe patches c1f5d72f8d16
_TF_LOAD_KNOWN = {"10c8d8d6f0cc"}


def _patch_tf_load_flush(module):
    if os.environ.get("LOCALPATCH_TF_FLUSH") == "1":
        return
    target = getattr(module, "load", None)
    if not callable(target):
        _log("tf-load-no-flush inactive: weights.load not found")
        return
    fp = _fingerprint(target)
    if fp not in _TF_LOAD_KNOWN:
        _log(f"tf-load-no-flush inactive: code changed (fingerprint {fp}); loading normally")
        return
    torch = module.torch

    @functools.wraps(target)
    def load(*args, **kwargs):
        real = torch.cuda.empty_cache
        torch.cuda.empty_cache = lambda: None
        try:
            return target(*args, **kwargs)
        finally:
            torch.cuda.empty_cache = real
            real()

    module.load = load
    _log(f"tf-load-no-flush active (fingerprint {fp}, pid {os.getpid()})")


# ---------------------------------------------------------------------------
# TensorFold (Flash Next CUDA engine): checkpoint reads streamed through pinned buffers
#
# reader._Reader.get() reads each tensor into pageable host memory and then copies it
# with .to("cuda"), a synchronous copy that also waits for the GPU work queued by the
# previous layer; release() then drops the page cache of the shards read (fadvise
# DONTNEED) on the loading thread. Profile (2026-10-01): ~20% of the load in .to(),
# ~10% in release(). Here get() reads the same bytes in 64 MiB pieces into two reused
# pinned buffers and copies each to the GPU asynchronously (the next piece is read while
# the previous one copies), and release() does the same fadvise on a background thread.
# The tensors (bytes, dtype, shape, device) are exactly what get() returned before.
# Measured 2026-10-01 (3 restarts each, other patches on): 47 s -> 43-44 s.
# (O_DIRECT reads and a read-ahead of the vision tower were tried on top: no gain, removed.)
# Disable: LOCALPATCH_TF_READER=0.
#
# Tested fingerprints of (_Reader.get, _Reader.release):
#   (4d4e86eab461, a8e30af9ff2c)  tensorfold 0.3.6.3 + recipe patches c1f5d72f8d16
_TF_READER_KNOWN = {("4d4e86eab461", "a8e30af9ff2c")}


def _patch_tf_reader(module):
    if os.environ.get("LOCALPATCH_TF_READER") == "0":
        return
    cls = getattr(module, "_Reader", None)
    if cls is None or not callable(getattr(cls, "get", None)) or not callable(getattr(cls, "release", None)):
        _log("tf-pinned-reader inactive: _Reader.get/release not found")
        return
    fp = (_fingerprint(cls.get), _fingerprint(cls.release))
    if fp not in _TF_READER_KNOWN:
        _log(f"tf-pinned-reader inactive: code changed (fingerprints {fp}); loading normally")
        return
    import concurrent.futures
    torch = module.torch
    orig_get, orig_release = cls.get, cls.release
    piece = 64 << 20
    ring = []                                       # [(pinned buffer, event)] x 2, made on first use
    releaser = concurrent.futures.ThreadPoolExecutor(1, thread_name_prefix="localpatch-release")

    def get(self, name):
        if str(self.device).split(":")[0] != "cuda":
            return orig_get(self, name)
        shard = self.where[name]
        base, header = self._header(shard)
        entry = header[name]
        begin, end = entry["data_offsets"]
        size = end - begin
        if not ring:
            ring.extend((torch.empty((piece,), dtype=torch.uint8, pin_memory=True), torch.cuda.Event())
                        for _ in range(2))
        out = torch.empty((size,), dtype=torch.uint8, device=self.device)
        with open(self.dir / shard, "rb", buffering=0) as f:
            f.seek(base + begin)
            at, k = 0, 0
            while at < size:
                n = min(piece, size - at)
                buf, done = ring[k]
                done.synchronize()                  # its previous copy has finished
                view = memoryview(buf.numpy())
                got = 0
                while got < n:
                    r = f.readinto(view[got:n])
                    if not r:
                        raise IOError(f"short read of {name}")
                    got += r
                out[at:at + n].copy_(buf[:n], non_blocking=True)
                done.record()
                at += n
                k ^= 1
        self.touched.add(shard)
        return out.view(module._DT[entry["dtype"]]).reshape(entry["shape"])

    def release(self):
        shards, self.touched = set(self.touched), set()
        if shards:
            probe = object.__new__(cls)
            probe.dir, probe.touched = self.dir, shards
            releaser.submit(orig_release, probe)

    cls.get, cls.release = get, release
    _log(f"tf-pinned-reader active (fingerprints {fp}, pid {os.getpid()})")


# ---------------------------------------------------------------------------
# TensorFold: expert-weight nibble reordering in 2 bit operations instead of 8 int64 passes
#
# tensorfold.cuda.experts._nibbles() reorders the 8 nibbles of every 4-bit expert word
# (i0..i7 -> i0 i2 i4 i6 i1 i3 i5 i7) by widening to int64 and looping over the nibbles;
# it runs on ~60 GiB of expert words at every start. The same permutation is an
# "unshuffle" done with two masked delta swaps on int32 (Hacker's Delight 7-2), every
# intermediate staying inside 32 bits. Verified bit-identical against the original on
# random words, edge values and real layers (2026-10-01); int32 input only, anything
# else goes to the original. Measured (3 restarts each, other patches on): 81 s -> 43-44 s.
# Disable: LOCALPATCH_TF_NIBBLES=0.
#
# Tested fingerprints of experts._nibbles:
#   32a0c782892a  tensorfold 0.3.6.3 + recipe patches c1f5d72f8d16
_TF_NIBBLES_KNOWN = {"32a0c782892a"}


def _patch_tf_nibbles(module):
    if os.environ.get("LOCALPATCH_TF_NIBBLES") == "0":
        return
    target = getattr(module, "_nibbles", None)
    if not callable(target):
        _log("tf-fast-nibbles inactive: experts._nibbles not found")
        return
    fp = _fingerprint(target)
    if fp not in _TF_NIBBLES_KNOWN:
        _log(f"tf-fast-nibbles inactive: code changed (fingerprint {fp}); packing normally")
        return
    torch = module.torch

    @functools.wraps(target)
    def _nibbles(w):
        if w.dtype != torch.int32:
            return target(w)
        t = ((w >> 4) ^ w) & 0x00F000F0
        x = w ^ t ^ (t << 4)
        t = ((x >> 8) ^ x) & 0x0000FF00
        return x ^ t ^ (t << 8)

    module._nibbles = _nibbles
    _log(f"tf-fast-nibbles active (fingerprint {fp}, pid {os.getpid()})")


PATCHES = {
    "tensorfold.cuda.experts": _patch_tf_nibbles,
    "tensorfold.families.qwen4_exp.cuda.reader": _patch_tf_reader,
    "tensorfold.families.qwen4_exp.cuda.weights": _patch_tf_load_flush,
    "vllm.model_executor.layers.fused_moe.routed_experts": _patch_moe_mapping,
    "vllm.model_executor.model_loader.default_loader": _patch_default_loader,
    "vllm.renderers.base": _patch_mm_warmup,
}
