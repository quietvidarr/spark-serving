"""Local, update-safe speed patches for the model containers (vLLM and TensorFold).

Mounted read-only at /opt/localpatch and enabled with PYTHONPATH=/opt/localpatch
(set in each recipe's git-ignored .env / .env.local). Python imports this file at
start-up in every process (API server, engine, workers).

Each patch waits for its target module to be imported, checks the target code's
fingerprint against versions it was tested on, and only then applies itself.
If a recipe update changed the code, the patch logs "inactive" and does nothing,
so the model still loads normally (just without the speed-up).

Disable everything with LOCALPATCH_DISABLE=1.
"""
import importlib.abc
import importlib.util
import os
import sys

# Keep the image's own sitecustomize (Ubuntu ships one) working.
for _p in ("/usr/lib/python3.12/sitecustomize.py",):
    if os.path.exists(_p) and os.path.dirname(_p) != os.path.dirname(__file__):
        try:
            _spec = importlib.util.spec_from_file_location("_image_sitecustomize", _p)
            _spec.loader.exec_module(importlib.util.module_from_spec(_spec))
        except Exception:
            pass

if os.environ.get("LOCALPATCH_DISABLE") != "1":
    sys.path.insert(0, os.path.dirname(__file__))
    from localpatch_hooks import PATCHES  # noqa: E402

    class _PatchFinder(importlib.abc.MetaPathFinder):
        """Runs a patch right after its target module finishes importing."""

        def find_spec(self, name, path, target=None):
            patch = PATCHES.get(name)
            if patch is None:
                return None
            sys.meta_path.remove(self)          # let the normal finders locate it
            try:
                spec = importlib.util.find_spec(name)
            finally:
                sys.meta_path.insert(0, self)
            if spec is None or spec.loader is None:
                return None
            orig_exec = spec.loader.exec_module

            def exec_module(module, _orig=orig_exec, _patch=patch):
                _orig(module)
                try:
                    _patch(module)
                except Exception as e:  # never break the import
                    print(f"[localpatch] {name}: patch error, left unpatched: {e}", file=sys.stderr)

            spec.loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _PatchFinder())
