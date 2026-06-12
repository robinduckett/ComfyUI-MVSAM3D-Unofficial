"""Install hook — ComfyUI-Manager runs this automatically after installing the pack.

Fetches the PINNED MV-SAM3D vendor repo (registry archives have no git metadata,
so `git submodule update` alone can't do it). The pixi-env extras still require a
one-time `python scripts/setup_env.py` (they need the ComfyUI-SAM3DObjects env to
exist first).

Never hard-fails: with no git or no network, the node's preflight raises a clear
error with instructions at first use instead of breaking the whole install.
"""

import importlib.util
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    try:
        setup_env = _load("mvsam3d_setup_env", HERE / "scripts" / "setup_env.py")
        setup_env.fetch_vendor()
        print("[MV-SAM3D] vendor repo ready. Run 'python scripts/setup_env.py' once "
              "to provision the pixi-env extras (see README 'Install').")
    except Exception:
        traceback.print_exc()
        print("[MV-SAM3D] vendor fetch failed (no git / offline?). The nodes will "
              "explain how to fix this on first use; or run "
              "'python scripts/setup_env.py' manually.")

    # Tell the user NOW (not at first queue) if the prerequisite is missing.
    try:
        setup_env.resolve_pixi_python("")
        print("[MV-SAM3D] prerequisite check: sam3dobjects-nodes env found.")
    except Exception:
        print(
            "[MV-SAM3D] " + "=" * 64 + "\n"
            "[MV-SAM3D] PREREQUISITE MISSING: the 'sam3dobjects-nodes' environment\n"
            "[MV-SAM3D] was not found. This pack is an add-on to ComfyUI-SAM3DObjects\n"
            "[MV-SAM3D] and reuses its environment and model weights.\n"
            "[MV-SAM3D]   1. Install ComfyUI-SAM3DObjects and generate one mesh with it.\n"
            "[MV-SAM3D]   2. Run: python scripts/setup_env.py   (in this pack's folder)\n"
            "[MV-SAM3D] Until then the MV-SAM3D nodes will error with these same steps.\n"
            "[MV-SAM3D] " + "=" * 64
        )


if __name__ == "__main__":
    main()
    sys.exit(0)
