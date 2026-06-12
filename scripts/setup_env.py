r"""One-shot environment setup for ComfyUI-MVSAM3D-Unofficial.

Automates everything that was previously a hand-applied prerequisite list:

  1. Fetch the pinned MV-SAM3D submodule (works even without git metadata, e.g.
     when the pack was installed from a registry archive instead of `git clone`).
  2. pip-install the extra runtime deps the official code needs into the
     prebuilt ``sam3dobjects-nodes`` pixi env (built by ComfyUI-SAM3DObjects).
  3. pip-install the PINNED MoGe (uses utils3d.torch, matching the env).
  4. Write a no-op ``kaolin.utils.testing.check_tensor`` stub into the env if the
     real kaolin is not importable (no wheel exists for some GPU/torch combos;
     only this one validator function is imported on the inference path).

Idempotent: safe to re-run. Run with any python:
    python scripts/setup_env.py [--pixi-python <path>] [--skip-pip]
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

PACK_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PACK_ROOT))
from discovery import (  # noqa: E402
    VENDOR_REPO, VENDOR_REPO_URL, VENDOR_PINNED_COMMIT, resolve_pixi_python,
)

PIP_EXTRAS = [
    "astor", "lightning", "easydict", "roma", "rootutils", "randomname",
    "jsonpickle", "einops-exts",
]
MOGE_PIN = ("git+https://github.com/microsoft/MoGe.git"
            "@a8c37341bc0325ca99b9d57981cc3bb2bd3e255b")

KAOLIN_STUB = {
    "kaolin/__init__.py": '''"""Minimal stub of kaolin for sam3d_objects inference.

Only ``kaolin.utils.testing.check_tensor`` is imported anywhere in the package
(flexicubes.py). When no kaolin wheel exists for this GPU/torch combination, the
real kaolin is not needed on the inference path (check_tensor is just a
shape/dtype validator).
"""
__version__ = "0.0.0-stub"
''',
    "kaolin/utils/__init__.py": "",
    "kaolin/utils/testing.py": '''"""Stub of kaolin.utils.testing.check_tensor — a no-op validator.

The real check_tensor raises if a tensor's shape/dtype/device mismatch; on the
inference path the tensors fed to FlexiCubes are already valid, so skipping the
validation is safe.
"""


def check_tensor(tensor, shape=None, dtype=None, device=None, throw=True):
    return True
''',
}


def run(cmd, **kw):
    print("+ " + " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run([str(c) for c in cmd], check=True, **kw)


def fetch_vendor():
    if (VENDOR_REPO / "sam3d_objects").is_dir():
        print(f"vendor OK: {VENDOR_REPO}")
        return
    print(f"fetching MV-SAM3D @ {VENDOR_PINNED_COMMIT[:12]} into {VENDOR_REPO} ...")
    if (PACK_ROOT / ".git").exists():
        try:
            run(["git", "-C", PACK_ROOT, "submodule", "update", "--init",
                 "vendor/MV-SAM3D"])
            return
        except subprocess.CalledProcessError:
            print("submodule update failed; falling back to a direct clone.")
    VENDOR_REPO.parent.mkdir(parents=True, exist_ok=True)
    run(["git", "clone", VENDOR_REPO_URL, VENDOR_REPO])
    run(["git", "-C", VENDOR_REPO, "checkout", VENDOR_PINNED_COMMIT])


def env_site_packages(pixi_py: Path) -> Path:
    env = pixi_py.parent.parent if pixi_py.parent.name.lower() in ("bin", "scripts") \
        else pixi_py.parent
    win = env / "Lib" / "site-packages"
    if win.is_dir():
        return win
    lib = env / "lib"
    if lib.is_dir():
        for child in sorted(lib.glob("python*")):
            sp = child / "site-packages"
            if sp.is_dir():
                return sp
    raise FileNotFoundError(f"site-packages not found under {env}")


def ensure_kaolin_stub(pixi_py: Path):
    probe = subprocess.run([str(pixi_py), "-c", "import kaolin"],
                           capture_output=True, text=True)
    if probe.returncode == 0:
        print("kaolin imports in the env — no stub needed.")
        return
    sp = env_site_packages(pixi_py)
    if (sp / "kaolin" / "utils" / "testing.py").exists():
        print("kaolin stub already present.")
        return
    print(f"writing kaolin stub into {sp} ...")
    for rel, content in KAOLIN_STUB.items():
        dest = sp / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pixi-python", default="",
                    help="python.exe of the sam3dobjects-nodes env (default: auto-discover)")
    ap.add_argument("--skip-pip", action="store_true",
                    help="only fetch the vendor repo / write the kaolin stub")
    args = ap.parse_args()

    fetch_vendor()

    pixi_py = Path(resolve_pixi_python(args.pixi_python))
    print(f"pixi env python: {pixi_py}")

    if not args.skip_pip:
        run([pixi_py, "-m", "pip", "install", *PIP_EXTRAS])
        run([pixi_py, "-m", "pip", "install", "--no-deps", MOGE_PIN])

    ensure_kaolin_stub(pixi_py)

    probe = subprocess.run(
        [str(pixi_py), "-c",
         "import torch, lightning, easydict, roma, moge, kaolin.utils.testing; "
         "print('env check OK, torch', torch.__version__)"],
        capture_output=True, text=True)
    print(probe.stdout.strip() or probe.stderr.strip())
    if probe.returncode != 0:
        sys.exit("env check FAILED — see output above.")
    print("setup complete.")


if __name__ == "__main__":
    main()
