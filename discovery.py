"""Path discovery, validation, and preflight for ComfyUI-MVSAM3D-Unofficial.

Deliberately ComfyUI-free at module level (no ``folder_paths`` import) so every
function is unit-testable with any CPython. All "where is X on this machine"
knowledge lives here; ``nodes.py`` only calls ``resolve_*()`` and surfaces the
error messages to the ComfyUI user.

Resolution order for every external path:
    explicit node input  >  environment variable  >  discovery.
An empty string means "auto".
"""

import os
import re
import sys
from pathlib import Path

PACK_ROOT = Path(__file__).resolve().parent
VENDOR_REPO = PACK_ROOT / "vendor" / "MV-SAM3D"
VENDOR_REPO_URL = "https://github.com/devinli123/MV-SAM3D.git"
# The upstream commit this wrapper is validated against. setup_env.py fetches it;
# the submodule in .gitmodules is pinned to the same commit.
VENDOR_PINNED_COMMIT = "abb04b5e8af5bc33b0265bdf19937e76bbb6bcdd"

# The prebuilt pixi env provisioned by ComfyUI-SAM3DObjects (comfy-env).
PIXI_ENV_NAME = "sam3dobjects-nodes"

ENV_PIXI_PYTHON = "MVSAM3D_PIXI_PYTHON"
ENV_REPO = "MVSAM3D_REPO"
ENV_PIPELINE_YAML = "MVSAM3D_PIPELINE_YAML"

# Conservative allowlist for names that become filesystem path components
# (scene_name, object_name, filename_prefix). No separators, no leading dot,
# no '..' — blocks traversal out of the ComfyUI output directory.
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def safe_name(value: str, label: str) -> str:
    """Validate a node STRING input used as a path component. Raises ValueError."""
    v = (value or "").strip()
    if not v or not _SAFE_NAME.match(v) or ".." in v:
        raise ValueError(
            f"{label} must contain only letters, digits, '.', '_', '-' "
            f"(no path separators, no leading '.', no '..'); got {value!r}."
        )
    return v


def env_root_of_python(python_path) -> Path:
    """Environment root for an interpreter path.

    Windows conda/pixi layout: <env>/python.exe        -> parent
    POSIX layout:              <env>/bin/python        -> parent.parent
    venv on Windows:           <env>/Scripts/python.exe-> parent.parent
    """
    p = Path(python_path).resolve()
    if p.parent.name.lower() in ("bin", "scripts"):
        return p.parent.parent
    return p.parent


def _candidate_workspaces():
    """Yield possible comfy-env pixi workspace roots (where .pixi/envs lives)."""
    root = os.environ.get("COMFY_ENV_ROOT")
    if root:
        yield Path(root)
    # comfy-env knows its own workspace; it is importable in the ComfyUI host
    # python whenever ComfyUI-SAM3DObjects is installed.
    try:
        import comfy_env  # type: ignore

        for attr in ("get_workspace_dir",):
            fn = getattr(comfy_env, attr, None)
            if callable(fn):
                yield Path(fn())
                break
        else:
            from comfy_env.environment.cache import get_workspace_dir  # type: ignore

            yield Path(get_workspace_dir())
    except Exception:
        pass
    if sys.platform == "win32":
        local = os.environ.get("LOCALAPPDATA")
        if local:
            yield Path(local) / "Programs" / "comfy-env"


def _env_python(env_dir: Path) -> Path:
    if sys.platform == "win32":
        return env_dir / "python.exe"
    return env_dir / "bin" / "python"


def resolve_pixi_python(override: str = "") -> str:
    """Locate the python interpreter of the prebuilt ``sam3dobjects-nodes`` env."""
    tried = []
    for source, value in (
        ("node input 'pixi_python'", (override or "").strip()),
        (f"env var {ENV_PIXI_PYTHON}", os.environ.get(ENV_PIXI_PYTHON, "").strip()),
    ):
        if value:
            p = Path(value)
            if not p.is_file():
                raise FileNotFoundError(
                    f"pixi python from {source} does not exist: {value}"
                )
            if p.name.lower() not in ("python.exe", "python", "python3"):
                raise ValueError(
                    f"pixi python from {source} is not a python interpreter: {value}"
                )
            return str(p)

    for ws in _candidate_workspaces():
        cand = _env_python(ws / ".pixi" / "envs" / PIXI_ENV_NAME)
        tried.append(str(cand))
        if cand.is_file():
            return str(cand)

    raise FileNotFoundError(
        "Could not find the 'sam3dobjects-nodes' pixi environment python.\n"
        f"Tried: {tried or '(no comfy-env workspace found)'}\n"
        "This pack reuses the environment built by ComfyUI-SAM3DObjects — install "
        "and run that pack once first. Then either set the node's 'pixi_python' "
        f"input or the {ENV_PIXI_PYTHON} environment variable to the env's python. "
        "See README 'Install'."
    )


def resolve_repo_root(override: str = "") -> str:
    """Locate the official MV-SAM3D repo (the vendor/ submodule by default)."""
    for source, value in (
        ("node input 'repo_root'", (override or "").strip()),
        (f"env var {ENV_REPO}", os.environ.get(ENV_REPO, "").strip()),
        ("bundled submodule vendor/MV-SAM3D", str(VENDOR_REPO)),
    ):
        if not value:
            continue
        root = Path(value)
        if (root / "sam3d_objects").is_dir() and (
            root / "notebook" / "load_images_and_masks.py"
        ).is_file():
            return str(root)
        if source.startswith("bundled"):
            raise FileNotFoundError(
                f"The MV-SAM3D submodule is not initialized at {root}.\n"
                "Fix with ONE of:\n"
                "  * git submodule update --init   (inside this pack's directory)\n"
                "  * python scripts/setup_env.py   (fetches it even without git metadata)\n"
                f"  * set the node's 'repo_root' input or {ENV_REPO} to an existing "
                f"clone of {VENDOR_REPO_URL} at commit {VENDOR_PINNED_COMMIT[:12]}"
            )
        raise FileNotFoundError(
            f"repo_root from {source} is not an MV-SAM3D checkout "
            f"(missing sam3d_objects/ or notebook/load_images_and_masks.py): {value}"
        )
    raise AssertionError("unreachable")


def resolve_pipeline_yaml(override: str = "", models_dir: str = "") -> str:
    """Locate the SAM 3D Objects ``pipeline.yaml`` (ships with the model weights)."""
    tried = []
    for source, value in (
        ("node input 'pipeline_yaml'", (override or "").strip()),
        (f"env var {ENV_PIPELINE_YAML}", os.environ.get(ENV_PIPELINE_YAML, "").strip()),
    ):
        if value:
            if not Path(value).is_file():
                raise FileNotFoundError(
                    f"pipeline_yaml from {source} does not exist: {value}"
                )
            return value

    if models_dir:
        cand = Path(models_dir) / "sam3dobjects" / "pipeline.yaml"
        tried.append(str(cand))
        if cand.is_file():
            return str(cand)

    raise FileNotFoundError(
        "Could not find the SAM 3D Objects pipeline.yaml.\n"
        f"Tried: {tried}\n"
        "It is downloaded together with the model weights by ComfyUI-SAM3DObjects "
        "into <ComfyUI>/models/sam3dobjects/. Run that pack once so the weights "
        "exist, or set the node's 'pipeline_yaml' input or the "
        f"{ENV_PIPELINE_YAML} environment variable. See README 'Install'."
    )
