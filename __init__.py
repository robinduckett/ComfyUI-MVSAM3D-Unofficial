"""ComfyUI-MVSAM3D-Unofficial — unofficial wrapper driving the official
devinli123/MV-SAM3D code (pinned vendor/ submodule), unmodified, via a subprocess
into the prebuilt ``sam3dobjects-nodes`` pixi env. See nodes.py for the rationale.
"""
if __package__:
    from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
else:
    # Imported without a package context (e.g. pytest collecting the repo root,
    # which is a Package because of this very file). ComfyUI always imports the
    # pack as a package, so registering no nodes here is correct.
    NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS = {}, {}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
