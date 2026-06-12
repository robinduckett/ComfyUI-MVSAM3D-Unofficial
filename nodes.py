r"""ComfyUI nodes for the official MV-SAM3D (devinli123/MV-SAM3D) — unofficial wrapper.

These run in ComfyUI's own python and orchestrate a subprocess into the prebuilt
``sam3dobjects-nodes`` pixi env (worker/mvsam3d_worker.py). Rationale: comfy-aimdo
keys its isolated env on the *plugin directory name*, so giving this pack its own
comfy-env.toml would build a fresh multi-GB env (recompiling the CUDA extensions)
instead of reusing the donor's validated stack. Shelling into the existing env
keeps ComfyUI's torch untouched. In-memory IMAGE/MASK is marshaled via a scene dir.

The fusion math is the upstream authors' code, unmodified, taken from the pinned
``vendor/MV-SAM3D`` submodule (see discovery.VENDOR_PINNED_COMMIT).
"""

import json
import os
import shutil
import subprocess
import time

import numpy as np
from PIL import Image

from .discovery import (
    safe_name,
    resolve_pixi_python,
    resolve_repo_root,
    resolve_pipeline_yaml,
)

HERE = os.path.dirname(os.path.abspath(__file__))
WORKER = os.path.join(HERE, "worker", "mvsam3d_worker.py")

# Keep the last N worker lines so a failure can show its actual cause inline
# instead of "see console".
_ERROR_TAIL_LINES = 30


def _to_uint8(t):
    a = np.clip(t.cpu().numpy() if hasattr(t, "cpu") else np.asarray(t), 0, 1)
    return (a * 255.0 + 0.5).astype(np.uint8)


def _mask_to_rgba(mask_u8: np.ndarray) -> Image.Image:
    """Official loader reads masks from the ALPHA channel of an RGBA file
    (load_images_and_masks.py:load_mask_from_rgba); a grayscale PNG would be
    silently replaced by an all-ones mask. So: white RGB, mask in alpha."""
    h, w = mask_u8.shape[:2]
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., :3] = 255
    rgba[..., 3] = mask_u8
    return Image.fromarray(rgba, mode="RGBA")


class MVSAM3DSceneFromDir:
    """Use an existing on-disk scene in the official layout — including the paper's
    own ``data/example`` scene from the submodule (leave ``scene_dir`` empty)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "scene_dir": ("STRING", {
                    "default": "",
                    "tooltip": "Scene directory in the official MV-SAM3D layout. Empty = the "
                               "paper's bundled data/example scene (8 views, stuffed_toy)."}),
                "mask_prompt": ("STRING", {
                    "default": "",
                    "tooltip": "Mask subfolder name (scene/<mask_prompt>/ with RGBA masks; "
                               "images in scene/images/). Empty + empty scene_dir = "
                               "'stuffed_toy'. Empty with a custom scene_dir = flat layout "
                               "(N.png + N_mask.png in one folder)."}),
            },
        }

    RETURN_TYPES = ("MVSAM3D_SCENE",)
    RETURN_NAMES = ("scene",)
    FUNCTION = "run"
    CATEGORY = "MV-SAM3D"

    def run(self, scene_dir, mask_prompt):
        scene_dir = (scene_dir or "").strip()
        mask_prompt = (mask_prompt or "").strip()
        if not scene_dir:
            repo = resolve_repo_root("")
            scene_dir = os.path.join(repo, "data", "example")
            if not mask_prompt:
                mask_prompt = "stuffed_toy"
        if not os.path.isdir(scene_dir):
            raise FileNotFoundError(f"scene_dir does not exist: {scene_dir}")
        if mask_prompt:
            img_dir = os.path.join(scene_dir, "images")
            mask_dir = os.path.join(scene_dir, mask_prompt)
            if not os.path.isdir(img_dir):
                raise FileNotFoundError(
                    f"official layout expects images at {img_dir} (missing)")
            if not os.path.isdir(mask_dir):
                raise FileNotFoundError(
                    f"mask subfolder '{mask_prompt}' not found at {mask_dir}")
            n = len([f for f in os.listdir(img_dir) if f.lower().endswith((".png", ".jpg"))])
        else:
            n = len([f for f in os.listdir(scene_dir)
                     if f.lower().endswith((".png", ".jpg")) and "_mask" not in f])
        if n == 0:
            raise ValueError(f"no view images found in {scene_dir}")
        print(f"[MV-SAM3D] scene: {scene_dir} ({n} views, mask_prompt={mask_prompt or None})")
        scene = {"scene_dir": scene_dir, "mask_prompt": mask_prompt or None,
                 "names": None, "n_views": n}
        return (scene,)


class MVSAM3DLoadViews:
    """Write an in-memory IMAGE + MASK batch to a scene dir in the layout the official
    loader expects: scene/images/{i}.png and scene/{object}/{i}_mask.png (RGBA masks)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": "Batch of N views of the SAME object (use Image Batch to "
                               "combine LoadImage outputs)."}),
                "masks": ("MASK", {
                    "tooltip": "Batch of N foreground masks, one per view, same order. "
                               "The official pipeline requires masks."}),
                "object_name": ("STRING", {
                    "default": "object",
                    "tooltip": "Mask subfolder name (letters/digits/._- only)."}),
                "scene_name": ("STRING", {
                    "default": "scene",
                    "tooltip": "Scene folder name under output/mvsam3d_scenes "
                               "(letters/digits/._- only). Recreated on every run."}),
            },
        }

    RETURN_TYPES = ("MVSAM3D_SCENE",)
    RETURN_NAMES = ("scene",)
    FUNCTION = "run"
    CATEGORY = "MV-SAM3D"

    def run(self, images, masks, object_name, scene_name):
        import folder_paths

        object_name = safe_name(object_name, "object_name")
        scene_name = safe_name(scene_name, "scene_name")

        n = images.shape[0]
        if masks.shape[0] < n:
            raise ValueError(
                f"got {n} images but only {masks.shape[0]} masks — every view needs "
                "a mask (the official pipeline skips views without one).")

        base = os.path.join(folder_paths.get_output_directory(), "mvsam3d_scenes", scene_name)
        # Recreate from scratch: stale files from a previous (larger) run would be
        # globbed by the official loader and silently fused in.
        if os.path.isdir(base):
            shutil.rmtree(base)
        img_dir = os.path.join(base, "images")
        mask_dir = os.path.join(base, object_name)
        os.makedirs(img_dir)
        os.makedirs(mask_dir)

        names = []
        for i in range(n):
            name = str(i + 1)  # 1-indexed, natural sort order of the official loader
            names.append(name)
            Image.fromarray(_to_uint8(images[i])).save(os.path.join(img_dir, f"{name}.png"))
            m = masks[i]
            _mask_to_rgba(_to_uint8(m)).save(os.path.join(mask_dir, f"{name}_mask.png"))
        print(f"[MV-SAM3D] wrote {n} views to {base}")
        scene = {"scene_dir": base, "mask_prompt": object_name, "names": names, "n_views": n}
        return (scene,)


class MVSAM3DRunMultiView:
    """Run the official ``InferencePipelinePointMap.run_multi_view`` in the prebuilt
    pixi env via a subprocess worker. Exposes the published weighting knobs."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "scene": ("MVSAM3D_SCENE", {
                    "tooltip": "From MV-SAM3D Load Views or Scene From Dir."}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xFFFFFFFF,
                                 "tooltip": "Random seed (paper example uses 42)."}),
                "stage1_steps": ("INT", {"default": 50, "min": 1, "max": 200,
                                         "tooltip": "Stage-1 (sparse structure) denoise steps. Paper: 50."}),
                "stage2_steps": ("INT", {"default": 25, "min": 1, "max": 200,
                                         "tooltip": "Stage-2 (SLAT) denoise steps. Paper: 25."}),
                "ss_weighting": ("BOOLEAN", {"default": True,
                                             "tooltip": "Stage-1 attention-entropy weighting (paper method)."}),
                "ss_entropy_layer": ("INT", {"default": 9, "min": 0, "max": 32,
                                             "tooltip": "Stage-1 cross-attn layer tapped for entropy. Reference: 9."}),
                "ss_entropy_alpha": ("FLOAT", {"default": 30.0, "min": 0.0, "max": 200.0,
                                               "tooltip": "Stage-1 entropy softmax temperature. Reference: 30."}),
                "stage2_weighting": ("BOOLEAN", {"default": True,
                                                 "tooltip": "Stage-2 per-view weighting on/off."}),
                "stage2_weight_source": (["entropy", "visibility", "mixed"], {
                    "default": "entropy",
                    "tooltip": "entropy = paper headline method. visibility/mixed need a "
                               "DA3 npz (da3_npz input)."}),
                "stage2_entropy_alpha": ("FLOAT", {"default": 30.0, "min": 0.0, "max": 200.0}),
                "stage2_visibility_alpha": ("FLOAT", {"default": 30.0, "min": 0.0, "max": 200.0}),
                "stage2_attention_layer": ("INT", {"default": 6, "min": 0, "max": 32,
                                                   "tooltip": "Stage-2 cross-attn layer tapped for entropy. Reference: 6."}),
                "stage2_attention_step": ("INT", {"default": 0, "min": 0, "max": 200}),
                "stage2_min_weight": ("FLOAT", {"default": 0.001, "min": 0.0, "max": 1.0, "step": 0.001}),
                "stage2_weight_combine_mode": (["average", "multiply"], {"default": "average"}),
                "decode_formats": ("STRING", {
                    "default": "gaussian,mesh",
                    "tooltip": "Comma list of outputs: gaussian (-> .ply splat), mesh (-> .glb)."}),
                "filename_prefix": ("STRING", {"default": "mvsam3d",
                                               "tooltip": "Output file prefix (letters/digits/._- only)."}),
            },
            "optional": {
                "offload": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Stage-wise CPU offloading so the run fits 16 GB VRAM "
                               "(measured 13.95 GB peak). Device placement only — the "
                               "fusion math is unchanged. Disable on >24 GB cards for "
                               "the fully-resident original behavior."}),
                "da3_npz": ("STRING", {
                    "default": "",
                    "tooltip": "Optional Depth-Anything-3 pointmap npz (pointmaps_sam3d + "
                               "image_files). Required for visibility/mixed weighting."}),
                "pixi_python": ("STRING", {
                    "default": "",
                    "tooltip": "Empty = auto-discover the sam3dobjects-nodes env python "
                               "(or set MVSAM3D_PIXI_PYTHON)."}),
                "repo_root": ("STRING", {
                    "default": "",
                    "tooltip": "Empty = bundled vendor/MV-SAM3D submodule "
                               "(or set MVSAM3D_REPO)."}),
                "pipeline_yaml": ("STRING", {
                    "default": "",
                    "tooltip": "Empty = auto-discover models/sam3dobjects/pipeline.yaml "
                               "(or set MVSAM3D_PIPELINE_YAML)."}),
            },
        }

    RETURN_TYPES = ("MVSAM3D_RESULT", "STRING", "STRING")
    RETURN_NAMES = ("result", "glb_path", "ply_path")
    FUNCTION = "run"
    CATEGORY = "MV-SAM3D"

    def run(self, scene, **kw):
        import folder_paths
        import comfy.model_management as mm

        prefix = safe_name(kw["filename_prefix"], "filename_prefix")

        # ---- preflight: resolve every external path BEFORE any GPU/scene work ----
        pixi_py = resolve_pixi_python(kw.get("pixi_python", ""))
        repo_root = resolve_repo_root(kw.get("repo_root", ""))
        pipeline_yaml = resolve_pipeline_yaml(
            kw.get("pipeline_yaml", ""), models_dir=folder_paths.models_dir)
        if not os.path.isfile(WORKER):
            raise FileNotFoundError(f"worker script missing: {WORKER}")
        da3 = (kw.get("da3_npz") or "").strip()
        if kw["stage2_weight_source"] in ("visibility", "mixed") and not da3:
            raise ValueError(
                f"stage2_weight_source={kw['stage2_weight_source']!r} requires the "
                "da3_npz input (a Depth-Anything-3 pointmap npz). Use 'entropy' otherwise.")
        if da3 and not os.path.isfile(da3):
            raise FileNotFoundError(f"da3_npz does not exist: {da3}")

        out_dir = os.path.join(folder_paths.get_output_directory(), "mvsam3d",
                               f"{prefix}_{time.strftime('%Y%m%d-%H%M%S')}_{os.getpid()}")
        os.makedirs(out_dir, exist_ok=True)

        job = {
            "repo_root": repo_root,
            "pipeline_yaml": pipeline_yaml,
            "scene_dir": scene["scene_dir"],
            "mask_prompt": scene.get("mask_prompt"),
            "image_names": ",".join(scene["names"]) if scene.get("names") else None,
            "output_dir": out_dir,
            "filename_prefix": prefix,
            "seed": kw["seed"],
            "stage1_steps": kw["stage1_steps"], "stage2_steps": kw["stage2_steps"],
            "decode_formats": kw["decode_formats"],
            "ss_weighting": kw["ss_weighting"], "ss_entropy_layer": kw["ss_entropy_layer"],
            "ss_entropy_alpha": kw["ss_entropy_alpha"],
            "stage2_weighting": kw["stage2_weighting"],
            "stage2_weight_source": kw["stage2_weight_source"],
            "stage2_entropy_alpha": kw["stage2_entropy_alpha"],
            "stage2_visibility_alpha": kw["stage2_visibility_alpha"],
            "stage2_attention_layer": kw["stage2_attention_layer"],
            "stage2_attention_step": kw["stage2_attention_step"],
            "stage2_min_weight": kw["stage2_min_weight"],
            "stage2_weight_combine_mode": kw["stage2_weight_combine_mode"],
            "offload": kw.get("offload", True),
            "da3_npz": da3 or None,
        }
        job_path = os.path.join(out_dir, "job.json")
        with open(job_path, "w", encoding="utf-8") as f:
            json.dump(job, f, indent=2)

        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        # NOTE: no CONDA_PREFIX/CUDA_HOME/HF_HOME here — the worker derives its env
        # root from its own sys.executable, and the user's HF cache settings win.

        print(f"[MV-SAM3D] launching worker: {pixi_py} {WORKER} {job_path}")
        tail = []
        oom = False
        proc = subprocess.Popen([pixi_py, WORKER, job_path], env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", errors="replace", bufsize=1)
        try:
            for line in proc.stdout:
                line = line.rstrip()
                print(f"[MV-SAM3D] {line}")
                tail.append(line)
                if len(tail) > _ERROR_TAIL_LINES:
                    tail.pop(0)
                if "CUDA out of memory" in line or "OutOfMemoryError" in line:
                    oom = True
                if mm.processing_interrupted():
                    proc.kill()
                    proc.wait()
                    mm.throw_exception_if_processing_interrupted()
        finally:
            if proc.stdout:
                proc.stdout.close()
        rc = proc.wait()

        res_json = os.path.join(out_dir, f"{prefix}_result.json")
        if rc != 0 or not os.path.exists(res_json):
            hint = ""
            if oom:
                hint = ("\nHint: CUDA out of memory — this pipeline peaks ~14 GB with "
                        "offload=True (more without). Close other GPU apps or reduce views.")
            raise RuntimeError(
                f"MV-SAM3D worker failed (rc={rc}).{hint}\nLast worker output:\n  "
                + "\n  ".join(tail[-_ERROR_TAIL_LINES:])
                + f"\nFull log dir: {out_dir}")
        with open(res_json, encoding="utf-8") as f:
            res = json.load(f)
        written = res.get("written", {})
        result = {"output_dir": out_dir, "glb_path": written.get("glb", ""),
                  "ply_path": written.get("ply", ""), "pose": written.get("pose", {}),
                  "peak_vram_gb": res.get("peak_vram_gb")}
        return (result, result["glb_path"], result["ply_path"])


class MVSAM3DExport:
    """Surface the result GLB/PLY path for downstream 3D-preview / save nodes."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "result": ("MVSAM3D_RESULT",),
            "which": (["glb", "ply"], {"default": "glb",
                                       "tooltip": "glb = mesh, ply = Gaussian splat."}),
        }}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("path",)
    FUNCTION = "run"
    CATEGORY = "MV-SAM3D"
    OUTPUT_NODE = True

    def run(self, result, which):
        path = result.get(f"{which}_path", "")
        if not path:
            raise ValueError(
                f"no {which} in this result — add '{'mesh' if which == 'glb' else 'gaussian'}' "
                "to decode_formats on MV-SAM3D Run Multi-View.")
        print(f"[MV-SAM3D] {which} -> {path} | peak VRAM {result.get('peak_vram_gb')} GB")
        return (path,)


NODE_CLASS_MAPPINGS = {
    "MVSAM3DSceneFromDir": MVSAM3DSceneFromDir,
    "MVSAM3DLoadViews": MVSAM3DLoadViews,
    "MVSAM3DRunMultiView": MVSAM3DRunMultiView,
    "MVSAM3DExport": MVSAM3DExport,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MVSAM3DSceneFromDir": "MV-SAM3D Scene From Dir",
    "MVSAM3DLoadViews": "MV-SAM3D Load Views",
    "MVSAM3DRunMultiView": "MV-SAM3D Run Multi-View (unofficial)",
    "MVSAM3DExport": "MV-SAM3D Export",
}
