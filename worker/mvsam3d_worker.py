r"""MV-SAM3D worker — runs INSIDE the ``sam3dobjects-nodes`` pixi env (its python).

Driven by a JSON job file (argv[1]); the ComfyUI nodes (which run in ComfyUI's own
python) write the job + scene, Popen this script with the pixi interpreter, stream
its stdout, and read back the exported GLB/PLY. This keeps the heavy pinned CUDA
stack out of ComfyUI's process and reuses the already-built env.

It drives the OFFICIAL devinli123/MV-SAM3D code unmodified:
  OmegaConf.load(pipeline.yaml) -> rendering_engine=pytorch3d -> hydra.instantiate
  -> InferencePipelinePointMap.run_multi_view(...) with the published kwargs.

The optional stage offloading (mvsam3d_offload.py) only moves modules between
CPU/GPU at stage boundaries — it changes no math.

Exit codes: 0 ok; 2 bad job/inputs; 3 a requested decode format failed to export.
"""

import json
import os
import sys
import time
from pathlib import Path


def log(msg):
    print(f">>> {msg}", flush=True)


def fail(msg, code=2):
    print(f"!!! {msg}", flush=True)
    sys.exit(code)


if len(sys.argv) < 2:
    fail("usage: mvsam3d_worker.py <job.json>")
with open(sys.argv[1], encoding="utf-8") as _f:
    JOB = json.load(_f)


def require(key):
    v = JOB.get(key)
    if not v:
        fail(f"job.json is missing required key {key!r}")
    return v


REPO = require("repo_root")
PIPELINE_YAML = require("pipeline_yaml")
SCENE_DIR = require("scene_dir")
OUTPUT_DIR = require("output_dir")
for _p, _label in ((REPO, "repo_root"), (PIPELINE_YAML, "pipeline_yaml"),
                   (SCENE_DIR, "scene_dir")):
    if not os.path.exists(_p):
        fail(f"{_label} does not exist: {_p}")

# --- env vars BEFORE importing sam3d_objects (backends latch at import) ---
# We ARE the pixi env's interpreter, so the env root comes from sys.executable —
# no path needs to be passed in (and no machine-specific default exists).
_exe = Path(sys.executable).resolve()
ENV_ROOT = _exe.parent.parent if _exe.parent.name.lower() in ("bin", "scripts") else _exe.parent
os.environ["CONDA_PREFIX"] = str(ENV_ROOT)
os.environ["CUDA_HOME"] = str(ENV_ROOT)  # pixi env ships the CUDA toolchain
os.environ["LIDRA_SKIP_INIT"] = "true"
os.environ.setdefault("ATTN_BACKEND", "sdpa")
os.environ.setdefault("SPARSE_BACKEND", "spconv")
os.environ.setdefault("SPARSE_ATTN_BACKEND", "sdpa")
# Deliberately NO HF_HOME override: the user's HuggingFace cache settings win.

import numpy as np
import cv2  # noqa: F401  import BEFORE torch (Windows DLL search order)

sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "notebook"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # mvsam3d_offload

from omegaconf import OmegaConf
from hydra.utils import instantiate
import torch

import sam3d_objects  # noqa: F401  required upstream side-effect import
from sam3d_objects.pipeline.inference_pipeline_pointmap import InferencePipelinePointMap
from sam3d_objects.utils.latent_weighting import WeightingConfig
from load_images_and_masks import load_images_and_masks_from_path
from mvsam3d_offload import patch_load_to_cpu, install_stage_offloading

OFFLOAD = bool(JOB.get("offload", True))

# ---------------------------------------------------------------- pipeline
log(f"constructing pipeline (offload={'on' if OFFLOAD else 'off'})...")
t0 = time.time()
if OFFLOAD:
    patch_load_to_cpu(InferencePipelinePointMap)  # keep one-time build < 16GB
cfg = OmegaConf.load(PIPELINE_YAML)
cfg.rendering_engine = "pytorch3d"
cfg.compile_model = False
cfg.workspace_dir = os.path.dirname(PIPELINE_YAML)
# Some weight distributions ship .safetensors while the yaml names .ckpt; rewrite
# only when the .safetensors actually exists next to the yaml.
for k in list(cfg.keys()):
    v = cfg[k]
    if k.endswith("_ckpt_path") and isinstance(v, str) and v.endswith(".ckpt"):
        st = v[:-5] + ".safetensors"
        if os.path.exists(st) or os.path.exists(os.path.join(cfg.workspace_dir, st)):
            cfg[k] = st
pipeline = instantiate(cfg)
if OFFLOAD:
    install_stage_offloading(pipeline)
log(f"pipeline ready in {time.time() - t0:.0f}s")

# ---------------------------------------------------------------- views
view_images, view_masks, names = load_images_and_masks_from_path(
    input_path=Path(SCENE_DIR),
    mask_prompt=JOB.get("mask_prompt") or None,
    image_names=(JOB["image_names"].split(",") if JOB.get("image_names") else None),
)
log(f"{len(view_images)} views: {names}")

# ---------------------------------------------------------------- external pointmaps (DA3)
view_pointmaps = None
da3 = JOB.get("da3_npz")
if da3:
    if not os.path.exists(da3):
        fail(f"da3_npz does not exist: {da3}")
    try:
        # allow_pickle=False: a pickled npz from an untrusted workflow would be
        # arbitrary code execution. pointmaps_sam3d is a float array and
        # image_files loads fine as a fixed-width unicode array.
        data = np.load(da3, allow_pickle=False)
        pms = data["pointmaps_sam3d"]            # (N,3,H,W), OpenCV convention
        files = [os.path.splitext(os.path.basename(str(f)))[0]
                 for f in data["image_files"]]
    except ValueError as e:
        fail(f"could not read {da3} without pickle (refusing pickled npz for "
             f"security): {e}. Re-export it with plain arrays "
             f"(image_files as a string array, not dtype=object).")
    by_name = {f: pms[i] for i, f in enumerate(files)}
    view_pointmaps = [by_name.get(n) for n in names]
    log(f"DA3 pointmaps matched for "
        f"{sum(p is not None for p in view_pointmaps)}/{len(names)} views")

# ---------------------------------------------------------------- weighting (Stage 2)
weight_source = JOB.get("stage2_weight_source", "entropy")
stage2_weighting = JOB.get("stage2_weighting", True)
weighting_config = WeightingConfig(
    entropy_alpha=JOB.get("stage2_entropy_alpha", 30.0),
    attention_layer=JOB.get("stage2_attention_layer", 6),
    attention_step=JOB.get("stage2_attention_step", 0),
    min_weight=JOB.get("stage2_min_weight", 0.001),
    weight_source=weight_source,
    visibility_alpha=JOB.get("stage2_visibility_alpha", 30.0),
    weight_combine_mode=JOB.get("stage2_weight_combine_mode", "average"),
) if stage2_weighting else None

if weight_source in ("visibility", "mixed") and view_pointmaps is None:
    fail(f"stage2_weight_source={weight_source} requires a DA3 npz (da3_npz).")

# ---------------------------------------------------------------- run
decode_formats = [s.strip() for s in (JOB.get("decode_formats") or "gaussian,mesh").split(",")
                  if s.strip()]
unknown = [s for s in decode_formats if s not in ("gaussian", "mesh")]
if unknown:
    fail(f"unknown decode_formats {unknown}; valid: gaussian, mesh")
log(f"run_multi_view: weight_source={weight_source}, seed={JOB.get('seed', 42)}, "
    f"stage1={JOB.get('stage1_steps', 50)}/stage2={JOB.get('stage2_steps', 25)}")
t1 = time.time()
result = pipeline.run_multi_view(
    view_images=view_images,
    view_masks=view_masks,
    view_pointmaps=view_pointmaps,
    seed=JOB.get("seed", 42),
    mode="multidiffusion",
    stage1_inference_steps=JOB.get("stage1_steps", 50),
    stage2_inference_steps=JOB.get("stage2_steps", 25),
    decode_formats=decode_formats,
    with_mesh_postprocess=False,      # published CLI values (run_inference_weighted.py)
    with_texture_baking=False,
    use_vertex_color=True,
    weighting_config=weighting_config,
    ss_weighting=JOB.get("ss_weighting", True),
    ss_entropy_layer=JOB.get("ss_entropy_layer", 9),
    ss_entropy_alpha=JOB.get("ss_entropy_alpha", 30.0),
    ss_warmup_steps=1,
)
log(f"run_multi_view done in {time.time() - t1:.0f}s")

# ---------------------------------------------------------------- export
out_dir = Path(OUTPUT_DIR)
out_dir.mkdir(parents=True, exist_ok=True)
prefix = JOB.get("filename_prefix", "mvsam3d")
written = {}
errors = []
if "mesh" in decode_formats:
    try:
        if result.get("glb") is None:
            raise RuntimeError("pipeline returned no 'glb' object")
        p = out_dir / f"{prefix}.glb"
        result["glb"].export(str(p))
        written["glb"] = str(p)
        log(f"wrote {p}")
    except Exception as e:
        errors.append(f"glb export failed: {e!r}")
if "gaussian" in decode_formats:
    try:
        g = result.get("gs") or (result.get("gaussian") or [None])[0]
        if g is None or not hasattr(g, "save_ply"):
            raise RuntimeError("pipeline returned no saveable gaussian object")
        p = out_dir / f"{prefix}.ply"
        g.save_ply(str(p))
        written["ply"] = str(p)
        log(f"wrote {p}")
    except Exception as e:
        errors.append(f"ply export failed: {e!r}")

pose = {}
for k in ("scale", "rotation", "translation"):
    v = result.get(k)
    if v is not None:
        pose[k] = v.detach().cpu().numpy().tolist() if torch.is_tensor(v) else v
written["pose"] = pose
peak = round(torch.cuda.max_memory_allocated() / 1e9, 2) if torch.cuda.is_available() else None
with open(out_dir / f"{prefix}_result.json", "w", encoding="utf-8") as f:
    json.dump({"written": written, "peak_vram_gb": peak, "errors": errors}, f, indent=2)
log(f"PEAK VRAM {peak} GB")

if errors:
    # A requested output did not materialize: fail LOUD so the ComfyUI node raises
    # instead of returning empty paths from a "successful" run.
    fail("; ".join(errors), code=3)
log("WORKER DONE")
