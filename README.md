# ComfyUI-MVSAM3D-Unofficial

ComfyUI nodes for [MV-SAM3D](https://github.com/devinli123/MV-SAM3D)
(arXiv:2603.11633), the multi-view extension of Meta's SAM 3D Objects.

Feed it several images and masks of the same object from different angles and it
reconstructs a single mesh (`.glb`) and/or Gaussian splat (`.ply`) using the
paper's entropy-weighted multi-view fusion. The MV-SAM3D code is not modified or
copied — it's pulled in as a pinned git submodule and run as published. This repo
contains the ComfyUI nodes and the subprocess plumbing, nothing else.

Not affiliated with Meta or the MV-SAM3D authors, hence "Unofficial".

## Requirements

- Windows or Linux, NVIDIA GPU. 16 GB VRAM is enough with the default stage
  offloading (measured peak: 13.95 GB for the 8-view example scene). Running
  with offloading disabled wants roughly 24 GB.
- [ComfyUI-SAM3DObjects](https://github.com/PozzettiAndrea/ComfyUI-SAM3DObjects)
  installed and working, meaning it has produced at least one single-view mesh.
  This pack reuses its `sam3dobjects-nodes` pixi environment and its downloaded
  weights (`models/sam3dobjects/`, including `pipeline.yaml`). Without it,
  nothing here runs.
- A few extra GB of disk for the MV-SAM3D checkout.

## Install

Via ComfyUI Manager or the registry: install the pack, then run the one-time
environment setup from a terminal (the install hook fetches the pinned MV-SAM3D
repo on its own, but the pixi-env extras need a real shell once):

```bash
cd ComfyUI/custom_nodes/ComfyUI-MVSAM3D-Unofficial
python scripts/setup_env.py
```

Via git:

```bash
cd ComfyUI/custom_nodes
git clone --recursive https://github.com/robinduckett/ComfyUI-MVSAM3D-Unofficial.git
cd ComfyUI-MVSAM3D-Unofficial
python scripts/setup_env.py
```

`setup_env.py` is idempotent and safe to re-run. It:

1. fetches the pinned MV-SAM3D submodule into `vendor/MV-SAM3D` if missing
   (handles installs made without `--recursive`, or from a registry archive
   that has no git metadata);
2. pip-installs the extra runtime deps the MV-SAM3D code needs into the
   `sam3dobjects-nodes` env (`astor lightning easydict roma rootutils randomname
   jsonpickle einops-exts`);
3. pip-installs [MoGe](https://github.com/microsoft/MoGe) at the pinned commit
   `a8c37341` with `--no-deps` (the current MoGe release expects a newer
   `utils3d` than the env ships);
4. writes a no-op `kaolin.utils.testing.check_tensor` stub if no kaolin wheel
   exists for your GPU/torch combination — that one validator function is the
   only thing the inference path imports from kaolin;
5. smoke-checks the env imports.

On a standard setup no paths need configuring; the nodes find the pixi-env
python and `models/sam3dobjects/pipeline.yaml` on their own. To override, use
the node inputs `pixi_python` / `repo_root` / `pipeline_yaml`, or the
environment variables `MVSAM3D_PIXI_PYTHON` / `MVSAM3D_REPO` /
`MVSAM3D_PIPELINE_YAML` (node input wins over env var wins over discovery).

## Reproducing the paper example

Open `example_workflows/mvsam3d_paper_example.json` and queue it. With
`scene_dir` left empty, the Scene From Dir node uses the example scene that
ships inside the MV-SAM3D repo (`data/example`: 8 views of a stuffed toy), and
the node defaults match the published run settings (seed 42, stage1 50 /
stage2 25 steps, SS entropy layer 9, SLAT entropy layer 6, alpha 30).

Expect roughly 3 minutes of pipeline build plus 4–5 minutes of inference on a
16 GB card. Output lands in `output/mvsam3d/…/mvsam3d_example.glb|.ply` and
shows up in the Preview3D node. The authors' own renders of the same scene are
in `vendor/MV-SAM3D/data/example/visualization_results/` if you want to compare.

## Nodes

| Node | Purpose |
|---|---|
| MV-SAM3D Scene From Dir | Use an on-disk scene in the MV-SAM3D layout (`images/` + `<mask_prompt>/` with RGBA masks, or a flat folder of `N.png` + `N_mask.png`). Empty `scene_dir` = the bundled example scene. |
| MV-SAM3D Load Views | Write a ComfyUI `IMAGE` + `MASK` batch to a scene dir, recreated on every run. Masks are saved as RGBA-alpha because that's the channel the MV-SAM3D loader reads. |
| MV-SAM3D Run Multi-View (unofficial) | Run `InferencePipelinePointMap.run_multi_view` in the pixi-env subprocess, with the stage-1 and stage-2 weighting knobs exposed. |
| MV-SAM3D Export | Pull the GLB or PLY path out of a result for Preview3D / save nodes. Errors if the requested format wasn't produced. |

### Using your own views

`LoadImage` per view, batch them (e.g. KJNodes' Image Batch Multi), batch the
masks the same way, then Load Views → Run Multi-View → Export → Preview3D.
Masks can come from whatever segmentation you use; every view needs one, since
the MV-SAM3D loader skips views without a mask. All views must show the same
object.

### Weighting

`entropy` (the default) is the paper's main method and needs nothing extra.
`visibility` and `mixed` also need a Depth-Anything-3 pointmap export passed in
via `da3_npz`: an `.npz` containing `pointmaps_sam3d` `(N,3,H,W)` and
`image_files`, saved without pickled objects. Producing that file requires a
separate DA3 install, which this pack doesn't automate.

## How it works

The nodes run in ComfyUI's own python; the heavy lifting happens in a
subprocess (`worker/mvsam3d_worker.py`) launched with the `sam3dobjects-nodes`
pixi-env interpreter.

Why a subprocess instead of an isolated env of our own: comfy-aimdo keys
isolated envs on the plugin directory name, so giving this pack its own env
spec would rebuild the multi-GB CUDA stack (pytorch3d, spconv, flash-attn, …)
from scratch. Reusing the env that ComfyUI-SAM3DObjects already built avoids
that, and keeps ComfyUI's torch untouched.

Details:

- The worker reads a JSON job file and derives its env root from its own
  `sys.executable`. Images and masks travel via a scene directory; results come
  back as files plus a result JSON. Exit code 2 means bad input, 3 means a
  requested export failed — either way the node raises with the worker's last
  output lines instead of returning empty paths.
- `worker/mvsam3d_offload.py` (on by default) keeps only the current stage's
  models on the GPU. That's what makes the run fit 16 GB: 13.95 GB peak instead
  of ~20 GB with everything resident. It moves modules between devices and
  touches no math; turn it off via the `offload` input on bigger cards.
- The pipeline is rebuilt on every run (~3 min). A persistent worker process
  could amortize that, but it doesn't exist yet.

## Troubleshooting

- "Could not find the 'sam3dobjects-nodes' pixi environment python" — install
  and run ComfyUI-SAM3DObjects first, or set `MVSAM3D_PIXI_PYTHON`.
- "The MV-SAM3D submodule is not initialized" — `git submodule update --init`,
  or `python scripts/setup_env.py`.
- "Could not find … pipeline.yaml" — run ComfyUI-SAM3DObjects once so it
  downloads the weights, or set `MVSAM3D_PIPELINE_YAML`.
- `ModuleNotFoundError` inside the worker (lightning, roma, moge, kaolin, …) —
  the env extras are missing; run `python scripts/setup_env.py`.
- CUDA out of memory — keep `offload` on and close other GPU apps. The run
  peaks around 14 GB; cards under 16 GB are untested.
- Cancelling: ComfyUI's interrupt kills the worker at its next output line.

## License & attribution

The code in this repo is MIT — see [`LICENSE`](LICENSE), including the scope
notice. MV-SAM3D itself is not redistributed here: `vendor/MV-SAM3D` is a git
submodule pointer, fetched from upstream onto your machine, and governed by
Meta's SAM License (read it before commercial use). The SAM 3D Objects weights
are also SAM-License material, downloaded by ComfyUI-SAM3DObjects (GPL-3.0),
whose code this pack doesn't import or link.

If you use this in published work, cite MV-SAM3D (arXiv:2603.11633) and Meta
SAM 3D Objects. The method is theirs; this repo is just the ComfyUI wiring.
