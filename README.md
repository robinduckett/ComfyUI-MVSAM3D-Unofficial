# ComfyUI-MVSAM3D-Unofficial

**Unofficial ComfyUI wrapper for [MV-SAM3D](https://github.com/devinli123/MV-SAM3D)
(arXiv:2603.11633) — runs the authors' official code, unmodified, inside ComfyUI.**

Give it N images + masks of the same object from different angles; it reproduces the
paper's entropy-weighted multi-view SAM 3D Objects reconstruction and hands back a
mesh (`.glb`) and/or Gaussian splat (`.ply`). The fusion math is the upstream
authors' code, taken from a **pinned git submodule** — this pack contributes only
ComfyUI nodes and subprocess glue. "Unofficial" means: not by the MV-SAM3D authors,
not by Meta; all method credit is theirs.

> Looking for a ComfyUI-**native** re-implementation (fusion injected into the
> ComfyUI-SAM3DObjects sampler, no subprocess)? That is the sibling project
> `ComfyUI-MV-SAM3D`. **This** pack is the conservative choice: it executes the
> published pipeline byte-for-byte and is the path that has been GPU-validated
> end-to-end (13.95 GB peak / 267 s for the 8-view example on an RTX 5070 Ti).

## Requirements

| | |
|---|---|
| OS | Windows or Linux (the reused pixi env targets win-64 / linux-64) |
| GPU | NVIDIA, **16 GB VRAM** with the default stage offloading (measured 13.95 GB peak); ~24 GB+ to run fully resident (`offload=false`) |
| Prerequisite pack | [ComfyUI-SAM3DObjects](https://github.com/PozzettiAndrea/ComfyUI-SAM3DObjects) installed **and working** (one successful single-view mesh). This pack reuses its `sam3dobjects-nodes` pixi environment and its downloaded SAM 3D Objects weights (`models/sam3dobjects/`, incl. `pipeline.yaml`). |
| Disk | The prerequisite env + weights are multi-GB; this pack itself adds ~the size of the MV-SAM3D repo (submodule) |

## Install

```bash
cd ComfyUI/custom_nodes
git clone --recursive https://github.com/robinduckett/ComfyUI-MVSAM3D-Unofficial.git
cd ComfyUI-MVSAM3D-Unofficial
python scripts/setup_env.py
```

`setup_env.py` (idempotent, re-run safe):

1. fetches the **pinned** MV-SAM3D submodule into `vendor/MV-SAM3D` if missing
   (covers installs without `--recursive` or from a registry archive);
2. pip-installs the extra runtime deps the official code needs into the
   `sam3dobjects-nodes` env (`astor lightning easydict roma rootutils randomname
   jsonpickle einops-exts`);
3. pip-installs the **pinned** [MoGe](https://github.com/microsoft/MoGe)
   (`a8c37341`, `--no-deps` — matches the env's `utils3d`);
4. writes a no-op `kaolin.utils.testing.check_tensor` stub if no kaolin wheel
   exists for your GPU/torch combination (only this validator is imported on the
   inference path);
5. runs an import smoke-check of the env.

No paths need configuring on a standard setup — the nodes auto-discover the pixi
env python and `models/sam3dobjects/pipeline.yaml`. Overrides, in priority order:
node inputs `pixi_python` / `repo_root` / `pipeline_yaml`, then environment
variables `MVSAM3D_PIXI_PYTHON` / `MVSAM3D_REPO` / `MVSAM3D_PIPELINE_YAML`.

## Reproduce the paper's example (one click)

Open `example_workflows/mvsam3d_paper_example.json` and run it. With
`scene_dir` left empty, **MV-SAM3D Scene From Dir** uses the paper's own bundled
example scene (`vendor/MV-SAM3D/data/example`: 8 views of a stuffed toy), and the
node defaults are the published run settings (seed 42, stage1 50 / stage2 25
steps, SS entropy layer 9, SLAT entropy layer 6, alpha 30, entropy weighting).
Output lands in `output/mvsam3d/…/mvsam3d_example.glb|.ply` and previews in
`Preview3D`. Compare against the renders in
`vendor/MV-SAM3D/data/example/visualization_results/`.

## Nodes

| Node | Purpose |
|---|---|
| **MV-SAM3D Scene From Dir** | Use an on-disk scene in the official layout (`images/` + `<mask_prompt>/` with RGBA masks, or a flat `N.png` + `N_mask.png` folder). Empty `scene_dir` = the bundled paper example. |
| **MV-SAM3D Load Views** | Write an in-memory ComfyUI `IMAGE` + `MASK` batch to a scene dir (recreated every run). Masks are saved as RGBA-alpha, the format the official loader reads. |
| **MV-SAM3D Run Multi-View (unofficial)** | Run the official `InferencePipelinePointMap.run_multi_view` with the published kwargs in the pixi-env subprocess. Exposes the Stage-1 (SS entropy) and Stage-2 (entropy / visibility / mixed) weighting knobs. |
| **MV-SAM3D Export** | Surface the GLB/PLY path for `Preview3D` / save nodes; errors if the requested format wasn't produced. |

### Your own views

`LoadImage` (one per view) → core **Image Batch** nodes to combine → **MV-SAM3D
Load Views** (`images` + `masks`, e.g. from your segmentation pack of choice or
LoadImage's alpha output) → **Run Multi-View** → **Export** → `Preview3D`. Every
view must show the **same object**; masks are required (the official pipeline
skips unmasked views).

### Weighting knobs

`entropy` (default) is the paper's headline method and needs nothing extra.
`visibility` / `mixed` additionally require a Depth-Anything-3 pointmap export
(`da3_npz`: an `.npz` with `pointmaps_sam3d` `(N,3,H,W)` and `image_files`,
saved **without** pickled objects) — generating it needs a separate DA3 install
and is not automated by this pack.

## How it works (architecture)

The nodes run in ComfyUI's own python and do the heavy lifting in a
**subprocess** (`worker/mvsam3d_worker.py`) launched with the prebuilt
`sam3dobjects-nodes` pixi-env interpreter:

- comfy-aimdo keys isolated envs on the *plugin directory name*, so giving this
  pack its own env spec would rebuild the multi-GB CUDA stack from scratch.
  Reusing the donor-built env avoids that and keeps ComfyUI's torch untouched.
- The worker derives the env root from its own `sys.executable` and reads a JSON
  job file; images/masks are marshaled via a scene directory; results come back
  as files + a result JSON. Exit codes distinguish bad input (2) from export
  failure (3), and the node surfaces the worker's last output lines on failure.
- `worker/mvsam3d_offload.py` (default **on**) keeps only the current stage's
  models on the GPU so the whole run fits 16 GB (measured 13.95 GB / 267 s for
  the 8-view example; ~20 GB if everything stays resident). It is pure device
  placement — **no fusion math is touched**. Disable via the `offload` input to
  run the original fully-resident behavior on bigger cards.
- Cold start builds the pipeline (~2–3 min) on every run; a persistent worker
  server could amortize this later.

## Troubleshooting

- **"Could not find the 'sam3dobjects-nodes' pixi environment python"** — install
  and run ComfyUI-SAM3DObjects once first, or set `MVSAM3D_PIXI_PYTHON`.
- **"The MV-SAM3D submodule is not initialized"** — `git submodule update --init`
  or `python scripts/setup_env.py`.
- **"Could not find … pipeline.yaml"** — run ComfyUI-SAM3DObjects once so it
  downloads the weights, or set `MVSAM3D_PIPELINE_YAML`.
- **`ModuleNotFoundError` inside the worker** (lightning, roma, moge, kaolin…) —
  run `python scripts/setup_env.py` (the env extras are missing).
- **CUDA out of memory** — keep `offload=true`, close other GPU apps; the run
  peaks ~14 GB. Below 16 GB VRAM is untested.
- Cancelling: the node kills the worker at the next output line after you press
  ComfyUI's interrupt.

## License & attribution

The wrapper code in this repo is **MIT** — see [`LICENSE`](LICENSE), including
the scope notice. **MV-SAM3D itself is not redistributed here**: `vendor/MV-SAM3D`
is a git submodule pointer, fetched from upstream onto your machine, and is
governed by **Meta's SAM License** (non-permissive — review it, especially for
commercial use). The SAM 3D Objects weights are likewise SAM-License-governed and
are downloaded by ComfyUI-SAM3DObjects (GPL-3.0), whose code this pack does not
import or link.

If you use this in published work, cite **MV-SAM3D (arXiv:2603.11633)** and
**Meta SAM 3D Objects**. This project is not affiliated with Meta or the
MV-SAM3D authors.
