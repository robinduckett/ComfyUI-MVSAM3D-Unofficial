r"""Stage-wise CPU offloading for InferencePipelinePointMap (MV-SAM3D).

The official pipeline loads every model to CUDA in __init__ and keeps it resident:
ss_generator (6.7GB) + slat_generator (4.9GB) + 2 condition embedders (~1.2GB DINO
each) + MoGe (1.2GB) + decoders. On a 16GB card that peaks ~20GB and spills to
system RAM (catastrophically slow). But the stages are disjoint:

    preprocess (MoGe) -> Stage 1 (ss_generator + ss embedder + ss_decoder)
                      -> Stage 2 (slat_generator + slat embedder)
                      -> decode  (decoders)

So we wrap the four stage methods to move the BIG modules on/off the GPU at stage
boundaries, keeping only what the current stage needs resident. Small decoders
(<1GB total, ss_decoder is also used inside Stage 1) stay resident. No change to
any fusion math — pure device shuffling.

Usage:
    from mvsam3d_offload import install_stage_offloading
    install_stage_offloading(pipeline)   # after construction, before run_multi_view
"""
import torch


def patch_load_to_cpu(pipeline_cls):
    """Force model construction to land on CPU (call BEFORE instantiate()).

    __init__ loads every generator/decoder to self.device (cuda) via
    instantiate_and_load_from_pretrained, so at the end of construction ALL models
    are briefly co-resident on the GPU (~18.6GB peak on the example). Override the
    loader to land each model on CPU instead; install_stage_offloading() then moves
    them to the GPU on demand. Keeps the one-time build well under 16GB so it is safe
    alongside a running ComfyUI.
    """
    if getattr(pipeline_cls, "_mvsam3d_cpu_patched", False):
        return
    _orig = pipeline_cls.instantiate_and_load_from_pretrained

    def _patched(self, config, ckpt_path, state_dict_fn=None,
                 state_dict_key="state_dict", device="cuda"):
        return _orig(self, config, ckpt_path, state_dict_fn=state_dict_fn,
                     state_dict_key=state_dict_key, device="cpu")

    pipeline_cls.instantiate_and_load_from_pretrained = _patched
    pipeline_cls._mvsam3d_cpu_patched = True


def _to(module, device):
    """Move an nn.Module, or a DepthModel-style wrapper exposing `.model`, to device."""
    if module is None:
        return
    if isinstance(module, torch.nn.Module):
        module.to(device)
    elif hasattr(module, "model") and isinstance(module.model, torch.nn.Module):
        module.model.to(device)   # DepthModel wrapper (MoGe) — inner nn.Module


def install_stage_offloading(pipeline, device="cuda", cpu="cpu", verbose=True):
    models = pipeline.models                  # ModuleDict
    emb = pipeline.condition_embedders        # plain dict
    DECODERS = ("ss_decoder", "slat_decoder_gs", "slat_decoder_gs_4", "slat_decoder_mesh")

    def vram():
        return torch.cuda.memory_allocated() / 1e9

    def log(msg):
        if verbose:
            print(f"[offload] {msg} | resident {vram():.1f}GB", flush=True)

    # Start fully on CPU, then pin the small decoders on GPU (ss_decoder is needed
    # inside Stage 1; all decoders together are <1GB so keep them resident).
    for k in list(models.keys()):
        _to(models[k], cpu)
    for k in list(emb.keys()):
        _to(emb[k], cpu)
    _to(getattr(pipeline, "depth_model", None), cpu)
    torch.cuda.empty_cache()
    for d in DECODERS:
        if d in models:
            _to(models[d], device)
    log("init: generators+embedders+MoGe -> CPU, decoders -> GPU")

    orig_cp = pipeline.compute_pointmap
    orig_ss = pipeline.sample_sparse_structure_multi_view
    orig_slat = pipeline.sample_slat_multi_view_weighted

    def compute_pointmap(*a, **k):
        _to(getattr(pipeline, "depth_model", None), device)
        return orig_cp(*a, **k)

    def sample_ss(*a, **k):
        _to(getattr(pipeline, "depth_model", None), cpu)
        _to(models["slat_generator"], cpu)
        _to(emb.get("slat_condition_embedder"), cpu)
        torch.cuda.empty_cache()
        _to(models["ss_generator"], device)
        _to(emb.get("ss_condition_embedder"), device)
        log("Stage 1: ss_generator on GPU")
        return orig_ss(*a, **k)

    def sample_slat(*a, **k):
        _to(models["ss_generator"], cpu)
        _to(emb.get("ss_condition_embedder"), cpu)
        _to(getattr(pipeline, "depth_model", None), cpu)
        torch.cuda.empty_cache()
        _to(models["slat_generator"], device)
        _to(emb.get("slat_condition_embedder"), device)
        log("Stage 2: slat_generator on GPU")
        return orig_slat(*a, **k)

    pipeline.compute_pointmap = compute_pointmap
    pipeline.sample_sparse_structure_multi_view = sample_ss
    pipeline.sample_slat_multi_view_weighted = sample_slat
    return pipeline
