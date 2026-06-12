"""CPU unit tests for the glue logic (no ComfyUI, no GPU, no model).

Run with pytest, or directly:  python tests/test_glue_cpu.py
"""

import importlib.util
import sys
import types
from pathlib import Path

import pytest

PACK_ROOT = Path(__file__).resolve().parent.parent


def _load_from_file(module_name: str, path: Path):
    """Load a pack module straight from its file — keeps pytest's importer away
    from the pack root (which is itself a package directory)."""
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


discovery = _load_from_file("mvsam3d_unofficial_discovery", PACK_ROOT / "discovery.py")
safe_name = discovery.safe_name
env_root_of_python = discovery.env_root_of_python


# ---------------------------------------------------------------- safe_name

@pytest.mark.parametrize("ok", ["scene", "stuffed_toy", "a1.b-c_d", "X"])
def test_safe_name_accepts(ok):
    assert safe_name(ok, "x") == ok


@pytest.mark.parametrize("bad", [
    "", "  ", "..", "a/..", "a/b", "a\\b", "..\\evil", "../evil",
    ".hidden", "C:\\abs", "/abs", "a\x00b", "con dir",
])
def test_safe_name_rejects(bad):
    with pytest.raises(ValueError):
        safe_name(bad, "x")


# ---------------------------------------------------------------- env root

def test_env_root_windows_flat_layout(tmp_path):
    py = tmp_path / "env" / "python.exe"
    py.parent.mkdir(parents=True)
    py.touch()
    assert env_root_of_python(py) == py.parent


def test_env_root_posix_bin_layout(tmp_path):
    py = tmp_path / "env" / "bin" / "python"
    py.parent.mkdir(parents=True)
    py.touch()
    assert env_root_of_python(py) == tmp_path / "env"


def test_env_root_windows_venv_scripts_layout(tmp_path):
    py = tmp_path / "venv" / "Scripts" / "python.exe"
    py.parent.mkdir(parents=True)
    py.touch()
    assert env_root_of_python(py) == tmp_path / "venv"


# ---------------------------------------------------------------- resolvers

def test_resolve_pixi_python_rejects_non_python(tmp_path):
    exe = tmp_path / "evil.exe"
    exe.touch()
    with pytest.raises(ValueError):
        discovery.resolve_pixi_python(str(exe))


def test_resolve_pixi_python_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        discovery.resolve_pixi_python(str(tmp_path / "nope" / "python.exe"))


def test_resolve_repo_root_rejects_non_checkout(tmp_path):
    with pytest.raises(FileNotFoundError):
        discovery.resolve_repo_root(str(tmp_path))


def test_resolve_repo_root_accepts_valid_checkout(tmp_path):
    (tmp_path / "sam3d_objects").mkdir()
    (tmp_path / "notebook").mkdir()
    (tmp_path / "notebook" / "load_images_and_masks.py").touch()
    assert discovery.resolve_repo_root(str(tmp_path)) == str(tmp_path)


def test_resolve_pipeline_yaml_from_candidate_dirs(tmp_path):
    yaml = tmp_path / "rootB" / "sam3dobjects" / "pipeline.yaml"
    yaml.parent.mkdir(parents=True)
    yaml.touch()
    found = discovery.resolve_pipeline_yaml("", candidate_dirs=[
        str(tmp_path / "rootA" / "sam3dobjects"),   # doesn't exist -> skipped
        str(yaml.parent),
    ])
    assert found == str(yaml)


def test_resolve_pipeline_yaml_missing_lists_all_tried(tmp_path):
    with pytest.raises(FileNotFoundError) as ei:
        discovery.resolve_pipeline_yaml("", candidate_dirs=[
            str(tmp_path / "a"), str(tmp_path / "b")])
    msg = str(ei.value)
    assert "a" in msg and "b" in msg


# ---------------------------------------------------------------- nodes.py helpers
# Import nodes.py with stub ComfyUI modules so its pure helpers are testable.

def _import_nodes():
    for name in ("folder_paths", "comfy", "comfy.model_management"):
        sys.modules.setdefault(name, types.ModuleType(name))
    pkg = types.ModuleType("mvsam3d_unofficial")
    pkg.__path__ = [str(PACK_ROOT)]
    sys.modules.setdefault("mvsam3d_unofficial", pkg)
    sys.modules["mvsam3d_unofficial.discovery"] = discovery
    nodes = _load_from_file("mvsam3d_unofficial.nodes", PACK_ROOT / "nodes.py")
    return nodes


def test_mask_written_as_rgba_alpha():
    """The official loader reads masks from the ALPHA channel; a grayscale mask
    file would silently become an all-ones mask. Pin the RGBA contract."""
    np_ = pytest.importorskip("numpy")
    nodes = _import_nodes()
    mask = np_.zeros((4, 4), dtype=np_.uint8)
    mask[1:3, 1:3] = 255
    img = nodes._mask_to_rgba(mask)
    assert img.mode == "RGBA"
    out = np_.array(img)
    assert (out[..., 3] == mask).all()


def test_node_mappings_complete():
    nodes = _import_nodes()
    assert set(nodes.NODE_CLASS_MAPPINGS) == set(nodes.NODE_DISPLAY_NAME_MAPPINGS)
    assert {"MVSAM3DSceneFromDir", "MVSAM3DLoadViews", "MVSAM3DRunMultiView",
            "MVSAM3DExport"} == set(nodes.NODE_CLASS_MAPPINGS)


def test_export_path_is_output_relative_for_preview3d(tmp_path):
    """Preview3D loads model_file relative to the output dir via /view; an
    absolute path fails with 'Error loading model'. Pin the relativize contract."""
    nodes = _import_nodes()
    out_root = tmp_path / "output"
    glb = out_root / "mvsam3d" / "run_1" / "m.glb"
    glb.parent.mkdir(parents=True)
    glb.touch()
    rel = nodes._output_relative(str(glb), str(out_root))
    assert rel == "mvsam3d/run_1/m.glb"
    # outside the output dir -> unchanged
    other = tmp_path / "elsewhere" / "m.glb"
    other.parent.mkdir(parents=True)
    other.touch()
    assert nodes._output_relative(str(other), str(out_root)) == str(other)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
