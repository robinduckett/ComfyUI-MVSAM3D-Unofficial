"""Generate the 4-view upscaled custom-views workflow JSON.

Mirrors the per-view prep of the norn-4view-upscaled graph (RealESRGAN x4 on the
image, alpha-derived mask inverted + scaled to match) but feeds the MV-SAM3D
Unofficial nodes. Depth estimation is gone on purpose: the official pipeline
computes pointmaps itself (MoGe). Core nodes only, so the shipped example adds
no node-pack dependencies.

Usage:
    python scripts/gen_custom_views_workflow.py out.json [img1 img2 img3 img4] [prefix]
"""

import json
import sys

CNR = "mvsam3d-unofficial"


def build(filenames, prefix="custom4"):
    nodes, links = [], []
    lid = [0]

    def link(fn, fs, tn, ts, typ):
        lid[0] += 1
        links.append([lid[0], fn, fs, tn, ts, typ])
        return lid[0]

    def node(nid, typ, pos, size, inputs, outputs, widgets, props=None, **extra):
        n = {"id": nid, "type": typ, "pos": pos, "size": size, "flags": {},
             "order": nid, "mode": 0, "inputs": inputs, "outputs": outputs,
             "properties": props or {"Node name for S&R": typ},
             "widgets_values": widgets}
        n.update(extra)
        nodes.append(n)
        return n

    def ours(typ):
        return {"Node name for S&R": typ, "cnr_id": CNR}

    note = (
        "MV-SAM3D: 4 custom views, upscaled.\n\n"
        "Per view: RealESRGAN x4 on the image; the foreground mask comes from the "
        "image's alpha channel (InvertMask because LoadImage's MASK output is "
        "inverted alpha) and is scaled x4 to match. No alpha? Feed masks from your "
        "segmentation pack into Batch Masks instead.\n\n"
        "No depth-estimate step is needed: the MV-SAM3D pipeline computes pointmaps "
        "itself (MoGe).\n\n"
        "weight_source=entropy is the paper method. 'visibility'/'mixed' need a "
        "Depth-Anything-3 npz on da3_npz.\n\n"
        "View order = batch order; put your reference/front view first. More views: "
        "Batch Images / Batch Masks grow extra inputs as you connect them. Images "
        "of different sizes are auto-resized to the first view's size."
    )
    # ---- layout: column x positions and a row pitch with real padding.
    # ComfyUI draws ~30px of title bar ABOVE the stored size, so the pitch
    # must exceed the tallest per-view stack (LoadImage 314 / preview 250+).
    C_LOAD, C_UP, C_MASK1, C_MASK2 = -1640, -1320, -1000, -730
    C_BATCH, C_VIEWS, C_RUN, C_EXP, C_PREV = -450, -160, 190, 610, 920
    ROW0, PITCH = -260, 480

    node(90, "Note", [C_LOAD, -680], [500, 300], [], [], [note],
         props={}, color="#432", bgcolor="#653")

    node(1, "UpscaleModelLoader", [C_MASK1, -540], [320, 60],
         [],
         [{"name": "UPSCALE_MODEL", "type": "UPSCALE_MODEL", "links": []}],
         ["RealESRGAN_x4plus.pth"])

    img_out, mask_out = [], []  # (node_id, slot) per view
    for i, fname in enumerate(filenames):
        y = ROW0 + i * PITCH
        li, ui, pi = 100 + i * 10, 101 + i * 10, 106 + i * 10
        mi, ti, si, ki = 102 + i * 10, 103 + i * 10, 104 + i * 10, 105 + i * 10
        node(li, "LoadImage", [C_LOAD, y], [274, 314],
             [],
             [{"name": "IMAGE", "type": "IMAGE", "links": []},
              {"name": "MASK", "type": "MASK", "links": []}],
             [fname, "image"])
        node(ui, "ImageUpscaleWithModel", [C_UP, y], [270, 50],
             [{"name": "upscale_model", "type": "UPSCALE_MODEL",
               "link": link(1, 0, ui, 0, "UPSCALE_MODEL")},
              {"name": "image", "type": "IMAGE", "link": link(li, 0, ui, 1, "IMAGE")}],
             [{"name": "IMAGE", "type": "IMAGE", "links": []}],
             [])
        # Preview of the upscaled view, directly under its upscaler.
        node(pi, "PreviewImage", [C_UP, y + 130], [270, 250],
             [{"name": "images", "type": "IMAGE", "link": link(ui, 0, pi, 0, "IMAGE")}],
             [], [])
        node(mi, "InvertMask", [C_MASK1, y], [220, 50],
             [{"name": "mask", "type": "MASK", "link": link(li, 1, mi, 0, "MASK")}],
             [{"name": "MASK", "type": "MASK", "links": []}],
             [])
        node(ti, "MaskToImage", [C_MASK1, y + 120], [220, 50],
             [{"name": "mask", "type": "MASK", "link": link(mi, 0, ti, 0, "MASK")}],
             [{"name": "IMAGE", "type": "IMAGE", "links": []}],
             [])
        node(si, "ImageScaleBy", [C_MASK2, y], [230, 90],
             [{"name": "image", "type": "IMAGE", "link": link(ti, 0, si, 0, "IMAGE")}],
             [{"name": "IMAGE", "type": "IMAGE", "links": []}],
             ["bilinear", 4.0])
        node(ki, "ImageToMask", [C_MASK2, y + 160], [230, 80],
             [{"name": "image", "type": "IMAGE", "link": link(si, 0, ki, 0, "IMAGE")}],
             [{"name": "MASK", "type": "MASK", "links": []}],
             ["red"])
        img_out.append((ui, 0))
        mask_out.append((ki, 0))

    mid_y = ROW0 + (len(filenames) - 1) * PITCH // 2
    # Modern variadic batch nodes (ImageBatch / chained pairs are deprecated).
    node(200, "BatchImagesNode", [C_BATCH, mid_y - 120],
         [230, 40 + 26 * len(img_out)],
         [{"name": f"image{k}", "type": "IMAGE",
           "link": link(src[0], src[1], 200, k, "IMAGE")}
          for k, src in enumerate(img_out)],
         [{"name": "IMAGE", "type": "IMAGE", "links": []}],
         [])
    node(210, "BatchMasksNode", [C_BATCH, mid_y + 120],
         [230, 40 + 26 * len(mask_out)],
         [{"name": f"mask{k}", "type": "MASK",
           "link": link(src[0], src[1], 210, k, "MASK")}
          for k, src in enumerate(mask_out)],
         [{"name": "MASK", "type": "MASK", "links": []}],
         [])

    node(300, "MVSAM3DLoadViews", [C_VIEWS, mid_y - 30], [300, 130],
         [{"name": "images", "type": "IMAGE",
           "link": link(200, 0, 300, 0, "IMAGE")},
          {"name": "masks", "type": "MASK", "link": link(210, 0, 300, 1, "MASK")}],
         [{"name": "scene", "type": "MVSAM3D_SCENE", "links": []}],
         [prefix, prefix], props=ours("MVSAM3DLoadViews"))

    node(301, "MVSAM3DRunMultiView", [C_RUN, mid_y - 290], [360, 580],
         [{"name": "scene", "type": "MVSAM3D_SCENE",
           "link": link(300, 0, 301, 0, "MVSAM3D_SCENE")}],
         [{"name": "result", "type": "MVSAM3D_RESULT", "links": []},
          {"name": "glb_path", "type": "STRING", "links": []},
          {"name": "ply_path", "type": "STRING", "links": []}],
         [42, "fixed", 50, 25, True, 9, 30.0, True, "entropy", 30.0, 30.0, 6, 0,
          0.001, "average", "gaussian,mesh", prefix, True, "", "", "", ""],
         props=ours("MVSAM3DRunMultiView"))

    node(302, "MVSAM3DExport", [C_EXP, mid_y - 290], [250, 90],
         [{"name": "result", "type": "MVSAM3D_RESULT",
           "link": link(301, 0, 302, 0, "MVSAM3D_RESULT")}],
         [{"name": "path", "type": "STRING", "links": []},
          {"name": "abs_path", "type": "STRING", "links": []}],
         ["glb"], props=ours("MVSAM3DExport"))

    node(303, "Preview3D", [C_PREV, mid_y - 290], [400, 450],
         [{"name": "camera_info", "type": "LOAD3D_CAMERA", "link": None},
          {"name": "bg_image", "type": "IMAGE", "link": None},
          {"name": "model_file", "type": "STRING",
           "link": link(302, 0, 303, 2, "STRING")}],
         [], ["", ""])

    # backfill output 'links' arrays from the links table
    by_id = {n["id"]: n for n in nodes}
    for l, fn, fs, tn, ts, _typ in links:
        outs = by_id[fn]["outputs"]
        if fs < len(outs):
            outs[fs]["links"].append(l)

    return {"id": f"mvsam3d-unofficial-{prefix}",
            "revision": 0,
            "last_node_id": max(n["id"] for n in nodes),
            "last_link_id": lid[0],
            "nodes": sorted(nodes, key=lambda n: n["id"]),
            "links": links, "groups": [], "config": {}, "extra": {},
            "version": 0.4}


if __name__ == "__main__":
    out = sys.argv[1]
    files = sys.argv[2:6] if len(sys.argv) >= 6 else (
        ["view_1.png", "view_2.png", "view_3.png", "view_4.png"])
    prefix = sys.argv[6] if len(sys.argv) >= 7 else "custom4"
    wf = build(files, prefix)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(wf, f, indent=2)
    print(f"wrote {out}: {len(wf['nodes'])} nodes, {wf['last_link_id']} links")
