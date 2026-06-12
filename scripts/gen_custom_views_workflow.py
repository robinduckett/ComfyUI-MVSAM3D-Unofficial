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
        "segmentation pack into the mask chain instead.\n\n"
        "No depth-estimate step is needed: the MV-SAM3D pipeline computes pointmaps "
        "itself (MoGe).\n\n"
        "weight_source=entropy is the paper method. 'visibility'/'mixed' need a "
        "Depth-Anything-3 npz on da3_npz.\n\n"
        "View order = batch order; put your reference/front view first. Images of "
        "different sizes are auto-resized to the first view's size by Image Batch."
    )
    node(90, "Note", [-1560, -700], [460, 240], [], [], [note],
         props={}, color="#432", bgcolor="#653")

    node(1, "UpscaleModelLoader", [-1560, -420], [315, 60],
         [],
         [{"name": "UPSCALE_MODEL", "type": "UPSCALE_MODEL", "links": []}],
         ["RealESRGAN_x4plus.pth"])

    img_out, maskimg_out = [], []  # (node_id, slot) per view
    y = -300
    for i, fname in enumerate(filenames):
        li, ui, mi, ti, si = 100 + i * 10, 101 + i * 10, 102 + i * 10, 103 + i * 10, 104 + i * 10
        node(li, "LoadImage", [-1560, y], [274, 314],
             [],
             [{"name": "IMAGE", "type": "IMAGE", "links": []},
              {"name": "MASK", "type": "MASK", "links": []}],
             [fname, "image"])
        node(ui, "ImageUpscaleWithModel", [-1240, y], [240, 46],
             [{"name": "upscale_model", "type": "UPSCALE_MODEL",
               "link": link(1, 0, ui, 0, "UPSCALE_MODEL")},
              {"name": "image", "type": "IMAGE", "link": link(li, 0, ui, 1, "IMAGE")}],
             [{"name": "IMAGE", "type": "IMAGE", "links": []}],
             [])
        node(mi, "InvertMask", [-1240, y + 80], [210, 26],
             [{"name": "mask", "type": "MASK", "link": link(li, 1, mi, 0, "MASK")}],
             [{"name": "MASK", "type": "MASK", "links": []}],
             [])
        node(ti, "MaskToImage", [-1240, y + 140], [210, 26],
             [{"name": "mask", "type": "MASK", "link": link(mi, 0, ti, 0, "MASK")}],
             [{"name": "IMAGE", "type": "IMAGE", "links": []}],
             [])
        node(si, "ImageScaleBy", [-1240, y + 200], [240, 82],
             [{"name": "image", "type": "IMAGE", "link": link(ti, 0, si, 0, "IMAGE")}],
             [{"name": "IMAGE", "type": "IMAGE", "links": []}],
             ["bilinear", 4.0])
        img_out.append((ui, 0))
        maskimg_out.append((si, 0))
        y += 340

    def chain_batches(sources, base_id, x):
        prev = sources[0]
        for k in range(1, len(sources)):
            bid = base_id + k
            node(bid, "ImageBatch", [x, -260 + k * 110], [210, 46],
                 [{"name": "image1", "type": "IMAGE",
                   "link": link(prev[0], prev[1], bid, 0, "IMAGE")},
                  {"name": "image2", "type": "IMAGE",
                   "link": link(sources[k][0], sources[k][1], bid, 1, "IMAGE")}],
                 [{"name": "IMAGE", "type": "IMAGE", "links": []}],
                 [])
            prev = (bid, 0)
        return prev

    img_batch = chain_batches(img_out, 200, -940)
    mask_batch_img = chain_batches(maskimg_out, 210, -940)

    node(220, "ImageToMask", [-700, 160], [240, 60],
         [{"name": "image", "type": "IMAGE",
           "link": link(mask_batch_img[0], mask_batch_img[1], 220, 0, "IMAGE")}],
         [{"name": "MASK", "type": "MASK", "links": []}],
         ["red"])

    node(300, "MVSAM3DLoadViews", [-700, -160], [300, 130],
         [{"name": "images", "type": "IMAGE",
           "link": link(img_batch[0], img_batch[1], 300, 0, "IMAGE")},
          {"name": "masks", "type": "MASK", "link": link(220, 0, 300, 1, "MASK")}],
         [{"name": "scene", "type": "MVSAM3D_SCENE", "links": []}],
         [prefix, prefix], props=ours("MVSAM3DLoadViews"))

    node(301, "MVSAM3DRunMultiView", [-360, -300], [360, 560],
         [{"name": "scene", "type": "MVSAM3D_SCENE",
           "link": link(300, 0, 301, 0, "MVSAM3D_SCENE")}],
         [{"name": "result", "type": "MVSAM3D_RESULT", "links": []},
          {"name": "glb_path", "type": "STRING", "links": []},
          {"name": "ply_path", "type": "STRING", "links": []}],
         [42, "fixed", 50, 25, True, 9, 30.0, True, "entropy", 30.0, 30.0, 6, 0,
          0.001, "average", "gaussian,mesh", prefix, True, "", "", "", ""],
         props=ours("MVSAM3DRunMultiView"))

    node(302, "MVSAM3DExport", [40, -300], [240, 80],
         [{"name": "result", "type": "MVSAM3D_RESULT",
           "link": link(301, 0, 302, 0, "MVSAM3D_RESULT")}],
         [{"name": "path", "type": "STRING", "links": []}],
         ["glb"], props=ours("MVSAM3DExport"))

    node(303, "Preview3D", [320, -300], [400, 440],
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
