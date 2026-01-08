#!/usr/bin/env python3
"""
ONNXRuntime inference + SCRFD decode for the "pre-reshape" modified ONNX.

Model assumptions (your modified model):
- Input:  (1, 3, 640, 640) float32
- Outputs: 9 tensors with shapes:
    (80,80,1,2), (40,40,1,2), (20,20,1,2)      -> cls (2 anchors)
    (80,80,1,8), (40,40,1,8), (20,20,1,8)      -> reg (2 anchors * 4 distances)
    (80,80,1,20),(40,40,1,20),(20,20,1,20)     -> kps (2 anchors * 10 coords)

Decode (SCRFD typical):
- reg gives distances (l,t,r,b) from anchor center, in "stride units"
  box = [cx - l*stride, cy - t*stride, cx + r*stride, cy + b*stride]
- kps gives 5 points offsets (dx,dy) from anchor center, in "stride units"
  kp_i = [cx + dx*stride, cy + dy*stride]

Preprocess default (common InsightFace/SCRFD):
- Resize to 640x640 (no letterbox)
- BGR -> RGB
- (x - 127.5) / 128.0
- NCHW float32

Install:
  pip install onnxruntime opencv-python numpy

Run:
  python scrfd_ort_infer.py --model model_fixed.onnx --image test.jpg --save out.jpg

If your model expects BGR (no swap), pass --no-swap-rb.
"""

import argparse
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import numpy as np
import onnxruntime as ort


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def preprocess(
    bgr: np.ndarray,
    size: int = 640,
    mean: Tuple[float, float, float] = (127.5, 127.5, 127.5),
    std: Tuple[float, float, float] = (128.0, 128.0, 128.0),
    swap_rb: bool = True,
) -> Tuple[np.ndarray, Dict]:
    h0, w0 = bgr.shape[:2]
    resized = cv2.resize(bgr, (size, size), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite("det_resized.jpeg", resized)
    if swap_rb:
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    x = resized.astype(np.float32)
    x = (x - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)
    x = np.transpose(x, (2, 0, 1))[None, :, :, :]  # NCHW
    meta = {
        "orig_hw": (h0, w0),
        "in_hw": (size, size),
        "sx": w0 / float(size),
        "sy": h0 / float(size),
    }
    return x.astype(np.float32), meta


def nms_xyxy(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> List[int]:
    """Classic NMS. boxes: (N,4) in xyxy."""
    if boxes.size == 0:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)).astype(np.float32)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)

        inds = np.where(iou <= iou_thr)[0]
        order = order[inds + 1]
    return keep


def to_anchors2_scores(cls_hw1c: np.ndarray) -> np.ndarray:
    """
    cls_hw1c: (H,W,1,2)
    ONNX 'Reshape' in your original graph effectively produced (H*W*2, 1)
    by flattening and grouping by 1 element per row.
    We can do exactly: reshape(-1, 1).
    """
    assert cls_hw1c.ndim == 4 and cls_hw1c.shape[2] == 1 and cls_hw1c.shape[3] == 2
    return cls_hw1c.reshape((-1, 1))


def to_anchors2_reg(reg_hw1c: np.ndarray) -> np.ndarray:
    """
    reg_hw1c: (H,W,1,8) -> (H*W*2, 4)
    """
    assert reg_hw1c.ndim == 4 and reg_hw1c.shape[2] == 1 and reg_hw1c.shape[3] == 8
    return reg_hw1c.reshape((-1, 4))


def to_anchors2_kps(kps_hw1c: np.ndarray) -> np.ndarray:
    """
    kps_hw1c: (H,W,1,20) -> (H*W*2, 10)
    """
    assert kps_hw1c.ndim == 4 and kps_hw1c.shape[2] == 1 and kps_hw1c.shape[3] == 20
    return kps_hw1c.reshape((-1, 10))


def make_centers(h: int, w: int, stride: int) -> np.ndarray:
    ys = np.arange(h, dtype=np.float32) * float(stride)
    xs = np.arange(w, dtype=np.float32) * float(stride)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")  # (H,W)
    centers = np.stack([xx, yy], axis=-1).reshape((-1, 2))
    return centers


@dataclass
class Detection:
    box_xyxy: np.ndarray  # (4,)
    kps: np.ndarray  # (5,2)
    score: float


def decode_scrfd_one_scale(
    cls_hw1c: np.ndarray,
    reg_hw1c: np.ndarray,
    kps_hw1c: np.ndarray,
    stride: int,
    score_thr: float,
    apply_sigmoid: bool,
) -> List[Detection]:
    h, w, _, _ = cls_hw1c.shape

    scores = to_anchors2_scores(cls_hw1c).astype(np.float32)  # (H*W*2,1)
    if apply_sigmoid:
        scores = sigmoid(scores)

    scores = scores[:, 0]  # (N,)
    keep = scores > score_thr
    if not np.any(keep):
        return []

    reg = to_anchors2_reg(reg_hw1c).astype(np.float32)  # (H*W*2,4)
    kps = to_anchors2_kps(kps_hw1c).astype(np.float32)  # (H*W*2,10)

    # centers per location, then expand for 2 anchors: repeat each center twice.
    centers_hw = make_centers(h, w, stride)  # (H*W,2)
    centers = np.repeat(centers_hw, repeats=2, axis=0)  # (H*W*2,2)
    cx = centers[:, 0]
    cy = centers[:, 1]

    # Decode boxes from distances (l,t,r,b) in stride units.
    l = reg[:, 0] * stride
    t = reg[:, 1] * stride
    r = reg[:, 2] * stride
    b = reg[:, 3] * stride
    x1 = cx - l
    y1 = cy - t
    x2 = cx + r
    y2 = cy + b
    boxes = np.stack([x1, y1, x2, y2], axis=-1)

    # Decode 5 keypoints: offsets in stride units.
    kps_xy = kps.reshape((-1, 5, 2))
    kps_xy[:, :, 0] = cx[:, None] + kps_xy[:, :, 0] * stride
    kps_xy[:, :, 1] = cy[:, None] + kps_xy[:, :, 1] * stride

    # Filter by score
    idxs = np.where(keep)[0]
    dets: List[Detection] = []
    for i in idxs:
        dets.append(Detection(box_xyxy=boxes[i], kps=kps_xy[i], score=float(scores[i])))
    return dets


def draw_dets(bgr: np.ndarray, dets: List[Detection]) -> np.ndarray:
    out = bgr.copy()
    for d in dets:
        x1, y1, x2, y2 = d.box_xyxy.astype(int)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            out,
            f"{d.score:.2f}",
            (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
        for kx, ky in d.kps.astype(int):
            cv2.circle(out, (kx, ky), 2, (0, 0, 255), -1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--providers", nargs="*", default=["CPUExecutionProvider"])
    ap.add_argument("--score-thr", type=float, default=0.5)
    ap.add_argument("--nms-iou", type=float, default=0.4)
    ap.add_argument("--topk", type=int, default=5000, help="pre-NMS topk by score")
    ap.add_argument("--max-dets", type=int, default=200)
    ap.add_argument("--no-swap-rb", action="store_true", help="Do not BGR->RGB swap")
    ap.add_argument("--save", default="", help="Save visualization path (optional)")
    args = ap.parse_args()

    # Load image
    bgr0 = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if bgr0 is None:
        raise RuntimeError(f"Failed to read image: {args.image}")

    x, meta = preprocess(bgr0, size=640, swap_rb=(not args.no_swap_rb))

    # ORT session
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(args.model, sess_options=so, providers=args.providers)

    in_name = sess.get_inputs()[0].name
    outs = sess.run(None, {in_name: x})
    out_infos = sess.get_outputs()
    outputs: Dict[str, np.ndarray] = {
        info.name: arr for info, arr in zip(out_infos, outs)
    }

    # Group outputs by (H,W,C) and pick cls/reg/kps per scale
    # We identify by last dim: 2=cls, 8=reg, 20=kps.
    per_scale: Dict[Tuple[int, int], Dict[int, np.ndarray]] = {}
    for name, arr in outputs.items():
        if arr.ndim != 4:
            continue
        h, w, one, c = arr.shape
        if one != 1 or h != w:
            continue
        per_scale.setdefault((h, w), {})[c] = arr

    # Determine whether sigmoid is needed:
    # If scores are already probabilities, they should be ~[0,1].
    # We'll check the 80x80 cls tensor if available.
    apply_sigmoid = False
    cls_probe = None
    if (80, 80) in per_scale and 2 in per_scale[(80, 80)]:
        cls_probe = to_anchors2_scores(per_scale[(80, 80)][2]).astype(np.float32)
    else:
        # fallback: pick any cls tensor
        for (h, w), m in per_scale.items():
            if 2 in m:
                cls_probe = to_anchors2_scores(m[2]).astype(np.float32)
                break
    if cls_probe is not None:
        mn = float(cls_probe.min())
        mx = float(cls_probe.max())
        # Heuristic: if outside [0,1] by a margin, treat as logits.
        if mn < -0.1 or mx > 1.1:
            apply_sigmoid = True
        print(
            f"[INFO] cls probe range: min={mn:.3f}, max={mx:.3f} -> apply_sigmoid={apply_sigmoid}"
        )
    else:
        print("[WARN] Could not probe cls output; assuming apply_sigmoid=False")

    # Decode each scale (80->stride8, 40->stride16, 20->stride32)
    all_dets: List[Detection] = []
    for (h, w), m in sorted(per_scale.items(), key=lambda kv: kv[0][0], reverse=True):
        if 2 not in m or 8 not in m or 20 not in m:
            continue
        stride = 640 // h  # 80->8, 40->16, 20->32
        dets = decode_scrfd_one_scale(
            cls_hw1c=m[2],
            reg_hw1c=m[8],
            kps_hw1c=m[20],
            stride=stride,
            score_thr=args.score_thr,
            apply_sigmoid=apply_sigmoid,
        )
        print(
            f"[INFO] scale {h}x{w} stride={stride}: dets={len(dets)} above thr={args.score_thr}"
        )
        all_dets.extend(dets)

    if not all_dets:
        print("[INFO] No detections.")
        return

    # Convert to arrays for NMS
    boxes = np.stack([d.box_xyxy for d in all_dets], axis=0).astype(np.float32)
    scores = np.array([d.score for d in all_dets], dtype=np.float32)

    # TopK prefilter
    if boxes.shape[0] > args.topk:
        idx = np.argsort(scores)[::-1][: args.topk]
        boxes = boxes[idx]
        scores = scores[idx]
        all_dets = [all_dets[int(i)] for i in idx]

    keep = nms_xyxy(boxes, scores, args.nms_iou)
    keep = keep[: args.max_dets]
    final = [all_dets[i] for i in keep]

    # Map back to original image size (since we resized directly 640->orig)
    sx = meta["sx"]
    sy = meta["sy"]
    for d in final:
        d.box_xyxy = np.array(
            [
                d.box_xyxy[0] * sx,
                d.box_xyxy[1] * sy,
                d.box_xyxy[2] * sx,
                d.box_xyxy[3] * sy,
            ],
            dtype=np.float32,
        )
        d.kps = np.stack([d.kps[:, 0] * sx, d.kps[:, 1] * sy], axis=-1).astype(
            np.float32
        )

    print(f"[INFO] Final detections after NMS: {len(final)}")
    for i, d in enumerate(final[:10]):
        x1, y1, x2, y2 = d.box_xyxy
        print(
            f"  #{i:02d} score={d.score:.3f} box=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f})"
        )

    if args.save:
        vis = draw_dets(bgr0, final)
        cv2.imwrite(args.save, vis)
        print("[INFO] Saved:", args.save)


if __name__ == "__main__":
    main()
