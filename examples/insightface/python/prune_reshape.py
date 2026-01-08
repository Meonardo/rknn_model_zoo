#!/usr/bin/env python3
"""
Cut ONNX outputs upstream of specified nodes (Reshape/Transpose), prune dead nodes,
force static input shape, run shape inference, and (optionally) force output shapes
to the NCHW conv outputs for SCRFD.

This version fixes your new RKNN issue:
  Conv (NPU) -> Transpose (CPU) -> Output
by cutting outputs BEFORE Transpose, so outputs become (1,C,H,W) and Transpose disappears.

Usage:
  python onnx_cut_prune_fix_shapes_v3.py in.onnx out.onnx \
    --input-shape 1 3 640 640 \
    --cut-nodes Reshape_... Transpose_... \
    --force-outputs-nchw
"""

import argparse
import sys
from collections import defaultdict, deque
from typing import Dict, List, Set

import onnx
from onnx import helper, shape_inference, TensorProto


def set_valueinfo_shape(vi: onnx.ValueInfoProto, dims: List[int]) -> None:
    t = vi.type.tensor_type
    if not t.HasField("shape"):
        t.shape.CopyFrom(onnx.TensorShapeProto())
    del t.shape.dim[:]
    for d in dims:
        dim = t.shape.dim.add()
        dim.dim_value = int(d)


def find_node_by_name(graph: onnx.GraphProto, name: str):
    for n in graph.node:
        if n.name == name:
            return n
    return None


def build_consumers(graph: onnx.GraphProto) -> Dict[str, List[onnx.NodeProto]]:
    consumers = defaultdict(list)
    for n in graph.node:
        for inp in n.input:
            if inp:
                consumers[inp].append(n)
    return consumers


def any_output_reachable_from(
    start_tensor: str,
    consumers: Dict[str, List[onnx.NodeProto]],
    output_names: Set[str],
) -> Set[str]:
    """Walk forward from a tensor and collect any graph outputs reachable."""
    reachable_outputs: Set[str] = set()
    q = deque([start_tensor])
    seen: Set[str] = set([start_tensor])

    while q:
        t = q.popleft()
        if t in output_names:
            reachable_outputs.add(t)
        for node in consumers.get(t, []):
            for out_t in node.output:
                if out_t and out_t not in seen:
                    seen.add(out_t)
                    q.append(out_t)
    return reachable_outputs


def prune_to_outputs(model: onnx.ModelProto) -> onnx.ModelProto:
    """Keep only nodes/initializers needed to compute current graph outputs."""
    graph = model.graph
    needed_tensors: Set[str] = set(o.name for o in graph.output)

    kept_nodes_rev: List[onnx.NodeProto] = []
    for node in reversed(graph.node):
        if any(o in needed_tensors for o in node.output if o):
            kept_nodes_rev.append(node)
            for inp in node.input:
                if inp:
                    needed_tensors.add(inp)

    kept_nodes = list(reversed(kept_nodes_rev))

    init_by_name = {i.name: i for i in graph.initializer}
    kept_inits = [init_by_name[name] for name in init_by_name if name in needed_tensors]

    new_graph = helper.make_graph(
        nodes=kept_nodes,
        name=graph.name,
        inputs=list(graph.input),
        outputs=list(graph.output),
        initializer=kept_inits,
    )

    new_model = helper.make_model(
        new_graph, producer_name="onnx_cut_prune_fix_shapes_v3.py"
    )
    new_model.ir_version = model.ir_version
    del new_model.opset_import[:]
    new_model.opset_import.extend(model.opset_import)
    return new_model


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("input_onnx")
    ap.add_argument("output_onnx")
    ap.add_argument(
        "--input-shape",
        nargs=4,
        type=int,
        required=True,
        help="Static input shape: N C H W, e.g. 1 3 640 640",
    )
    ap.add_argument(
        "--cut-nodes",
        nargs="+",
        required=True,
        help="Node names (Reshape/Transpose/etc). Any graph outputs reachable from these nodes' outputs "
        "will be replaced by the node's data input tensor (input[0]).",
    )
    ap.add_argument(
        "--force-outputs-nchw",
        action="store_true",
        help="Force 9 SCRFD outputs in order to NCHW conv shapes: "
        "(1,2,80,80),(1,8,80,80),(1,20,80,80), ... ,(1,20,20,20)",
    )
    args = ap.parse_args()

    model = onnx.load(args.input_onnx)
    g = model.graph

    if len(g.input) == 0:
        print("[ERROR] Model has no inputs.", file=sys.stderr)
        return 2

    # ----------------------------------------------------------------------
    # (A) Replace graph outputs downstream of specified nodes with node.input[0]
    #     Works for both Reshape and Transpose (and any other node type).
    # ----------------------------------------------------------------------
    consumers = build_consumers(g)
    output_names = set(o.name for o in g.output)

    replacements: Dict[str, str] = {}  # old_output_name -> new_tensor_name

    for nname in args.cut_nodes:
        node = find_node_by_name(g, nname)
        if node is None:
            print(f"[WARN] Cut node not found: {nname}", file=sys.stderr)
            continue
        if len(node.input) < 1 or len(node.output) < 1:
            print(
                f"[WARN] Node {nname} missing inputs/outputs, skipping", file=sys.stderr
            )
            continue

        pre_in = node.input[0]  # tensor BEFORE this node (what we want as output)
        post_out = node.output[0]  # tensor AFTER this node (current tail)

        reached = any_output_reachable_from(post_out, consumers, output_names)
        if not reached:
            print(
                f"[INFO] Node {nname} output {post_out} does not reach any current graph outputs.",
                file=sys.stderr,
            )
            continue

        for out_name in reached:
            replacements[out_name] = pre_in

    if not replacements:
        print(
            "[ERROR] None of the specified cut nodes reached any graph outputs.",
            file=sys.stderr,
        )
        return 2

    old_outputs = list(g.output)
    new_outputs = []
    replaced_cnt = 0
    for out in old_outputs:
        if out.name in replacements:
            new_name = replacements[out.name]
            elem_type = out.type.tensor_type.elem_type
            if elem_type == TensorProto.UNDEFINED:
                elem_type = TensorProto.FLOAT
            vi = helper.make_tensor_value_info(new_name, elem_type, None)
            new_outputs.append(vi)
            replaced_cnt += 1
        else:
            new_outputs.append(out)

    del g.output[:]
    g.output.extend(new_outputs)

    # ----------------------------------------------------------------------
    # (B) Prune dead nodes (removes Transpose/Reshape tails)
    # ----------------------------------------------------------------------
    model = prune_to_outputs(model)
    g = model.graph

    print(f"[INFO] Replaced {replaced_cnt} graph output(s) with upstream tensors:")
    for old_out, new_tensor in replacements.items():
        print(f"  {old_out} -> {new_tensor}")

    # ----------------------------------------------------------------------
    # (C) Force static input shape
    # ----------------------------------------------------------------------
    set_valueinfo_shape(g.input[0], list(args.input_shape))

    # ----------------------------------------------------------------------
    # (D) Shape inference (best-effort)
    # ----------------------------------------------------------------------
    try:
        model = shape_inference.infer_shapes(model)
        g = model.graph
        print("[INFO] Shape inference succeeded.")
    except Exception as e:
        print(f"[WARN] Shape inference failed (often OK): {e}", file=sys.stderr)

    # ----------------------------------------------------------------------
    # (E) Force SCRFD NCHW output shapes (optional)
    #     We apply in *current graph.output order* (keeps your existing ordering).
    # ----------------------------------------------------------------------
    if args.force_outputs_nchw:
        forced_shapes = [
            [1, 2, 80, 80],
            [1, 8, 80, 80],
            [1, 20, 80, 80],
            [1, 2, 40, 40],
            [1, 8, 40, 40],
            [1, 20, 40, 40],
            [1, 2, 20, 20],
            [1, 8, 20, 20],
            [1, 20, 20, 20],
        ]
        if len(g.output) != 9:
            print(
                f"[WARN] Expected 9 outputs for SCRFD, got {len(g.output)}. "
                f"Not forcing shapes.",
                file=sys.stderr,
            )
        else:
            for out, shp in zip(g.output, forced_shapes):
                set_valueinfo_shape(out, shp)
                t = out.type.tensor_type
                if t.elem_type == TensorProto.UNDEFINED:
                    t.elem_type = TensorProto.FLOAT
            print(
                "[INFO] Forced NCHW output shapes for 9 SCRFD heads (by output order)."
            )

    onnx.save(model, args.output_onnx)
    print("[DONE] Saved:", args.output_onnx)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# python prune_reshape.py ../model/buffalo_s/det_500m_fixed.onnx ../model/buffalo_s/det_500m_pruned.onnx --input-shape 1 3 640 640 --reshape Reshape_160 Reshape_163 Reshape_156 Reshape_185 Reshape_188 Reshape_181 Reshape_210 Reshape_213 Reshape_206 --force-output-shapes-by-name

# python prune_reshape.py ../model/buffalo_s/det_500m_fixed.onnx ../model/buffalo_s/det_500m_pruned.onnx --input-shape 1 3 640 640 --cut-nodes Reshape_160 Reshape_163 Reshape_156 Reshape_185 Reshape_188 Reshape_181 Reshape_210 Reshape_213 Reshape_206 Transpose_154 Transpose_158 Transpose_161 Transpose_179 Transpose_183 Transpose_186 Transpose_204 Transpose_208 Transpose_211 --force-outputs-nchw