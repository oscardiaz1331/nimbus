"""Converts an ONNX FP32 graph to FP16 (``pip install onnxconverter-common``)."""
from __future__ import annotations

from pathlib import Path


def convert_fp16(onnx_path: Path, output_path: Path, *, keep_io_types: bool = True) -> Path:
    """Write an FP16 copy of ``onnx_path`` to ``output_path``.

    keep_io_types=True keeps the graph's external inputs/outputs float32 so
    callers can keep feeding plain numpy float32 arrays — only weights and
    internal activations become fp16. Set False for a purely fp16 I/O graph.
    """
    import onnx
    from onnxconverter_common import float16

    print(f"Converting {onnx_path} to FP16 (keep_io_types={keep_io_types})...")

    model = onnx.load(str(onnx_path))
    # Protect existing Cast nodes (common in Ultralytics' end-to-end export with
    # baked-in NMS, and in RF-DETR's wrapper) by NAME, not by adding "Cast" to
    # op_block_list. keep_io_types only protects the model's declared graph
    # outputs; it doesn't know about an internal Cast whose downstream consumer
    # still expects float32, and retargeting that Cast to float16 breaks the
    # type contract there. But blocking the "Cast" *op type* is catastrophic:
    # onnxconverter-common protects a blocked op by wrapping it in new Cast
    # nodes, inserted into the same node list it's still iterating live — so
    # each protective Cast is itself a "Cast" op, gets wrapped in more
    # protective Casts, which get wrapped again, forever. That single-line
    # change turned a <2s conversion into a 30+ minute hang on RF-DETR's
    # ~1600-node DINOv2-backbone graph (node count and time grew in lockstep,
    # unboundedly). Blocking by exact node name sidesteps this entirely: the
    # newly-generated wrapper nodes get distinct names and never re-match.
    cast_node_names = [n.name for n in model.graph.node if n.op_type == "Cast"]

    # Names of Constant nodes that feed directly into an op in
    # DEFAULT_OP_BLOCK_LIST (Resize, TopK, NonMaxSuppression, ...). Those ops
    # require float32/int64 inputs even though onnxconverter-common's
    # op_block_list only protects the *blocked node's own* attributes, not an
    # upstream Constant node producing one of its inputs (Ultralytics'
    # export represents Resize's scale factors that way, not as a plain
    # initializer - process_initializers() already protects true
    # initializers feeding a blocked node, just not Constant-node outputs).
    # Converting that Constant to fp16 anyway produces a graph ONNX Runtime
    # rejects at load time ("Type 'tensor(float16)' of input parameter ...
    # of operator (Resize) ... is invalid"). The block list's own Cast-
    # wrapping safety net (insert_cast32_before_node) can't catch this
    # either: it only wraps inputs that already have a graph.value_info
    # entry, which requires the shape-inference pass this call deliberately
    # skips.
    #
    # Blocked by NODE NAME, not by adding "Constant" to op_block_list:
    # plenty of *other* Constant nodes feed ordinary ops (e.g. an attention
    # block's 1/sqrt(d) scale factor into a Mul) and need converting in
    # lockstep with the rest of the graph - blanket-blocking every Constant
    # left those with one fp16 and one stale-fp32 operand instead, a
    # different type-mismatch error at load time.
    blocked_op_inputs = {
        input_name for node in model.graph.node if node.op_type in float16.DEFAULT_OP_BLOCK_LIST for input_name in node.input
    }
    protected_constant_names = [
        node.name for node in model.graph.node if node.op_type == "Constant" and node.output[0] in blocked_op_inputs
    ]

    # disable_shape_infer=True skips onnxconverter-common's internal shape-inference
    # pass, which is what makes this slow on big graphs (DINOv2/RF-DETR-sized models
    # in particular). Nothing downstream reads value_info off the fp16 model — int8
    # quantization runs on the fp32 graph, and onnxruntime doesn't need it to load or
    # run a model — so we don't re-run shape inference afterward to restore it.
    fp16_model = float16.convert_float_to_float16(
        model,
        keep_io_types=keep_io_types,
        node_block_list=[*cast_node_names, *protected_constant_names],
        disable_shape_infer=True,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(fp16_model, str(output_path))
    return output_path