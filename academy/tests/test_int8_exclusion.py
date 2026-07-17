"""Tests for int8_converter's postprocessing-tail exclusion — the pure-graph
logic that keeps QDQ quantization away from YOLO's mixed-range decode head
(coordinates + confidences + mask coefficients in one tensor). Needs only
onnx; quantize_static itself is an integration concern, same as before.
"""
from __future__ import annotations

import unittest

from onnx import TensorProto, helper

from utils.optimizers.int8_converter import _ensure_node_names, postprocess_tail_node_names


def _model(nodes):
    inp = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
    out = helper.make_tensor_value_info("out", TensorProto.FLOAT, [1])
    # no checker run: the tail walk only reads op_type and input/output names,
    # so dangling weight inputs like "w1" are fine here.
    return helper.make_model(helper.make_graph(nodes, "g", [inp], [out]))


class TestPostprocessTailNodeNames(unittest.TestCase):
    def test_nodes_after_last_compute_op_are_the_tail(self):
        model = _model([
            helper.make_node("Conv", ["x", "w1"], ["c1"], name="conv1"),
            helper.make_node("Relu", ["c1"], ["r1"], name="relu1"),
            helper.make_node("Conv", ["r1", "w2"], ["c2"], name="conv2"),
            helper.make_node("Sigmoid", ["c2"], ["s1"], name="sig_tail"),
            helper.make_node("Concat", ["c2", "s1"], ["out"], name="concat_tail", axis=0),
        ])
        self.assertEqual(set(postprocess_tail_node_names(model)), {"sig_tail", "concat_tail"})

    def test_non_compute_node_feeding_a_matmul_is_not_tail(self):
        model = _model([
            helper.make_node("Conv", ["x", "w1"], ["c1"], name="conv1"),
            helper.make_node("Sigmoid", ["c1"], ["s1"], name="sig_mid"),
            helper.make_node("MatMul", ["s1", "w2"], ["m1"], name="matmul1"),
            helper.make_node("Relu", ["m1"], ["out"], name="relu_tail"),
        ])
        self.assertEqual(set(postprocess_tail_node_names(model)), {"relu_tail"})

    def test_graph_without_compute_ops_is_all_tail(self):
        model = _model([helper.make_node("Identity", ["x"], ["out"], name="id1")])
        self.assertEqual(postprocess_tail_node_names(model), ["id1"])


class TestEnsureNodeNames(unittest.TestCase):
    def test_unnamed_nodes_get_unique_names_and_tail_stays_addressable(self):
        model = _model([
            helper.make_node("Conv", ["x", "w1"], ["c1"]),  # unnamed
            helper.make_node("Sigmoid", ["c1"], ["out"]),  # unnamed tail
        ])
        _ensure_node_names(model)
        names = [n.name for n in model.graph.node]
        self.assertTrue(all(names))
        self.assertEqual(len(set(names)), len(names))
        # the whole point: an unnamed tail node must end up excludable by name
        self.assertEqual(len(postprocess_tail_node_names(model)), 1)
        self.assertTrue(postprocess_tail_node_names(model)[0])

    def test_generated_name_collision_is_avoided(self):
        model = _model([
            helper.make_node("Conv", ["x", "w1"], ["c1"], name="Sigmoid_1"),
            helper.make_node("Sigmoid", ["c1"], ["out"]),  # unnamed, index 1
        ])
        _ensure_node_names(model)
        names = [n.name for n in model.graph.node]
        self.assertEqual(len(set(names)), len(names))
        self.assertTrue(all(names))


if __name__ == "__main__":
    unittest.main()
