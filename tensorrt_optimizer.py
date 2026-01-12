#!/usr/bin/env python3
# coding=utf-8
"""
- filename: tensorrt_optimizer.py
- description: Optimize graph for tensorrt inference
- author: qiuchuyu@kuaishou.com
- date: 2020-12-24 19:19:00
"""
import base64
import tensorflow as tf
from .graph_transforms import TransformGraph
from tensorflow.core.framework import types_pb2 as tpb
from tensorflow.core.framework import node_def_pb2 as npb
from tensorflow.core.framework import graph_pb2 as gpb
from tensorflow.core.framework import tensor_pb2 as tensorpb
from tensorflow.python.framework import tensor_util
from .miotf_util import *


def FindNodeInfo(graph, node_name):
    node_name = node_name.split(":")[0]
    for idx, node in enumerate(graph.node):
        if node.name == node_name:
            return idx, node


def IsCompressIndexIndices(node_name, graph_def):
    if node_name.find("COMPRESS_INDEX_") != -1:
        return True

    idx, node = FindNodeInfo(graph_def, node_name)
    if node.op == "Cast":
        if node.input[0].find("COMPRESS_INDEX_") != -1:
            return True

    return False


def IsCompressIndexGatherDirect(node, graph_def):
    if node.op == "GatherV2" and IsCompressIndexIndices(node.input[1], graph_def):
        return True
    return False


def IsCompressIndexGather(node_name, graph_def):
    idx, node = FindNodeInfo(graph_def, node_name)
    return IsCompressIndexGatherDirect(node, graph_def), node


def FindNodeWithCompressIndexInput(graph_def):
    for idx, node in enumerate(graph_def.node):
        for input_name in node.input:
            succ, node = IsCompressIndexGather(input_name, graph_def)
            if succ:
                print(
                    f"node: {node.name}, op: {node.op}  has COMPRESS_INDEX input: {node.input[0]}"
                )


def RemoveMioVariable(graph_def):
    # Replace VariableFromMioComponentTable to placeholder
    # In convenience of optimizing
    mio_address_set = set()
    mio_variable = {}

    # find all MioVariable and MioAddress
    new_graph = gpb.GraphDef()
    new_graph.CopyFrom(graph_def)
    del new_graph.node[:]

    # add all nodes and substitute MioVariable
    for node in graph_def.node:
        if node.op == "VariableFromMioComponentTable":
            mio_variable[node.name] = node
            assert len(node.input) == 1
            if node.input[0] not in mio_address_set:
                mio_address_set.add(node.input[0])
            placeholder = npb.NodeDef()
            placeholder.name = node.name
            placeholder.op = "Placeholder"
            placeholder.attr["dtype"].type = tpb.DataType.DT_FLOAT
            placeholder.attr["shape"].CopyFrom(node.attr["shape"])
            new_graph.node.append(placeholder)
        else:
            new_graph.node.append(node)

    # remove all MioAddress
    index_list = []
    idx = 0
    mio_address = {}
    for node in new_graph.node:
        if node.name in mio_address_set:
            index_list.append(idx)
            mio_address[node.name] = node
        idx += 1

    for i in sorted(index_list, reverse=True):
        del new_graph.node[i]

    return new_graph, mio_variable, mio_address


def AllInputSame(inputs):
    for name in inputs[:-1]:
        if name != inputs[0]:
            return False
    return True


def RemoveSpecificNode(graph_def, node_map, reserve_node):
    """从后往前删除 graph_def 中指定的 node"""
    sorted_node_keys = sorted(node_map, key=node_map.get, reverse=True)
    for key in sorted_node_keys:
        if key in reserve_node and node_map[key] is not None:
            continue
        del graph_def.node[node_map[key]]


def FindPatternReduceSum(graph_def):
    node_map = {
        "expand_dim_in": None,
        "expert_in": None,
        "split": None,
        "concat": None,
        "mul": None,
        "reduce_sum": None,
    }
    for idx, node in enumerate(graph_def.node):
        if node.op != "Sum":
            continue
        node_map["reduce_sum"] = idx

        if len(node.input) != 2:
            continue
        _, reduce_indice_node = FindNodeInfo(graph_def, node.input[1])
        if reduce_indice_node.op != "Const":
            continue
        if len(reduce_indice_node.attr["value"].tensor.int_val) != 1:
            continue
        reduce_indice = reduce_indice_node.attr["value"].tensor.int_val[0]
        if reduce_indice != 2:
            continue

        mul_idx, mul_node = FindNodeInfo(graph_def, node.input[0])
        if mul_node.op != "Mul":
            continue
        node_map["mul"] = mul_idx

        # mul input 0 is expert_in concat
        expert_in_concat_idx, expert_in_concat_node = FindNodeInfo(
            graph_def, mul_node.input[0]
        )
        if expert_in_concat_node.op != "ConcatV2":
            continue
        concat_input_num = expert_in_concat_node.attr["N"].i
        if concat_input_num + 1 != len(expert_in_concat_node.input):
            continue
        _, concat_axis_node = FindNodeInfo(
            graph_def, expert_in_concat_node.input[concat_input_num]
        )
        if concat_axis_node.op != "Const":
            continue
        concat_axis_num = concat_axis_node.attr["value"].tensor.int_val[0]
        if concat_axis_num != 2:
            continue
        node_map["expert_in"] = expert_in_concat_idx

        # mul input 1 is concat
        concat_idx, concat_node = FindNodeInfo(graph_def, mul_node.input[1])
        if concat_node.op != "ConcatV2":
            continue
        concat_input_num = concat_node.attr["N"].i
        if concat_input_num + 1 != len(concat_node.input):
            continue
        if not AllInputSame(concat_node.input):
            continue
        _, concat_axis_node = FindNodeInfo(
            graph_def, concat_node.input[concat_input_num]
        )
        if concat_axis_node.op != "Const":
            continue
        concat_axis_num = concat_axis_node.attr["value"].tensor.int_val[0]
        if concat_axis_num != 1:
            continue
        node_map["concat"] = concat_idx

        # concat input is split
        split_idx, split_node = FindNodeInfo(graph_def, concat_node.input[0])
        if split_node.op != "Split":
            continue
        node_map["split"] = split_idx

        # split input 1 is expand_dim
        expand_dim_idx, expand_dim_node = FindNodeInfo(graph_def, split_node.input[1])
        if expand_dim_node.op != "ExpandDims":
            continue
        node_map["expand_dim_in"] = expand_dim_idx
        break
    if node_map["expand_dim_in"] == None or node_map["expert_in"] == None:
        return None
    return node_map


def SpecialRuleReplaceReduceSum(graph_def):
    while True:
        node_map = FindPatternReduceSum(graph_def)
        if node_map == None:
            break
        print(node_map)

        ## add transpose perm node
        # transpose perm tensor
        new_tensor = tensorpb.TensorProto()
        new_tensor.dtype = tpb.DataType.DT_INT32
        new_tensor.int_val.append(0)
        new_tensor.int_val.append(2)
        new_tensor.int_val.append(1)
        new_tensor.tensor_shape.dim.add().size = 3
        # transpose perm node
        new_transpose_perm_node = npb.NodeDef()
        new_transpose_perm_node.op = "Const"
        new_transpose_perm_node.name = (
            graph_def.node[node_map["reduce_sum"]].name + "_transpose/perm"
        )
        new_transpose_perm_node.attr["dtype"].type = tpb.DataType.DT_INT32
        new_transpose_perm_node.attr["value"].tensor.CopyFrom(new_tensor)

        # add transpose node
        new_transpose_node = npb.NodeDef()
        new_transpose_node.name = (
            graph_def.node[node_map["reduce_sum"]].name + "_transpose"
        )
        new_transpose_node.op = "Transpose"
        new_transpose_node.input.append(graph_def.node[node_map["expert_in"]].name)
        new_transpose_node.input.append(new_transpose_perm_node.name)
        new_transpose_node.attr["T"].type = tpb.DataType.DT_FLOAT
        new_transpose_node.attr["Tperm"].type = tpb.DataType.DT_INT32

        # add fused gated sum node
        new_fused_sum_node = npb.NodeDef()
        new_fused_sum_node.name = (
            graph_def.node[node_map["reduce_sum"]].name + "_FusedGatedSum"
        )
        new_fused_sum_node.op = "FusedGatedSum"
        new_fused_sum_node.input.append(new_transpose_node.name)
        new_fused_sum_node.input.append(graph_def.node[node_map["expand_dim_in"]].name)
        new_fused_sum_node.attr["T"].type = tpb.DataType.DT_FLOAT

        # replace pattern's output with new output
        for node in graph_def.node:
            for idx, name in enumerate(node.input):
                if name == graph_def.node[node_map["reduce_sum"]].name:
                    node.input[idx] = new_fused_sum_node.name

        # remove pattern node
        reserve_node = ["expand_dim_in", "expert_in"]
        RemoveSpecificNode(graph_def, node_map, reserve_node)
        graph_def.node.append(new_transpose_perm_node)
        graph_def.node.append(new_transpose_node)
        graph_def.node.append(new_fused_sum_node)
        print("Add FusedGatedSum node: ", new_fused_sum_node.name)
    return graph_def


def FindPatternMultiheadAttention(graph_def):
    node_map = {
        "querys_in": None,
        "keys_in": None,
        "values_in": None,
        "querys_split": None,
        "keys_split": None,
        "values_split": None,
        "querys_pack": None,
        "keys_pack": None,
        "values_pack": None,
        "qk_matmul": None,
        "real_div": None,
        "softmax": None,
        "qkv_matmul": None,
        "transpose": None,
        "reshape": None,
    }
    nh, emb_size, scale = None, None, None
    for idx, node in enumerate(graph_def.node):
        if node.op != "Reshape":
            continue
        node_map["reshape"] = idx
        transpose_idx, transpose_node = FindNodeInfo(graph_def, node.input[0])
        if transpose_node.op != "Transpose":
            continue
        _, shape_node = FindNodeInfo(graph_def, node.input[1])
        # print(shape_node)
        if shape_node.op != "Const" and shape_node.op != "Pack":
            continue
        if shape_node.op == "Const":
            shape_tensor = tf.make_ndarray(shape_node.attr["value"].tensor)
            if len(shape_tensor) != 2:
                continue
            feature_length = shape_tensor[1]
        else:
            if len(shape_node.input) != 2:
                continue
            _, shape = FindNodeInfo(graph_def, shape_node.input[1])
            if shape.op != "Const":
                continue
            feature_length = tf.make_ndarray(shape.attr["value"].tensor)
        node_map["transpose"] = transpose_idx

        qkv_matmul_idx, qkv_matmul_node = FindNodeInfo(
            graph_def, transpose_node.input[0]
        )
        if (
            qkv_matmul_node.op != "BatchMatMul"
            and qkv_matmul_node.op != "BatchMatMulV2"
        ):
            continue
        node_map["qkv_matmul"] = qkv_matmul_idx

        softmax_idx, softmax_node = FindNodeInfo(graph_def, qkv_matmul_node.input[0])
        values_idx, values_pack_node = FindNodeInfo(graph_def, qkv_matmul_node.input[1])
        if softmax_node.op != "Softmax" or values_pack_node.op != "Pack":
            continue
        node_map["softmax"] = softmax_idx
        node_map["values_pack"] = values_idx

        add_bias_idx, add_bias_node = FindNodeInfo(graph_def, softmax_node.input[0])
        real_div_input_node = softmax_node.input[0]
        if add_bias_node.op == "Add" or add_bias_node.op == "AddV2":
            expand_bias_idx, expand_bias_node = FindNodeInfo(
                graph_def, add_bias_node.input[1]
            )
            if expand_bias_node.op == "ExpandDims":
                bias_input_idx, bias_input_node = FindNodeInfo(
                    graph_def, expand_bias_node.input[0]
                )
                node_map["expand_bias"] = bias_input_idx
                node_map["add_bias"] = add_bias_idx
                real_div_input_node = add_bias_node.input[0]

        real_div_idx, real_div_node = FindNodeInfo(graph_def, real_div_input_node)
        if real_div_node.op != "RealDiv":
            continue
        node_map["real_div"] = real_div_idx

        _, real_div_y_node = FindNodeInfo(graph_def, real_div_node.input[1])
        if real_div_y_node.op != "Const":
            continue
        scale = real_div_y_node.attr["value"].tensor.float_val[0]

        qk_matmul_idx, qk_matmul_node = FindNodeInfo(graph_def, real_div_node.input[0])
        if qk_matmul_node.op != "BatchMatMul" and qk_matmul_node.op != "BatchMatMulV2":
            continue
        node_map["qk_matmul"] = qk_matmul_idx

        querys_pack_idx, querys_pack_node = FindNodeInfo(
            graph_def, qk_matmul_node.input[0]
        )
        keys_pack_idx, keys_pack_node = FindNodeInfo(graph_def, qk_matmul_node.input[1])
        if querys_pack_node.op != "Pack" or keys_pack_node.op != "Pack":
            continue
        node_map["querys_pack"] = querys_pack_idx
        node_map["keys_pack"] = keys_pack_idx

        querys_split_idx, querys_split_node = FindNodeInfo(
            graph_def, querys_pack_node.input[0]
        )
        keys_split_idx, keys_split_node = FindNodeInfo(
            graph_def, keys_pack_node.input[0]
        )
        values_split_idx, values_split_node = FindNodeInfo(
            graph_def, values_pack_node.input[0]
        )
        if (
            querys_split_node.op != "Split"
            or keys_split_node.op != "Split"
            or values_split_node.op != "Split"
        ):
            continue
        node_map["querys_split"] = querys_split_idx
        node_map["keys_split"] = keys_split_idx
        node_map["values_split"] = values_split_idx

        # if len(querys_split_node.input) <= 0 or len(keys_split_node.input) <= 0 or len(values_split_node.input) <= 0:
        #   continue
        _, querys_split_const_in_node = FindNodeInfo(
            graph_def, querys_split_node.input[0]
        )
        _, keys_split_const_in_node = FindNodeInfo(graph_def, keys_split_node.input[0])
        _, values_split_const_in_node = FindNodeInfo(
            graph_def, values_split_node.input[0]
        )
        if (
            querys_split_const_in_node.op != "Const"
            or keys_split_const_in_node.op != "Const"
            or values_split_const_in_node.op != "Const"
        ):
            continue
        querys_split_axis = querys_split_const_in_node.attr["value"].tensor.int_val[0]
        keys_split_axis = keys_split_const_in_node.attr["value"].tensor.int_val[0]
        values_split_axis = values_split_const_in_node.attr["value"].tensor.int_val[0]
        if querys_split_axis != 2 or keys_split_axis != 2 or values_split_axis != 2:
            continue

        querys_in_idx, _ = FindNodeInfo(graph_def, querys_split_node.input[1])
        keys_in_idx, _ = FindNodeInfo(graph_def, keys_split_node.input[1])
        values_in_idx, _ = FindNodeInfo(graph_def, values_split_node.input[1])
        node_map["querys_in"] = querys_in_idx
        node_map["keys_in"] = keys_in_idx
        node_map["values_in"] = values_in_idx

        # valid pattern parameters
        querys_split_num = querys_split_node.attr["num_split"].i
        keys_split_num = keys_split_node.attr["num_split"].i
        values_split_num = values_split_node.attr["num_split"].i
        if querys_split_num != keys_split_num or querys_split_num != values_split_num:
            return None, None, None, None
        nh = querys_split_num
        querys_pack_num = querys_pack_node.attr["N"].i
        keys_pack_num = keys_pack_node.attr["N"].i
        values_pack_num = values_pack_node.attr["N"].i
        if (
            querys_pack_num != nh
            or querys_pack_num != keys_pack_num
            or querys_pack_num != values_pack_num
        ):
            return None, None, None, None
        if feature_length % nh != 0:
            continue
        emb_size = int(feature_length / nh)
        break
    if (
        node_map["querys_in"] is None
        or node_map["keys_in"] is None
        or node_map["values_in"] is None
    ):
        return None, None, None, None
    return node_map, nh, emb_size, scale


def SpecialRuleReplaceMultiheadAttention(graph_def):
    while True:
        node_map, nh, emb_size, scale = FindPatternMultiheadAttention(graph_def)
        if node_map == None or nh == None or emb_size == None or scale == None:
            break
        print(node_map, nh, emb_size, scale)
        # construct FusedMultiheadAttention node.
        new_node = npb.NodeDef()
        new_node.name = (
            graph_def.node[node_map["reshape"]].name + "_FusedMultiheadAttention"
        )
        if "add_bias" in node_map and "expand_bias" in node_map:
            new_node.op = "FusedMultiheadAttentionWithBias"
        else:
            new_node.op = "FusedMultiheadAttention"
        new_node.attr["T"].type = tpb.DataType.DT_FLOAT
        new_node.attr["emb_size"].i = emb_size
        new_node.attr["nh"].i = nh
        new_node.attr["scale"].f = scale
        new_node.input.append(graph_def.node[node_map["querys_in"]].name)
        new_node.input.append(graph_def.node[node_map["keys_in"]].name)
        new_node.input.append(graph_def.node[node_map["values_in"]].name)
        if "add_bias" in node_map and "expand_bias" in node_map:
            new_node.input.append(graph_def.node[node_map["expand_bias"]].name)
        # use new node replace pattern
        for node in graph_def.node:
            # 修改所有 node 的 input 为 new_node 的输出
            for idx, input_name in enumerate(node.input):
                if input_name == graph_def.node[node_map["reshape"]].name:
                    node.input[idx] = new_node.name
        # 从后往前删除被替换的节点
        reserve_node = ["querys_in", "keys_in", "values_in", "expand_bias"]
        RemoveSpecificNode(graph_def, node_map, reserve_node)
        print("Add ", new_node.op, " OP ", new_node.name)
        graph_def.node.append(new_node)
    return graph_def


def FindSplitNodes(graph_def, mul_list):
    split_node_names = set()
    for mul_index in mul_list:
        mul_node = graph_def.node[mul_index]
        if not len(mul_node.input) == 2:
            return set()
        for input_name in mul_node.input:
            if ":" in input_name:
                split_node_names.add(":".join(input_name.split(":")[:-1]))
            else:
                split_node_names.add(input_name)
        if not len(split_node_names) == 2:
            return set()
    return split_node_names


def FindPatternSlotGate(graph_def):
    # split_v(inputs)       split(weights)
    #      |                   |
    #       \                 /
    #        #slot_num 个 mul
    #               |
    #             concat
    for idx, node in enumerate(graph_def.node):
        node_map = {
            "split_v_input": None,
            "split_weight": None,
            "mul": None,
            "concatv2": None,
        }

        slot_num = None
        slot_dims = []

        if node.op != "ConcatV2":
            continue
        node_map["concatv2"] = idx
        slot_num = len(node.input) - 1
        node_map["mul"] = []
        for mul_name in node.input:
            mul_idx, mul_node = FindNodeInfo(graph_def, mul_name)
            if mul_node.op != "Mul":
                break
            else:
                node_map["mul"].append(mul_idx)
        if not len(node_map["mul"]) == slot_num:
            continue

        # find split_v and split
        # 所有 mul 的两个输入要一样
        split_node_names = FindSplitNodes(graph_def, node_map["mul"])
        if not len(split_node_names) == 2:
            continue
        for node_name in split_node_names:
            split_idx, split_node = FindNodeInfo(graph_def, node_name)
            if (node_map["split_weight"] is None) and (split_node.op == "Split"):
                # get num_split, split_dim
                num_split = split_node.attr["num_split"].i
                if not num_split == slot_num:
                    continue
                _, axis_node = FindNodeInfo(graph_def, split_node.input[0])
                axis = tensor_util.MakeNdarray(axis_node.attr["value"].tensor).flatten()
                if (axis.size != 1) or (axis[0] != 1):
                    continue
                node_map["split_weight"] = split_idx

            if (node_map["split_v_input"] is None) and (split_node.op == "SplitV"):
                # num_split, split_dim
                num_split = split_node.attr["num_split"].i
                if not num_split == slot_num:
                    continue
                _, axis_node = FindNodeInfo(graph_def, split_node.input[2])
                axis = tensor_util.MakeNdarray(axis_node.attr["value"].tensor).flatten()
                if (axis.size != 1) or (axis[0] != 1):
                    continue

                # slot_dims
                _, split_dims_node = FindNodeInfo(graph_def, split_node.input[1])
                slot_dims = tensor_util.MakeNdarray(
                    split_dims_node.attr["value"].tensor
                ).flatten()
                if not len(slot_dims) == slot_num:
                    continue
                node_map["split_v_input"] = split_idx
        return node_map, slot_dims
    return None, None


def SpecialRuleReplaceSlotGate(graph_def, output_nodes):
    while True:
        node_map, slot_dims = FindPatternSlotGate(graph_def)
        if node_map == None:
            break

        print(node_map)
        print(slot_dims)
        slot_start = []
        start_counter = 0
        for dim in slot_dims:
            slot_start.append(start_counter)
            start_counter += dim

        # new slot_start, fused_slot_gate node
        slot_start_node = npb.NodeDef()
        slot_start_node.name = (
            graph_def.node[node_map["concatv2"]].name + "_FusedSlotGate_SlotStart"
        )
        slot_start_node.op = "Const"
        slot_start_node.attr["dtype"].type = tpb.DataType.DT_INT32
        start_tensor = tensorpb.TensorProto()
        start_tensor.dtype = tpb.DataType.DT_INT32
        start_tensor.int_val.extend(slot_start)
        start_tensor.tensor_shape.dim.add().size = len(slot_start)
        slot_start_node.attr["value"].tensor.CopyFrom(start_tensor)

        fused_node = npb.NodeDef()
        fused_node.name = graph_def.node[node_map["concatv2"]].name + "_FusedSlotGate"
        fused_node.op = "FusedSlotGate"
        fused_node.attr["T"].type = tpb.DataType.DT_FLOAT
        fused_node.attr["slot_num"].i = len(slot_dims)
        fused_node.attr["dim"].i = sum(slot_dims)
        fused_node.input.append(graph_def.node[node_map["split_v_input"]].input[0])
        fused_node.input.append(graph_def.node[node_map["split_weight"]].input[1])
        fused_node.input.append(graph_def.node[node_map["split_v_input"]].input[1])
        fused_node.input.append(slot_start_node.name)

        # 修改所有 input 是 concatv2 的 node 的 input 为 fused_node
        for node in graph_def.node:
            for idx, input_name in enumerate(node.input):
                if input_name == graph_def.node[node_map["concatv2"]].name:
                    node.input[idx] = fused_node.name

        graph_def.node.append(slot_start_node)
        graph_def.node.append(fused_node)

        # 删除不需要的 node
        transforms = ["strip_unused_nodes", "sort_by_execution_order"]
        graph_def = TransformGraph(graph_def, [], output_nodes, transforms)
    return graph_def


def FindPatternLayerNorm(graph_def):
    # 寻找 layer norm 的 pattern，不包含 beta 和 gama 参数
    node_map = {
        "input": None,
        "input_mean": None,
        "square_diff": None,
        "square_diff_mean": None,
        "add_eps": None,
        "rsqrt": None,
        "mul_0": None,
        "mul_1": None,
        "sub": None,
    }
    for idx, node in enumerate(graph_def.node):
        if node.op != "Sub":
            continue

        # mul0 and mul1
        mul0_id, mul0_node = FindNodeInfo(graph_def, node.input[0])
        if mul0_node.op != "Mul":
            continue

        mul1_id, mul1_node = FindNodeInfo(graph_def, node.input[1])
        if mul1_node.op != "Mul":
            continue

        input_from_mul0_id, input_from_mul0_node = FindNodeInfo(
            graph_def, mul0_node.input[0]
        )

        rsqrt_id, rsqrt_node = FindNodeInfo(graph_def, mul0_node.input[1])

        add_eps_id, add_eps_node = FindNodeInfo(graph_def, rsqrt_node.input[0])
        if add_eps_node.op != "AddV2":
            continue

        # variance mean
        square_diff_mean_id, square_diff_mean_node = FindNodeInfo(
            graph_def, add_eps_node.input[0]
        )
        if square_diff_mean_node.op != "Mean":
            continue
        if not square_diff_mean_node.attr["keep_dims"]:
            continue

        _, eps_node = FindNodeInfo(graph_def, add_eps_node.input[1])
        if eps_node.op != "Const":
            continue

        square_diff_id, square_diff_node = FindNodeInfo(
            graph_def, square_diff_mean_node.input[0]
        )
        if square_diff_node.op != "SquaredDifference":
            continue

        _, reduce_indice_node = FindNodeInfo(graph_def, square_diff_mean_node.input[1])
        if reduce_indice_node.op != "Const":
            continue

        input_from_square_diff_id, input_from_square_diff_node = FindNodeInfo(
            graph_def, square_diff_node.input[0]
        )
        input_mean_from_square_diff_id, input_mean_from_square_diff_node = FindNodeInfo(
            graph_def, square_diff_node.input[1]
        )

        input_mean_from_mul1_id, input_mean_from_mul1_node = FindNodeInfo(
            graph_def, mul1_node.input[0]
        )
        if input_mean_from_mul1_node.op != "Mean":
            continue
        if not input_mean_from_mul1_node.attr["keep_dims"]:
            continue

        rsqrt_from_mul1_id, rsqrt_from_mul1_node = FindNodeInfo(
            graph_def, mul1_node.input[1]
        )
        if rsqrt_id != rsqrt_from_mul1_id or rsqrt_node.op != "Rsqrt":
            continue

        input_from_input_mean_id, input_from_input_mean_node = FindNodeInfo(
            graph_def, input_mean_from_mul1_node.input[0]
        )
        if (
            input_from_square_diff_id != input_from_mul0_id
            or input_from_square_diff_id != input_from_input_mean_id
        ):
            continue

        _, reduce_indice_from_input_mean_node = FindNodeInfo(
            graph_def, input_mean_from_mul1_node.input[1]
        )
        if reduce_indice_from_input_mean_node.op != "Const":
            continue

        # eps value
        eps_value = tf.make_ndarray(eps_node.attr["value"].tensor)
        assert eps_value.size == 1, "ndarrry of eps' length should be 1."
        eps_value = eps_value.tolist()

        # reduce axes
        axes_content = []
        reduce_axes0 = tf.make_ndarray(
            reduce_indice_from_input_mean_node.attr["value"].tensor
        )
        reduce_axes1 = tf.make_ndarray(reduce_indice_node.attr["value"].tensor)
        if reduce_axes0.ndim != reduce_axes1.ndim:
            print("[WARNING] reduce axes dim not equal.")
            continue
        if reduce_axes0.ndim > 0:
            allAxisEqual = True
            for axis0, axis1 in zip(reduce_axes0, reduce_axes1):
                if axis0 != axis1:
                    allAxisEqual = False
                    break
                axes_content.append(axis0)
            if not allAxisEqual:
                print("[WARNING] reduce axis not equal")
                continue
        else:
            if reduce_axes0 != reduce_axes1:
                print("[WARNING] reduce axis not equal")
                continue
            axes_content.append(reduce_axes0.tolist())

        # assign all node
        node_map["sub"] = idx
        node_map["mul_0"] = mul0_id
        node_map["mul_1"] = mul1_id
        node_map["add_eps"] = add_eps_id
        node_map["square_diff_mean"] = square_diff_mean_id
        node_map["square_diff"] = square_diff_id
        node_map["input_mean"] = input_mean_from_mul1_id
        node_map["rsqrt"] = rsqrt_id
        node_map["input"] = input_from_input_mean_id
        return node_map, eps_value, axes_content
    return None, None, None


def SpecialRuleReplaceLayerNorm(graph_def):
    while True:
        node_map, eps, axes_content = FindPatternLayerNorm(graph_def)
        if not node_map:
            break

        axes_content = sorted(axes_content)
        axes_tenor = tensorpb.TensorProto()
        axes_tenor.dtype = tpb.DataType.DT_INT32
        axes_tenor.int_val.extend(axes_content)
        axes_tenor.tensor_shape.dim.add().size = len(axes_content)

        new_node = npb.NodeDef()
        new_node.name = graph_def.node[node_map["sub"]].name + "_FusedLayerNorm"
        new_node.op = "FusedLayerNorm"
        new_node.attr["T"].type = tpb.DataType.DT_FLOAT
        new_node.attr["eps"].f = eps
        new_node.attr["axes"].tensor.CopyFrom(axes_tenor)
        new_node.input.append(graph_def.node[node_map["input"]].name)

        for node in graph_def.node:
            for idx, input_name in enumerate(node.input):
                if input_name == graph_def.node[node_map["sub"]].name:
                    node.input[idx] = new_node.name
        RemoveSpecificNode(graph_def, node_map, ["input"])
        print("Add ", new_node.op, " OP, name: ", new_node.name)
        graph_def.node.append(new_node)
    return graph_def


def FindPatternBatchOnes(graph_def):
    node_map = {"Shape": None, "StridedSlice": None, "Pack": None, "Fill": None}
    for idx, node in enumerate(graph_def.node):
        if node.op == "Fill":
            # get fill value
            _, value_node = FindNodeInfo(graph_def, node.input[1])
            if not value_node.op == "Const":
                continue
            fill_param = tf.make_ndarray(value_node.attr["value"].tensor).flatten()
            if not len(fill_param) == 1:
                continue
            if not fill_param[0] == 1:
                continue
            dim_idx, input_dims = FindNodeInfo(graph_def, node.input[0])
            # input_value = FindNodeInfo(node.input[1])
            if input_dims.op == "Pack":
                # pack_dim = 1
                _, pack_dim_node = FindNodeInfo(graph_def, node.input[1])
                if not pack_dim_node.op == "Const":
                    continue
                pack_param = tf.make_ndarray(
                    pack_dim_node.attr["value"].tensor
                ).flatten()
                if not len(pack_param) == 1:
                    continue
                if not pack_param[0] == 1:
                    continue
                node_map["Fill"] = idx
                node_map["Pack"] = dim_idx
                break
    if node_map["Fill"] is None:
        return None

    pack_node = graph_def.node[node_map["Pack"]]
    slice_idx, slice_node = FindNodeInfo(graph_def, pack_node.input[0])
    if not slice_node.op == "StridedSlice":
        return None
    else:
        # begin_mask = 0
        if not slice_node.attr["begin_mask"].i == 0:
            return None
        # ellipsis_mask = 0
        if not slice_node.attr["ellipsis_mask"].i == 0:
            return None
        # end_mask = 0
        if not slice_node.attr["end_mask"].i == 0:
            return None
        # new_axis_mask = 0
        if not slice_node.attr["new_axis_mask"].i == 0:
            return None
        # shrink_axis_mask = 1
        if not slice_node.attr["shrink_axis_mask"].i == 1:
            return None
        # begin = 0
        input_param = []
        _, begin_node = FindNodeInfo(graph_def, slice_node.input[1])
        if not begin_node.op == "Const":
            return None
        input_param.extend(tf.make_ndarray(begin_node.attr["value"].tensor).flatten())
        # end = 1
        _, end_node = FindNodeInfo(graph_def, slice_node.input[2])
        if not end_node.op == "Const":
            return None
        input_param.extend(tf.make_ndarray(end_node.attr["value"].tensor).flatten())
        # stride = 1
        _, stride_node = FindNodeInfo(graph_def, slice_node.input[3])
        if not stride_node.op == "Const":
            return None
        input_param.extend(tf.make_ndarray(stride_node.attr["value"].tensor).flatten())
        if not input_param == [0, 1, 1]:
            return None
        node_map["StridedSlice"] = slice_idx

    slice_node = graph_def.node[node_map["StridedSlice"]]
    shape_idx, shape_node = FindNodeInfo(graph_def, slice_node.input[0])
    if not shape_node.op == "Shape":
        return None
    else:
        node_map["Shape"] = shape_idx
    return node_map


def SpecialRuleRemoveBatchOnes(graph_def, output_nodes):
    new_node = npb.NodeDef()
    # 因为有时候图里就有 op 叫 ones, 得先写个名字替代一下
    # 优化过程中很可能删掉了原有的 ones
    unique_name = "ones_to_be_replaced"
    new_node.name = unique_name
    new_node.op = "Placeholder"
    new_node.attr["dtype"].type = tpb.DataType.DT_FLOAT
    proto = tf.TensorShape([None, 1])
    new_node.attr["shape"].shape.CopyFrom(proto.as_proto())
    graph_def.node.append(new_node)
    transforms = ["strip_unused_nodes"]

    while True:
        node_map = FindPatternBatchOnes(graph_def)
        print(node_map)
        if node_map is None:
            break
        # replace nodes whose input is Fill
        fill_node_name = graph_def.node[node_map["Fill"]].name
        for idx, node in enumerate(graph_def.node):
            for input_idx, i in enumerate(node.input):
                if i == fill_node_name:
                    # print("change input of node: ", node)
                    graph_def.node[idx].input[input_idx] = new_node.name
        # strip unused nodes
        graph_def = TransformGraph(graph_def, [], output_nodes, transforms)

    # 将 unique_name 换成 ones, 并 check 图里没有 ones
    for node in graph_def.node:
        assert not (node.name == "ones"), "图里有 ones node, 暂时不能做图变换"
    for idx, node in enumerate(graph_def.node):
        if node.name == unique_name:
            graph_def.node[idx].name = "ones"
        for input_idx, input in enumerate(node.input):
            if input == unique_name:
                graph_def.node[idx].input[input_idx] = "ones"

    return graph_def


def GetStridedSliceBeginAndEnd(graph_def, strided_slice_node_idx):
    _, begin_node = FindNodeInfo(
        graph_def, graph_def.node[strided_slice_node_idx].input[1]
    )
    if begin_node.op == "Const":
        begin = begin_node.attr["value"].tensor.int_val[0]
    else:
        begin = -2
    _, end_node = FindNodeInfo(
        graph_def, graph_def.node[strided_slice_node_idx].input[2]
    )
    if end_node.op == "Const":
        end = end_node.attr["value"].tensor.int_val[0]
    else:
        end = -2
    return begin, end


def FindDynamicReshape(graph_def):
    # find shape -> strided_slice(0,1) -> pack -> reshape
    node_map = {"Shape": None, "StridedSlice": None, "Pack": None, "Reshape": None}
    for idx, node in enumerate(graph_def.node):
        if node.op == "Reshape":
            shape_idx, shape_node = FindNodeInfo(graph_def, node.input[1])
            if shape_node.op == "Pack":
                node_map["Reshape"] = idx
                node_map["Pack"] = shape_idx
                batch_idx, batch_node = FindNodeInfo(graph_def, shape_node.input[0])
                if batch_node.op == "StridedSlice":
                    begin, end = GetStridedSliceBeginAndEnd(graph_def, batch_idx)
                    if (begin == 0) and (end == 1):
                        node_map["StridedSlice"] = batch_idx
                        shape_idx, shape_node = FindNodeInfo(
                            graph_def, batch_node.input[0]
                        )
                        if shape_node.op == "Shape":
                            node_map["Shape"] = shape_idx
                            break

    if node_map["Shape"] is None:
        return None
    return node_map


def SpecialRuleRemoveReshape(graph_def):
    # find strided_slice -> pack -> reshape
    # add constant node
    while True:
        node_map = FindDynamicReshape(graph_def)
        print(node_map)
        if node_map is None:
            break

        # create new node constant
        new_node = npb.NodeDef()
        new_node.name = graph_def.node[node_map["Reshape"]].name + "_const_shape"
        new_node.op = "Const"
        new_node.attr["dtype"].type = tpb.DataType.DT_INT32
        new_tensor = tensorpb.TensorProto()
        new_tensor.dtype = tpb.DataType.DT_INT32
        new_tensor.int_val.append(-1)
        for idx in range(1, len(graph_def.node[node_map["Pack"]].input)):
            _, const_node = FindNodeInfo(
                graph_def, graph_def.node[node_map["Pack"]].input[idx]
            )
            assert const_node.op == "Const"
            assert const_node.attr["value"].tensor.dtype == tpb.DataType.DT_INT32
            assert len(const_node.attr["value"].tensor.int_val) == 1
            new_tensor.int_val.append(const_node.attr["value"].tensor.int_val[0])

        new_tensor.tensor_shape.dim.add().size = (
            graph_def.node[node_map["Pack"]].attr["N"].i
        )
        new_node.attr["value"].tensor.CopyFrom(new_tensor)

        # change original reshape node to const reshape
        graph_def.node[node_map["Reshape"]].input[1] = new_node.name

        # add new node
        graph_def.node.append(new_node)

    return graph_def


def FindTransformerBatchMatmul(graph_def):
    # when X*Y in transformer component, X is b*seq*action_size
    # Y is action_size*(nh*attr_emb) constant
    # but tensorflow donnot support this kind of matmul,
    # we need to squash X's first 2 dims first.
    # reshape(X, [-1, action_size])->Matmul(X, Y)->reshape(Z, [-1, seq, (nh*attr_emb)])
    # TensorRT support this kind of matmul, just remove reshapes
    node_map = {"Reshape_in": None, "MatMul": None, "Reshape_out": None}
    for idx, node in enumerate(graph_def.node):
        if node.op == "Reshape":
            # check is const shape and first is 1
            _, shape_node = FindNodeInfo(graph_def, node.input[1])
            if shape_node.op != "Const":
                continue
            shape = tf.make_ndarray(shape_node.attr["value"].tensor)
            if shape[0] != -1:
                continue
            node_map["Reshape_out"] = idx
            matmul_idx, matmul_node = FindNodeInfo(graph_def, node.input[0])
            if matmul_node.op == "MatMul":
                node_map["MatMul"] = matmul_idx
                reshape_idx, reshape_node = FindNodeInfo(
                    graph_def, matmul_node.input[0]
                )
                if reshape_node.op == "Reshape":
                    _, shape_node = FindNodeInfo(graph_def, reshape_node.input[1])
                    if shape_node.op != "Const":
                        continue
                    shape = tf.make_ndarray(shape_node.attr["value"].tensor)
                    if shape[0] != -1:
                        continue
                    node_map["Reshape_in"] = reshape_idx
                    break

    if node_map["Reshape_in"] is None:
        return None
    return node_map


def SpecialRuleTransformerBatchMatMulRemoveReshape(graph_def, output_nodes):
    # when X*Y in transformer component, X is b*seq*action_size
    # Y is action_size*(nh*attr_emb) constant
    # but tensorflow donnot support this kind of matmul,
    # we need to squash X's first 2 dims first.
    # reshape(X, [-1, action_size])->Matmul(X, Y)->reshape(Z, [-1, seq, (nh*attr_emb)])
    # TensorRT support this kind of matmul, just remove reshapes
    transforms = ["strip_unused_nodes", "sort_by_execution_order"]
    while True:
        node_map = FindTransformerBatchMatmul(graph_def)
        print(node_map)
        if node_map is None:
            break
        # create new node MatMul
        new_node = npb.NodeDef()
        new_node.CopyFrom(graph_def.node[node_map["MatMul"]])
        new_node.input[0] = graph_def.node[node_map["Reshape_in"]].input[0]
        # change all nodes's input from reshape_out to new Matmul
        old_export_tensor = graph_def.node[node_map["Reshape_out"]].name
        new_export_tensor = new_node.name
        for node_idx, node in enumerate(graph_def.node):
            if old_export_tensor in node.input:
                input_idx = list(node.input).index(old_export_tensor)
                print(
                    "replace node %s's input %s to %s"
                    % (node.name, str(input_idx), new_export_tensor)
                )
                graph_def.node[node_idx].input[input_idx] = new_export_tensor
        # delete original matmul node
        del graph_def.node[node_map["MatMul"]]
        # add new MatMul node
        graph_def.node.append(new_node)
        # remove unused nodes
        graph_def = TransformGraph(graph_def, [], output_nodes, transforms)
    return graph_def


def FindBatchMatmul(graph_def):
    node_map = {"ExpandDims": None, "Tile": None, "BatchMatMul": None}
    for idx, node in enumerate(graph_def.node):
        if node.op == "BatchMatMul" or node.op == "BatchMatMulV2":
            y_idx, y = FindNodeInfo(graph_def, node.input[1])
            if y.op == "Tile":
                node_map["BatchMatMul"] = idx
                node_map["Tile"] = y_idx
                break
    if node_map["BatchMatMul"] is None:
        return None

    tile_node = graph_def.node[node_map["Tile"]]
    expand_idx, expand_node = FindNodeInfo(graph_def, tile_node.input[0])
    if not expand_node.op == "ExpandDims":
        return None
    else:
        node_map["ExpandDims"] = expand_idx

    return node_map


def SpecialRuleTileMatmulToBatchMatMul(graph_def):
    # when X*Y, X is 3 dim, Y is 2 dim
    # we need to tile Y to 3 dim and do Matmul in TF
    # But we only need a BatchMatmul in GraphDef
    # expand_dims(y, 0)->tile(Pack([b, 1, 1]))->BatchMatMul
    # BatchMatMul(x(r3), y(r2))

    # This transformation will make graph not runable in tf
    while True:
        node_map = FindBatchMatmul(graph_def)
        print(node_map)
        if node_map is None:
            break

        # create new node BatchMatMul
        new_node = npb.NodeDef()
        new_node.CopyFrom(graph_def.node[node_map["BatchMatMul"]])
        new_node.input[1] = graph_def.node[node_map["ExpandDims"]].input[0]

        # delete original batchmatmul node
        del graph_def.node[node_map["BatchMatMul"]]
        # add new BatchMatMul node
        graph_def.node.append(new_node)

    return graph_def


def RemoveNodesOneByOne(graph, node_dict):
    for node_to_remove, subs_meta in node_dict.items():
        node_idx = -1
        input_tensor = None
        for i in range(len(graph.node)):
            if graph.node[i].name == node_to_remove:
                node_idx = i
                # print(graph.node[i])
                assert len(graph.node[i].input) == subs_meta[0]
                input_tensor = graph.node[i].input[subs_meta[2]]
                break
        node_input_sub = {node_to_remove, node_to_remove + ":0"}
        for i in range(len(graph.node)):
            for j in range(len(graph.node[i].input)):
                if graph.node[i].input[j] in node_input_sub:
                    # print(graph.node[i])
                    # print("substitute node %s's input %s from %s to %s" % (
                    #     graph.node[i].name, str(j), graph.node[i].input[j], input_tensor))
                    graph.node[i].input[j] = input_tensor
        del graph.node[node_idx]

    return graph


def RemoveIdentitySplitOp(graph):
    # node_name -> (input_size, output_size, substitute_input_index)
    nodes_to_be_removed = {}
    for node in graph.node:
        if node.op == "Split":
            # remove "num_split == 1" node
            if node.attr["num_split"].i == 1:
                nodes_to_be_removed[node.name] = (2, 1, 1)
    # remove all collected nodes and connect
    return RemoveNodesOneByOne(graph, nodes_to_be_removed)


def RemoveIndentity(graph):
    graph = RemoveIdentitySplitOp(graph)
    return graph


def FindExplicitConcatAndMul(graph_def):
    node_map = {"ExpandDims": None, "ConcatV2": None, "MulIdxIsConcat": None}
    for idx, node in enumerate(graph_def.node):
        if node.op == "ConcatV2":
            # check inputs all the same
            assert len(node.input) > 1
            all_same = True
            for input_name in node.input[:-1]:
                all_same = all_same & (input_name == node.input[0])
                if not all_same:
                    break
            if not all_same:
                continue
            # check axis is constant
            _, axis_node = FindNodeInfo(graph_def, node.input[-1])
            if not axis_node.op == "Const":
                continue
            concat_axis = axis_node.attr["value"].i
            # get expand node
            expand_idx, expand_node = FindNodeInfo(graph_def, node.input[0])

            if expand_node.op == "ExpandDims":
                _, axis_node = FindNodeInfo(graph_def, expand_node.input[1])
                if not axis_node.op == "Const":
                    continue
                if not concat_axis == axis_node.attr["value"].i:
                    continue
                node_map["ExpandDims"] = expand_idx
                node_map["ConcatV2"] = idx
                break
    if node_map["ConcatV2"] is None:
        return None

    concat_node = graph_def.node[node_map["ConcatV2"]]

    # find all nodes whose input is concat_node, check is mul
    idx_map = {}
    for idx, node in enumerate(graph_def.node):
        if concat_node.name in node.input:
            if not node.op == "Mul":
                return None
            idx_map[idx] = list(node.input).index(concat_node.name)
    if not idx_map:
        return None
    else:
        node_map["MulIdxIsConcat"] = idx_map
        return node_map


def SpecialRuleRemoveExplictConcat(graph_def):
    # find ExpandDims, ConcatV2, Mul
    # Note that ConcatV2 concatenates all same
    # because we can implicit broadcast in mul and pass concat
    # Remove ConcatV2 for not interrupting next Findxxx()
    while True:
        node_map = FindExplicitConcatAndMul(graph_def)
        print(node_map)
        if node_map is None:
            break

        # check ConcatV2 do not used by other nodes
        can_delete_subgraph = True
        cannot_export = {node_map["ConcatV2"]}
        ignored_nodes = set()
        for mul_idx, _ in node_map["MulIdxIsConcat"].items():
            ignored_nodes.add(mul_idx)
        for idx, node in enumerate(graph_def.node):
            if idx in ignored_nodes:
                continue
            else:
                for input in node.input:
                    if input in cannot_export:
                        can_delete_subgraph = False
        if not can_delete_subgraph:
            print("***cannot delete subgraph, continue****")
            continue

        # replace mul's input to expandDims
        for mul_idx, input_idx in node_map["MulIdxIsConcat"].items():
            print(
                "replace node %s's input %s to %s"
                % (
                    graph_def.node[mul_idx].name,
                    str(input_idx),
                    graph_def.node[node_map["ExpandDims"]].name,
                )
            )
            graph_def.node[mul_idx].input[input_idx] = graph_def.node[
                node_map["ExpandDims"]
            ].name

        # delete concatv2
        del graph_def.node[node_map["ConcatV2"]]

    return graph_def


def FindGatherMatmul(graph_def):
    gather_index = 1
    node_map = {"GatherV2": None, "BatchMatMulV2": None}
    for idx, node in enumerate(graph_def.node):
        if node.op == "BatchMatMulV2":
            found_fusion = False
            for gather_input_idx in range(2):
                gather_idx, gather_node = FindNodeInfo(
                    graph_def, node.input[gather_input_idx]
                )
                if gather_node.op != "GatherV2":
                    continue
                if not IsCompressIndexIndices(gather_node.input[1], graph_def):
                    continue
                print(
                    f"FindGatherMatmul node:{node.name}, gather=node.input[{gather_input_idx}]={node.input[gather_input_idx]}, gather indices: {gather_node.input[1]}"
                )
                node_map["BatchMatMulV2"] = idx
                node_map["GatherV2"] = gather_idx
                gather_index = gather_input_idx
                found_fusion = True
                break
            if found_fusion:
                break
    if node_map["BatchMatMulV2"] is None or node_map["GatherV2"] is None:
        return None, None
    return node_map, gather_index


def SpecialRuleFusedGatherMatmul(graph_def):
    """
    matmul(gather(params, indices), X) 或者
    matmul(X, gather(params, indices))
    --> FusedGatherMatmul(X, params, indices, is_gather_left, ...)
    """
    while True:
        node_map, gather_index = FindGatherMatmul(graph_def)
        if node_map == None:
            break
        new_node = npb.NodeDef()
        new_node.name = (
            graph_def.node[node_map["BatchMatMulV2"]].name + "_FusedGatherMatmul"
        )
        new_node.op = "FusedGatherMatmul"
        assert graph_def.node[node_map["GatherV2"]].attr["batch_dims"].i == 0
        new_node.attr["T"].type = (
            graph_def.node[node_map["BatchMatMulV2"]].attr["T"].type
        )
        new_node.attr["adj_x"].b = (
            graph_def.node[node_map["BatchMatMulV2"]].attr["adj_x"].b
        )
        new_node.attr["adj_y"].b = (
            graph_def.node[node_map["BatchMatMulV2"]].attr["adj_y"].b
        )
        new_node.attr["is_gather_left"].b = gather_index == 0
        if gather_index == 0:
            new_node.input.append(
                graph_def.node[node_map["BatchMatMulV2"]].input[1]
            )  # X
        else:
            new_node.input.append(
                graph_def.node[node_map["BatchMatMulV2"]].input[0]
            )  # X
        new_node.input.append(graph_def.node[node_map["GatherV2"]].input[0])  # params
        new_node.input.append(graph_def.node[node_map["GatherV2"]].input[1])  # indices
        print(
            f"FusedGatherMatmul inputs: {new_node.input[0]}, {new_node.input[1]}, {new_node.input[2]}"
        )

        # assert axis == 0
        for node in graph_def.node:
            if node.name == graph_def.node[node_map["GatherV2"]].input[2]:
                assert node.op == "Const"
                assert node.attr["value"].i == 0

        # 修改所有 input 是候选集 BatchMatMulV2 node 的 input 为 new fused_node
        for node in graph_def.node:
            for idx, input_name in enumerate(node.input):
                if input_name == graph_def.node[node_map["BatchMatMulV2"]].name:
                    node.input[idx] = new_node.name

        del graph_def.node[node_map["BatchMatMulV2"]]
        # del(graph_def.node[node_map["GatherV2"]])
        graph_def.node.append(new_node)

    return graph_def


def FindPatternCompressGather(graph_def):
    for idx, node in enumerate(graph_def.node):
        if node.op != "GatherV2":
            continue
        input_node_idx, input_node = FindNodeInfo(graph_def, node.input[0])
        if input_node.op != "GatherV2":
            continue
        if not IsCompressIndexIndices(input_node.input[1], graph_def):
            continue

        batch_dims = node.attr.get("batch_dims", 0)
        print(
            f"FindPatternCompressGather: {node.name} ({input_node.name} ({input_node.input[0]}, {input_node.input[1]}), {node.input[1]}, axis={node.input[2]}, batch_dims={batch_dims})"
        )
        pattern_nodes = dict()
        pattern_nodes["gather"] = idx
        pattern_nodes["gather_input"] = input_node_idx
        return pattern_nodes
    return None


def SpecialRuleFusedCompressGather(graph_def):
    """
    GatherV2(GahterV2(params, compress_index), indices, axis, batch_dims)
    -->
    FusedCompressGather(params, indices, axis, compress_index, batch_dims)
    """
    while True:
        pattern_nodes = FindPatternCompressGather(graph_def)
        if not pattern_nodes:
            break
        gather_node = graph_def.node[pattern_nodes["gather"]]
        input_node = graph_def.node[pattern_nodes["gather_input"]]

        new_node = npb.NodeDef()
        new_node.name = gather_node.name + "_FusedCompressGather"
        new_node.op = "FusedCompressGather"
        new_node.attr["Tparams"].type = gather_node.attr["Tparams"].type
        new_node.attr["Tindices"].type = gather_node.attr["Tindices"].type
        new_node.attr["Taxis"].type = gather_node.attr["Taxis"].type
        new_node.attr["batch_dims"].i = gather_node.attr["batch_dims"].i
        new_node.input.append(input_node.input[0])  # params
        new_node.input.append(gather_node.input[1])  # indices
        new_node.input.append(gather_node.input[2])  # axis
        new_node.input.append(input_node.input[1])  # compress_index

        # 修改 input 关系
        for node in graph_def.node:
            for idx, input_name in enumerate(node.input):
                if input_name == gather_node.name:
                    node.input[idx] = new_node.name

        del graph_def.node[pattern_nodes["gather"]]
        graph_def.node.append(new_node)

    return graph_def


# TODO(wuxikun) only_fast_compres_matmul 的场景判定是啥来着？记不得了 T_T
def FindPatternCompressMatmul(graph_def, only_fast_compres_matmul=False):
    gather_on_left = False
    node_map = {"GatherV2": None, "BatchMatMulV2": None}
    for idx, node in enumerate(graph_def.node):
        if (
            node.op == "BatchMatMulV2"
            or node.op == "BatchMatMul"
            or node.op == "MatMul"
        ):
            found_fusion = False
            for input_idx in range(2):
                input_node_idx, input_node = FindNodeInfo(
                    graph_def, node.input[input_idx]
                )
                if input_node.op != "GatherV2":
                    continue
                if not IsCompressIndexIndices(input_node.input[1], graph_def):
                    continue
                print(
                    f"FindPatternCompressMatmul node:{node.name}, gather=node.input[{input_idx}]={node.input[input_idx]}, gather indices: {input_node.input[1]}"
                )
                gather_on_left = input_idx == 0
                trans_a = graph_def.node[idx].attr["adj_x"].b

                # TODO(wuxikun) 为啥是 (!gather_on_left and !trans_a) 而不是 (axis == 0 && !trans_a and trans_b)
                if only_fast_compres_matmul and (gather_on_left or trans_a):
                    print(
                        f"skip FusedCompressMatmul, only_fast_compres_matmul:{only_fast_compres_matmul} gather_on_left:{gather_on_left} trans_a:{trans_a}"
                    )
                    continue
                node_map["BatchMatMulV2"] = idx
                node_map["GatherV2"] = input_node_idx
                found_fusion = True
                break
            if found_fusion:
                break
    if node_map["BatchMatMulV2"] is None or node_map["GatherV2"] is None:
        return None, None
    return node_map, gather_on_left


def SpecialRuleFusedCompressMatmul(graph_def, only_fast_compres_matmul=False):
    """
    matmul(gather(params, indices), X) 或者
    matmul(X, gather(params, indices))
    --> FusedCompressMatmul(X, params, indices, is_gather_left, ...)
    """
    while True:
        node_map, gather_on_left = FindPatternCompressMatmul(
            graph_def, only_fast_compres_matmul
        )
        if node_map == None:
            break
        _, gather_axis_input = FindNodeInfo(
            graph_def, graph_def.node[node_map["GatherV2"]].input[2]
        )
        assert graph_def.node[node_map["GatherV2"]].attr["batch_dims"].i == 0
        assert gather_axis_input.op == "Const"
        assert len(gather_axis_input.attr["value"].tensor.int_val) == 1

        gather_axis = gather_axis_input.attr["value"].tensor.int_val[0]
        assert gather_axis == 0 or gather_axis == 1

        new_node = npb.NodeDef()
        new_node.name = (
            graph_def.node[node_map["BatchMatMulV2"]].name + "_FusedCompressMatmul"
        )
        new_node.op = "FusedCompressMatmul"
        new_node.attr["T"].type = (
            graph_def.node[node_map["BatchMatMulV2"]].attr["T"].type
        )
        new_node.attr["adj_x"].b = (
            graph_def.node[node_map["BatchMatMulV2"]].attr["adj_x"].b
        )
        new_node.attr["adj_y"].b = (
            graph_def.node[node_map["BatchMatMulV2"]].attr["adj_y"].b
        )
        new_node.attr["is_gather_left"].b = gather_on_left
        new_node.attr["gather_axis"].i = gather_axis

        if gather_on_left:
            new_node.input.append(
                graph_def.node[node_map["BatchMatMulV2"]].input[1]
            )  # X
        else:
            new_node.input.append(
                graph_def.node[node_map["BatchMatMulV2"]].input[0]
            )  # X
        new_node.input.append(graph_def.node[node_map["GatherV2"]].input[0])  # params
        new_node.input.append(graph_def.node[node_map["GatherV2"]].input[1])  # indices
        print(
            f"FusedCompressGatherMatmul inputs: {new_node.input[0]}, {new_node.input[1]}, {new_node.input[2]}, is_gather_left:{gather_on_left}, gather_axis:{new_node.attr['gather_axis'].i}"
        )

        # 修改所有 input 是候选集 BatchMatMulV2 node 的 input 为 new fused_node
        for node in graph_def.node:
            for idx, input_name in enumerate(node.input):
                if input_name == graph_def.node[node_map["BatchMatMulV2"]].name:
                    node.input[idx] = new_node.name

        del graph_def.node[node_map["BatchMatMulV2"]]
        graph_def.node.append(new_node)

    return graph_def


def FindPatternGatherMulReduceSum(graph_def):
    node_map = dict()
    for idx, node in enumerate(graph_def.node):
        if node.op != "Sum":
            continue

        _, reduce_axis = FindNodeInfo(graph_def, node.input[1])
        if (
            reduce_axis.op != "Const"
            or reduce_axis.attr["value"].tensor.int_val[0] != -1
        ):
            continue

        mul_idx, mul_node = FindNodeInfo(graph_def, node.input[0])
        if mul_node.op != "Mul":
            continue

        gather_on_left = True
        yes, gather_node = IsCompressIndexGather(mul_node.input[0], graph_def)
        if not yes:
            gather_on_left = False
            yes, gather_node = IsCompressIndexGather(mul_node.input[1], graph_def)
            if not yes:
                gather_node = None
        if not gather_node:
            continue

        print(
            f"FindPatternGatherMulReduceSum Sum: {node.name} Mul: {mul_node.name} GatherV2: {gather_node.name} params: {gather_node.input[0]}"
        )
        node_map["SumNodeIdx"] = idx
        node_map["Sum"] = node
        node_map["Mul"] = mul_node
        node_map["GatherV2"] = gather_node
        node_map["gather_on_left"] = gather_on_left

        return node_map


def SpecialRuleFusedGatherMulReduceSum(graph_def):
    """
    reduce_sum(multiply(a, gather(b, indices, axis=i)), axis = -1)
    """
    while True:
        pattern = FindPatternGatherMulReduceSum(graph_def)
        if not pattern:
            break

        sum_node = pattern["Sum"]
        mul_node = pattern["Mul"]
        gather_node = pattern["GatherV2"]
        gather_on_left = pattern["gather_on_left"]

        _, gather_axis_input = FindNodeInfo(graph_def, gather_node.input[2])
        assert gather_node.attr["batch_dims"].i == 0
        assert gather_axis_input.op == "Const"
        assert len(gather_axis_input.attr["value"].tensor.int_val) == 1
        gather_axis = gather_axis_input.attr["value"].tensor.int_val[0]
        assert gather_axis == 0 or gather_axis == 1

        new_node = npb.NodeDef()
        new_node.name = sum_node.name + "_FusedGatherMulReduceSum"
        new_node.op = "FusedGatherMulReduceSum"
        new_node.attr["T"].type = sum_node.attr["T"].type
        new_node.attr["gather_axis"].i = gather_axis
        new_node.input.append(mul_node.input[1 if gather_on_left else 0])
        new_node.input.append(gather_node.input[0])
        new_node.input.append(gather_node.input[1])
        print(
            f"FusedGatherMulReduceSum matrix: {mul_node.input[1 if gather_on_left else 0]}, params:{gather_node.input[0]}, indices:{gather_node.input[1]}, gather_axis: {new_node.attr['gather_axis'].i}"
        )

        for node in graph_def.node:
            for idx, input_name in enumerate(node.input):
                if input_name == sum_node.name:
                    node.input[idx] = new_node.name

        del graph_def.node[pattern["SumNodeIdx"]]
        graph_def.node.append(new_node)

    return graph_def


def ConcatRename(graph_def):
    # 1. find all concat's axis node(must be Const Op)
    # 2. rename axis node
    # 3. rename all node's input
    const_rename = {}
    for node in graph_def.node:
        if node.op == "Concat" or node.op == "ConcatV2":
            if "axis" not in node.input[-1]:
                const_rename[node.input[-1]] = node.input[-1] + "/axis"

    for node in graph_def.node:
        if node.name in const_rename:
            print("Axis node: %s" % (node.name))
            assert node.op == "Const"
            node.name = const_rename[node.name]

    for node in graph_def.node:
        for idx in range(len(node.input)):
            if node.input[idx] in const_rename:
                node.input[idx] = const_rename[node.input[idx]]

    return graph_def


def RestoreMioVariable(graph_def, mio_variable, mio_address):
    new_graph = gpb.GraphDef()
    new_graph.CopyFrom(graph_def)
    del new_graph.node[:]

    # add all mio_address
    for _, node in mio_address.items():
        new_graph.node.append(node)

    # remove all temp placeholders
    for node in graph_def.node:
        if node.name in mio_variable:
            new_graph.node.append(mio_variable[node.name])
        else:
            new_graph.node.append(node)

    return new_graph


def GetInputsFromMioVariable(mio_v):
    if len(mio_v) > 0:
        assert isinstance(mio_v, dict)
    inputs = set()
    for k, v in mio_v.items():
        inputs.add(v.name)
    return list(inputs)


def GetInputsFromGraph(graph_def):
    inputs = set()
    for node in graph_def.node:
        if node.op == "Placeholder":
            inputs.add(node.name)
    return list(inputs)


def LoadGraphFromBase64(base64_graph):
    raw_base64 = base64_graph[9:]
    graph_string = base64.b64decode(raw_base64.encode("ascii"))
    graph_def = gpb.GraphDef()
    graph_def.ParseFromString(graph_string)
    return graph_def


def SaveGraphAsBase64(graph_def):
    base64_graph = base64.b64encode(graph_def.SerializeToString()).decode("ascii")
    return "base64://" + base64_graph


base_supported_op = set(
    [
        "VariableFromMioComponentTable",
        "Fill",
        "Placeholder",
        "Const",
        "Identity",
        "StopGradient",
        "Concat",
        "ConcatV2",
        "MatMul",
        "BatchMatMulV2",
        "BatchMatMul",
        "BiasAdd",
        "Exp",
        "Log",
        "Sqrt",
        "Recip",
        "Abs",
        "Neg",
        "Sin",
        "Cos",
        "Tan",
        "Sinh",
        "Cosh",
        "ASin",
        "ACos",
        "ATan",
        "ASinh",
        "ACosh",
        "ATanh",
        "Floor",
        "Ceil",
        "Add",
        "AddV2",
        "Mul",
        "Maximum",
        "Minimum",
        "Sub",
        "RealDiv",
        "Div",
        "Pow",
        "Relu",
        "Sigmoid",
        "Tanh",
        "LeakyRelu",
        "Elu",
        "Selu",
        "Softsign",
        "Clip",
        "Softmax",
        "Rsqrt",
        "Slice",
        "StridedSlice",
        "Reshape",
        "Transpose",
        "Mean",
        "Max",
        "Sum",
        "Prod",
        "Min",
        "Square",
        "SquaredDifference",
        "Gather",
        "GatherV2",
        "Split",
        "SplitV",
        "Pack",
        "ExpandDims",
        "FusedGatedSum",
        "FusedMultiheadAttention",
        "Softplus",
        "NonOp",
    ]
)

implicit_batch_extra_support_op = set([])
explicit_batch_extra_support_op = set(
    [
        "Tile",
        "Cast",
        "Shape",
        "TopKV2",
        "MioTopK",
        "LogicalNot",
        "Equal",
        "LogicalAnd",
        "LogicalOr",
        "LogicalXor",
        "Greater",
        "Less",
        "GreaterEqual",
        "LessEqual",
        "Squeeze",
        "Where",
        "SelectV2",
        "Select",
        "FusedSlotGate",
        "FusedMultiheadAttentionWithBias",
        "FusedLayerNorm",
        "Range",
        "FusedGatherMatmul",
        "FusedCompressGather",
        # "FusedCompressMatmul",
    ]
)


def CheckUnsupportedOp(graph_def, implicit_batch=True):
    supported_op = base_supported_op
    if implicit_batch:
        supported_op = set.union(supported_op, implicit_batch_extra_support_op)
    else:
        supported_op = set.union(supported_op, explicit_batch_extra_support_op)
    for node in graph_def.node:
        assert node.op in supported_op, (
            "Do not support op %s(%s), TensorRT 不支持此 Op，请根据指南文档中的 Op 列表调整图。本脚本可以通过识别一些预定义的图结构来替换掉其中的不兼容 Op，因此脚本的其它地方或许使用了这个 Op 却不报错。这些规则规定很严格，自己写的代码难以命中，所以请以支持列表为准。"
            % (node.op, node.name)
        )


def Optimize(
    bas64_graph,
    output_nodes,
    use_fused_op=False,
    implicit_batch=True,
    only_fast_compres_matmul=False,
):
    # Giving graph_def and output_nodes[], return new graph_def
    graph_def = LoadGraphFromBase64(bas64_graph)
    _, mio_v, mio_a = RemoveMioVariable(graph_def)
    inputs = GetInputsFromMioVariable(mio_v)
    if len(inputs) <= 0:
        inputs = GetInputsFromGraph(graph_def)
    if len(inputs) <= 0:
        print("[WARNING] cannot find input in graph")
    transforms = ["strip_unused_nodes", "remove_nodes(op=StopGradient)"]
    optimized_graph_def = TransformGraph(graph_def, inputs, output_nodes, transforms)
    # 先使用 fusion 的规则，因为后续的规则会打破 fusion 的规则
    if use_fused_op:
        pass
        # optimized_graph_def = SpecialRuleReplaceReduceSum(optimized_graph_def)
        # optimized_graph_def = SpecialRuleReplaceMultiheadAttention(optimized_graph_def)
    if use_fused_op and (not implicit_batch):
        pass
        # optimized_graph_def = SpecialRuleFusedCompressGather(optimized_graph_def)
        # optimized_graph_def = SpecialRuleFusedCompressMatmul(optimized_graph_def, only_fast_compres_matmul)
        # optimized_graph_def = SpecialRuleFusedGatherMatmul(optimized_graph_def)
        # optimized_graph_def = SpecialRuleReplaceSlotGate(optimized_graph_def, output_nodes)
        # optimized_graph_def = SpecialRuleReplaceLayerNorm(optimized_graph_def)

    optimized_graph_def = SpecialRuleRemoveBatchOnes(optimized_graph_def, output_nodes)
    transforms = ["merge_duplicate_nodes"]
    optimized_graph_def = TransformGraph(
        optimized_graph_def, inputs, output_nodes, transforms
    )
    transforms = ["strip_unused_nodes", "sort_by_execution_order"]
    optimized_graph_def = TransformGraph(
        optimized_graph_def, inputs, output_nodes, transforms
    )
    optimized_graph_def = SpecialRuleTileMatmulToBatchMatMul(optimized_graph_def)
    optimized_graph_def = TransformGraph(
        optimized_graph_def, inputs, output_nodes, transforms
    )
    optimized_graph_def = SpecialRuleTransformerBatchMatMulRemoveReshape(
        optimized_graph_def, output_nodes
    )
    optimized_graph_def = SpecialRuleRemoveReshape(optimized_graph_def)
    optimized_graph_def = TransformGraph(
        optimized_graph_def, inputs, output_nodes, transforms
    )
    optimized_graph_def = RemoveIndentity(optimized_graph_def)
    optimized_graph_def = SpecialRuleRemoveExplictConcat(optimized_graph_def)
    optimized_graph_def = TransformGraph(
        optimized_graph_def, inputs, output_nodes, transforms
    )
    optimized_graph_def = ConcatRename(optimized_graph_def)
    # ret = RestoreMioVariable(optimized_graph_def, mio_v, mio_a)
    # CheckUnsupportedOp(optimized_graph_def, implicit_batch=implicit_batch)
    print(
        "\n============== tensorrt_predict 使用注意: 虽然模型转换成功, 但是仍然有可能对某些 op 的配置不支持, 导致图挂掉. ==============\n"
    )

    # fout = open("./graph.opt.pb", 'wb')
    # fout.write(optimized_graph_def.SerializeToString())
    # fout.close()

    return SaveGraphAsBase64(optimized_graph_def)
