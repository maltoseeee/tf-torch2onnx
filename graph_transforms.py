#!/usr/bin/env python3
# coding=utf-8
"""
- file name: graph_transformer.py
- description: transform graph for tensorrt inference
- author: jiabowei@kuaishou.com
- date: 2021-11-25
"""

import tensorflow as tf
from tensorflow.core.framework import types_pb2 as tpb
from tensorflow.core.framework import node_def_pb2 as npb
from tensorflow.core.framework import graph_pb2 as gpb
from tensorflow.core.framework import tensor_pb2 as tensorpb
import re
import hashlib


def StripUnusedNodes(graph_def, inputs, outputs):
    """删除图中未使用的 Node"""
    used_node_idx = set()
    all_node_idx = set()
    idx_map = {}
    for idx, node in enumerate(graph_def.node):
        idx_map[node.name] = idx
        all_node_idx.add(idx)

    for output in outputs:
        node_set = set()
        node_queue = []
        for idx, node in enumerate(graph_def.node):
            if node.name != output:
                continue
            # 从输出往前逐层遍历
            node_queue.append(node.name)
            node_set.add(node.name)
            while len(node_queue) > 0:
                cur_node_name = node_queue.pop(0)
                cur_node_idx = idx_map[cur_node_name]
                cur_node = graph_def.node[cur_node_idx]
                used_node_idx.add(cur_node_idx)
                if len(cur_node.input) == 0:
                    continue
                for node_name in cur_node.input:
                    node_name = node_name.split(":")[0]
                    if node_name not in node_set:
                        node_set.add(node_name)
                        node_queue.append(node_name)

    unused_node_idx = all_node_idx - used_node_idx
    for idx in sorted(unused_node_idx, reverse=True):
        del graph_def.node[idx]
    return graph_def


def RemoveNodes(graph_def, inputs, outputs, transform):
    """从图中删除指定 op type 的 node"""
    support_ops = ["StopGradient"]
    pattern = "^remove_nodes\(op=([a-zA-Z]+)\)$"
    ops = re.findall(pattern, transform)
    if len(ops) != 1 or ops[0] not in support_ops:
        print("op to be removed not support: ", ops[0])
        return graph_def

    for op in ops:
        # 从图中找到所有要移除的节点
        node_idx = []
        for idx, node in enumerate(graph_def.node):
            if node.op != op:
                continue
            node_idx.append(idx)

        # 修改被移除节点的子节点为被移除节点的父节点
        for idx in node_idx:
            cur_node_name = graph_def.node[idx].name
            cur_node_input = graph_def.node[idx].input[0]
            for node_id, node in enumerate(graph_def.node):
                for input_id, input_name in enumerate(node.input):
                    input_name = input_name.split(":")[0]
                    if input_name == cur_node_name:
                        graph_def.node[node_id].input[input_id] = cur_node_input

        for idx in sorted(node_idx, reverse=True):
            del graph_def.node[idx]

    return graph_def


def NodeHash(node):
    md5 = hashlib.md5()
    md5.update(node.op.encode("utf-8"))
    md5.update(node.name.encode("utf-8"))
    for input_name in node.input:
        md5.update(input_name.encode("utf-8"))
    md5.update(node.device.encode("utf-8"))

    attrs = []
    for attr in node.attr:
        attrs.append(attr)
    sorted_attrs = sorted(attrs)
    for attr_name in sorted_attrs:
        md5.update(node.attr[attr_name].SerializeToString())
    return md5.hexdigest()


def MergeDuplicateNodes(graph_def, inputs, outputs):
    """合并内容重复的 node
    如果 Const node 有相同的 type 和 content；或者 node 有相同的 inputs 和 attributes 则合并他们
    """
    while True:
        any_duplicate_found = False
        node_hash_id_map = {}
        for idx, node in enumerate(graph_def.node):
            new_node = npb.NodeDef()
            new_node.CopyFrom(node)
            if new_node.name not in inputs and new_node.name not in outputs:
                new_node.name = ""
            node_hash = NodeHash(new_node)
            if node_hash not in node_hash_id_map:
                node_hash_id_map[node_hash] = []
            node_hash_id_map[node_hash].append(idx)

        node_ids_to_remove = []
        for k, v in node_hash_id_map.items():
            if len(v) <= 1:
                continue
            any_duplicate_found = True
            node_ids_to_remove.extend(v[1:])
            cur_node = graph_def.node[v[0]]
            for idx in v[1:]:
                for node in graph_def.node:
                    for input_idx, input_name in enumerate(node.input):
                        input_name_list = input_name.split(":")
                        if input_name_list[0] != graph_def.node[idx].name:
                            continue
                        if len(input_name_list) > 1:
                            assert len(input_name_list) == 2
                            node.input[input_idx] = (
                                cur_node.name + ":" + input_name_list[1]
                            )
                            print(node.input[input_idx])
                        else:
                            node.input[input_idx] = cur_node.name

        for idx in sorted(node_ids_to_remove, reverse=True):
            del graph_def.node[idx]

        if not any_duplicate_found:
            break
    return graph_def


def SortByExecutionOrder(graph_def, inputs, outputs):
    """对 graph 进行拓扑排序"""
    return graph_def


def TransformGraph(graph_def, inputs, outputs, transforms):
    for transform in transforms:
        if transform == "strip_unused_nodes":
            graph_def = StripUnusedNodes(graph_def, inputs, outputs)
        elif transform == "merge_duplicate_nodes":
            graph_def = MergeDuplicateNodes(graph_def, inputs, outputs)
        elif transform == "sort_by_execution_order":
            graph_def = SortByExecutionOrder(graph_def, inputs, outputs)
        elif transform.startswith("remove_nodes"):
            graph_def = RemoveNodes(graph_def, inputs, outputs, transform)
        else:
            assert False, "Dot not support transform: %s" % transform
    return graph_def
