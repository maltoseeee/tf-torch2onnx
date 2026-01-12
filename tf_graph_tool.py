#!/usr/bin/env python3
# coding=utf8
import argparse
import hashlib
import yaml
import json
import re
import sys
import base64
import tensorflow as tf
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import types_pb2 as tpb
from tensorflow.core.framework import node_def_pb2 as npb
from tensorflow.core.framework import tensor_pb2 as tensorpb
from tensorflow.core.framework import tensor_shape_pb2 as tensorshapepb
from google.protobuf import text_format

# import tf2onnx
# from tensorflow.tools.graph_transforms import TransformGraph
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

# sys.path.append('/home/root/code/dragon-kbuild/ks/serving_online/tensorrt_common')
from tensorrt_optimizer import Optimize
from tensorrt_optimizer import LoadGraphFromBase64, FindNodeInfo
from tensorrt_optimizer import RemoveMioVariable, RestoreMioVariable, TransformGraph


def LoadGraphPbAsBase64(pb_file):
    with open(pb_file, "rb") as f:
        encoded_graph = base64.b64encode(f.read())
    return "base64://" + encoded_graph.decode()


def LoadGraphDefAdBase64(graph_def):
    graph_str = graph_def.SerializeToString()
    return "base64://" + base64.b64encode(graph_str).decode()


def SaveOutputPb(graph_def, out_file, out_pbtxt):
    with open(out_file, "wb") as fp:
        fp.write(graph_def.SerializeToString())
    if out_pbtxt:
        with open(out_file + "txt", "w") as fp:
            text_proto = text_format.MessageToString(graph_def)
            fp.write(text_proto)


def SaveOnnxModel(onnx_model, save_path):
    with open(save_path, "wb") as fp:
        fp.write(onnx_model.SerializeToString())


def SaveOutputBase64Pb(b64_pb, out_file, out_pbtxt):
    b64_graph = b64_pb[9:]
    graph = base64.b64decode(b64_graph)
    with open(out_file, "wb") as fp:
        fp.write(graph)
    if out_pbtxt:
        with open(out_file + "txt", "w") as fp:
            graph_def = graph_pb2.GraphDef()
            graph_def.ParseFromString(graph)
            text_proto = text_format.MessageToString(graph_def)
            fp.write(text_proto)


def LoadConfigFromYaml(yaml_file):
    with open(yaml_file) as fp:
        params = []
        inputs = []
        targets = []
        yaml_content = yaml.load(fp, Loader=Loader)
        for value in yaml_content["param"]:
            params.append(value["name"])

        for value in yaml_content["embedding"]["slots_config"]:
            inputs.append(value["input_name"])

        for value in yaml_content["vec_input"]:
            inputs.append(value["name"])

        for _, value in yaml_content["graph_tensor_mapping"].items():
            target = value.split(":")[0]
            targets.append(target)
    return params, inputs, targets


def LoadConfigFromJson(json_file):
    with open(json_file) as fp:
        json_content = json.load(fp)
        params = []
        inputs = []
        targets = []
        for _, v in json_content["pipeline_manager_config"]["base_pipeline"][
            "processor"
        ].items():
            if (
                v["type_name"] != "MioPredictItemAttrEnricher"
                and v["type_name"] != "UniPredictFusedItemAttrEnricher"
                and v["type_name"] != "UniPredictItemAttrEnricher"
            ):
                continue
            for item in v["outputs"]:
                target = item["tensor_name"].replace(":0", "")
                targets.append(target)
            for item in v["param"]:
                params.append(item["name"])
            for item in v["inputs"]:
                inputs.append(item["tensor_name"])
            break
        return params, inputs, targets


def DoBacktrace(graph_def, targets):
    ret = {}

    if not isinstance(targets, list):
        targets = [targets]

    node_name_to_idx_map = {}
    for idx, node in enumerate(graph_def.node):
        node_name_to_idx_map[node.name] = idx

    for output in targets:
        for node_idx, node in enumerate(graph_def.node):
            node_name = node.name.split(":")[0]
            if node_name != output:
                continue
            node_idx_set = set()
            node_idx_set.add(node_idx)
            node_idx_queue = []
            node_idx_queue.append(node_idx)
            paths = []
            while len(node_idx_queue) > 0:
                cur_node_idx = node_idx_queue.pop(0)
                path = []
                for input_name in graph_def.node[cur_node_idx].input:
                    input_name = input_name.split(":")[0]
                    input_node_idx = node_name_to_idx_map[input_name]
                    path.append(input_name)
                    if input_node_idx not in node_idx_set:
                        node_idx_set.add(input_node_idx)
                        node_idx_queue.append(input_node_idx)
                if len(path) > 0:
                    paths.append(path)
            ret[output] = paths
            break
    return ret


def Backtrace(args):
    base64_graph = LoadGraphPbAsBase64(args.file)
    graph_def = LoadGraphFromBase64(base64_graph)
    targets = args.targets
    if not targets:
        if not args.yaml:
            print("No target to backtrace, return.")
            return
        _, _, targets = LoadConfigFromYaml(yaml_file)
    targets = [target.replace(":0", "") for target in targets]
    ret = DoBacktrace(graph_def, targets)
    if args.output:
        with open(args.output, "w") as fp:
            for k, v in ret.items():
                fp.write("====== {} =====\n".format(k))
                for item in v:
                    fp.write("  " + ", ".join(item) + "\n")


def DoOptimize(
    pb_file, yaml_file, out_file, fused=False, implicit_batch=True, output_txt=False
):
    base64_graph = LoadGraphPbAsBase64(pb_file)
    _, _, outputs = LoadConfigFromYaml(yaml_file)
    print(outputs)
    optimized_graph = Optimize(
        base64_graph, outputs, use_fused_op=fused, implicit_batch=implicit_batch
    )
    if out_file:
        SaveOutputBase64Pb(optimized_graph, out_file, output_txt)


def OptimizeGraph(args):
    DoOptimize(
        args.file,
        args.yaml,
        args.output,
        fused=args.fused,
        implicit_batch=args.implicit_batch,
        output_txt=args.txt,
    )


def Transform(args):
    targets = args.targets
    base64_graph = LoadGraphPbAsBase64(args.file)
    graph_def = LoadGraphFromBase64(base64_graph)
    # graph_def, mio_v, mio_a = RemoveMioVariable(graph_def)
    transforms = [
        "strip_unused_nodes",
        "remove_nodes(op=StopGradient)",
        "merge_duplicate_nodes",
    ]
    optimized_graph_def = TransformGraph(graph_def, [], targets, transforms)
    # if not args.strip:
    #   optimized_graph = RestoreMioVariable(optimized_graph_def, mio_v, mio_a)
    if args.output:
        SaveOutputPb(optimized_graph_def, args.output, args.txt)


def IsInternalInputBatchIndex(name):
    if name.startswith("COMPRESS_INDEX__"):
        return True
    else:
        return False


def ChangeInput(graph_def, old_name, new_name):
    for node in graph_def.node:
        for idx, input_name in enumerate(node.input):
            if input_name == old_name:
                node.input[idx] = new_name


def GetInputNode(graph_def, old_name):
    ret = []
    for node in graph_def.node:
        for idx, input_name in enumerate(node.input):
            if input_name == old_name:
                ret.append(node)
                break
    return ret


def FuseCast(graph_def):
    """
    Placeholder 或者 const op 后跟 cast，且 op 的 dtype 为 fp32， cast 为 fp32 -> fp16 的情况，
    将 op 的 dtype 改为 fp16，并移除 cast op
    """
    while True:
        found = False
        for idx, node in enumerate(graph_def.node):
            if node.op != "Cast":
                continue
            if (
                node.attr["SrcT"].type != tpb.DataType.DT_FLOAT
                or node.attr["DstT"].type != tpb.DataType.DT_HALF
            ):
                continue
            input_idx, input_node = FindNodeInfo(graph_def, node.input[0])
            if input_node.op != "Const" and input_node.op != "Placeholder":
                continue
            if input_node.attr["dtype"].type != tpb.DataType.DT_FLOAT:
                continue
            found = True
            input_node.attr["dtype"].type = tpb.DataType.DT_HALF
            for n in graph_def.node:
                for i, input_name in enumerate(n.input):
                    if input_name == node.name:
                        n.input[i] = input_node.name
            del graph_def.node[idx]
            break
        if not found:
            return graph_def


def DoConvert(graph_def, params, inputs, targets):
    transforms = ["strip_unused_nodes", "remove_nodes(op=StopGradient)"]
    optimized_graph = TransformGraph(graph_def, [], targets, transforms)

    for idx, node in enumerate(optimized_graph.node):
        if node.op != "VariableFromMioComponentTable":
            continue
        container_name = node.attr["container"].s.decode("utf-8")
        if container_name in inputs:
            placeholder = npb.NodeDef()
            placeholder.name = container_name
            placeholder.op = "Placeholder"
            placeholder.attr["dtype"].type = tpb.DataType.DT_FLOAT
            placeholder.attr["shape"].CopyFrom(node.attr["shape"])
            optimized_graph.node.append(placeholder)
            ChangeInput(optimized_graph, node.name, container_name)
        elif container_name in params:
            param_node = npb.NodeDef()
            param_node.name = container_name
            param_node.op = "Const"
            param_node.attr["dtype"].type = tpb.DataType.DT_FLOAT
            param_node.attr["value"].tensor.dtype = tpb.DataType.DT_FLOAT
            param_node.attr["value"].tensor.tensor_shape.CopyFrom(
                node.attr["shape"].shape
            )
            optimized_graph.node.append(param_node)
            ChangeInput(optimized_graph, node.name, container_name)
        elif IsInternalInputBatchIndex(container_name):
            # compress index to placeholder
            placeholder = npb.NodeDef()
            placeholder.name = container_name
            placeholder.op = "Placeholder"
            placeholder.attr["dtype"].type = tpb.DataType.DT_FLOAT
            placeholder.attr["shape"].CopyFrom(node.attr["shape"])
            dim = placeholder.attr["shape"].shape.dim.add()
            dim.size = 1

            ## add reshape after cast
            intput_nodes = GetInputNode(optimized_graph, node.name)
            assert len(intput_nodes) == 1
            cast_node = intput_nodes[0]
            assert cast_node.op == "Cast"
            # shape of reshape
            shape_node = npb.NodeDef()
            shape_node.name = cast_node.name + "_const_shape"
            shape_node.op = "Const"
            shape_node.attr["dtype"].type = tpb.DataType.DT_INT32
            shape_node.attr["value"].tensor.dtype = tpb.DataType.DT_INT32
            dim = shape_node.attr["value"].tensor.tensor_shape.dim.add()
            dim.size = -1
            # reshape
            reshape_node = npb.NodeDef()
            reshape_node.name = cast_node.name + "_reshape"
            reshape_node.op = "Reshape"
            reshape_node.attr["T"].type = tpb.DataType.DT_INT32
            reshape_node.attr["Tshape"].type = tpb.DataType.DT_INT32
            ChangeInput(optimized_graph, cast_node.name, reshape_node.name)
            in1 = reshape_node.input.append(cast_node.name)
            reshape_node.input.append(shape_node.name)

            ChangeInput(optimized_graph, node.name, container_name)

            optimized_graph.node.append(reshape_node)
            optimized_graph.node.append(shape_node)
            optimized_graph.node.append(placeholder)
        else:
            print(container_name)
            assert False

    optimized_graph_def = TransformGraph(optimized_graph, [], targets, transforms)
    return optimized_graph_def


def Convert(args):
    base64_graph = LoadGraphPbAsBase64(args.file)
    graph_def = LoadGraphFromBase64(base64_graph)
    if not args.yaml and not args.json:
        print("at least one yaml or json should be specified.")
        return
    if args.yaml:
        params, inputs, targets = LoadConfigFromYaml(args.yaml)
    if args.json:
        params, inputs, targets = LoadConfigFromJson(args.json)
    if args.optimize:
        base64_graph = LoadGraphDefAdBase64(graph_def)
        base64_graph = Optimize(
            base64_graph, targets, use_fused_op=True, implicit_batch=False
        )
        graph_def = LoadGraphFromBase64(base64_graph)
    converted_graph = DoConvert(graph_def, params, inputs, targets)

    if args.fuse_cast:
        converted_graph = FuseCast(converted_graph)

    # converted_graph = TransformGraph(converted_graph, inputs, targets, ['fold_old_batch_norms'])
    if args.output:
        SaveOutputPb(converted_graph, args.output, args.txt)


def ToOnnx(args):
    base64_graph = LoadGraphPbAsBase64(args.file)
    graph_def = LoadGraphFromBase64(base64_graph)
    if not args.yaml and not args.json:
        print("at least one yaml or json should be specified.")
        return
    if args.yaml:
        params, inputs, targets = LoadConfigFromYaml(args.yaml)
    if args.json:
        params, inputs, targets = LoadConfigFromJson(args.json)

    normal_graph_def = DoConvert(graph_def, params, inputs, targets)

    targets = [t + ":0" for t in targets]
    inputs = [i + ":0" for i in inputs]
    model_proto, external_tensor_storage = tf2onnx.convert.from_graph_def(
        normal_graph_def, input_names=inputs, output_names=targets
    )
    if args.output:
        SaveOnnxModel(model_proto, args.output)


def Args():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="graph pb file")

    subparser = parser.add_subparsers()

    optimizer_parser = subparser.add_parser("optimize", help="optimize graph")
    optimizer_parser.add_argument("yaml", help="yaml file")
    optimizer_parser.add_argument("-o", "--output", help="output optimized graph")
    optimizer_parser.add_argument("-t", "--txt", default=False, help="save pbtxt")
    optimizer_parser.add_argument(
        "-f", "--fused", action="store_true", help="use fused op"
    )
    optimizer_parser.add_argument(
        "-i",
        "--implicit_batch",
        action="store_true",
        help="whether use implicit batch mode.",
    )
    optimizer_parser.set_defaults(func=OptimizeGraph)

    backtrace_parser = subparser.add_parser("backtrace", help="backtrace graph.")
    backtrace_parser.add_argument("-y", "--yaml", help="yaml file")
    backtrace_parser.add_argument("-o", "--output", help="output backtrace result.")
    backtrace_parser.add_argument("--txt", action="store_true", help="save pbtxt")
    backtrace_parser.add_argument(
        "-t", "--targets", nargs="*", help="targets node to backtrace."
    )
    backtrace_parser.set_defaults(func=Backtrace)

    transform_parser = subparser.add_parser("transform", help="transform graph.")
    transform_parser.add_argument(
        "-t", "--targets", nargs="+", help="targets node to transform."
    )
    transform_parser.add_argument("-o", "--output", help="output transformed graph")
    transform_parser.add_argument("--txt", action="store_true", help="save pbtxt")
    transform_parser.add_argument(
        "-s", "--strip", action="store_true", help="strip mio variable"
    )
    transform_parser.set_defaults(func=Transform)

    convert_parser = subparser.add_parser(
        "convert", help="convert mio-tf to normal tf graph."
    )
    convert_parser.add_argument("-y", "--yaml", help="yaml file")
    convert_parser.add_argument("-j", "--json", help="target output")
    convert_parser.add_argument("-o", "--output")
    convert_parser.add_argument("-t", "--txt", action="store_true", help="save pbtxt")
    convert_parser.add_argument(
        "--optimize", action="store_true", help="optimize graph"
    )
    convert_parser.add_argument(
        "--fuse_cast", action="store_true", help="fuse cast in tf fp16"
    )
    convert_parser.set_defaults(func=Convert)

    # onnx_parser = subparser.add_parser('onnx', help='convert mio-tf to onnx graph')
    # onnx_parser.add_argument('-j', '--json', help='json file')
    # onnx_parser.add_argument('-y', '--yaml', help='yaml file')
    # onnx_parser.add_argument('-o', '--output')
    # onnx_parser.set_defaults(func=ToOnnx)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = Args()
    args.func(args)
