import json, base64
import tensorflow as tf
import yaml
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import node_def_pb2
from tensorflow.core.framework import tensor_shape_pb2
from tensorflow.python.framework import tensor_util


def GetNode(graph_def, name):
    for node in graph_def.node:
        if node.name == name:
            return node
    return None


def FindNodeInfo(graph, node_name):
    node_name = node_name.split(":")[0]
    for idx, node in enumerate(graph.node):
        if node.name == node_name:
            return idx, node


def IsCompressIndexName(node_name):
    return node_name.find("COMPRESS_INDEX__") != -1


def FindCompressIndexNode(graph_def):
    for idx, node in enumerate(graph_def.node):
        if IsCompressIndexName(node.name):
            return node
    return None


def BuildTFConstNode(name, dtype, const_values):
    node = node_def_pb2.NodeDef()
    node.name = name
    node.op = "Const"
    node.attr["dtype"].type = dtype

    tensor = node.attr["value"].tensor
    tensor.dtype = node.attr["dtype"].type
    for v in const_values:
        tensor.int_val.append(v)

    shape = tensor.tensor_shape
    dim = tensor_shape_pb2.TensorShapeProto.Dim()
    dim.size = len(const_values)
    shape.dim.append(dim)
    return node


def ChangeInputs(graph_def, old_name, new_name):
    for idx, node in enumerate(graph_def.node):
        for i, v in enumerate(node.input):
            if v == old_name:
                node.input[i] = new_name


def ChangeInputsAll(graph_def, lookup_table):
    for idx, node in enumerate(graph_def.node):
        for i, v in enumerate(node.input):
            if v in lookup_table:
                node.input[i] = lookup_table[v]


def GetContainerName(node):
    return node.attr["container"].s.decode("utf-8")


def GetRealName(node):
    idx = node.name.find("VAR_SPLIT_")
    if idx != -1:
        return node.name[idx + 10 :]
    return ""


def extract_json_object(filename, json_path):
    jobj = json.load(open(filename))

    rec = json_path.split(".")
    for key in rec:
        if key in jobj:
            jobj = jobj[key]
        else:
            print(f"json path error @ {key}, full: {json_path}")
            return None

    if not isinstance(jobj, dict):
        print(f"json path error: {json_path}, is not dict value")
        return None

    return jobj


def get_inputs_outputs_params_from_json(config_file, uni_fused_path):
    conf = extract_json_object(config_file, uni_fused_path)
    output_tensor_names = [c["tensor_name"] for c in conf["outputs"]]
    outputs = [e.split(":")[0] for e in output_tensor_names]

    inputs = [c["tensor_name"] for c in conf["inputs"]]
    params = [c["name"] for c in conf["param"]]

    print(f"\noutputs num: {len(outputs)}, ", output_tensor_names)
    print(f"\ninputs num: {len(inputs)}")
    print(f"\nparams num: {len(params)}")

    return inputs, output_tensor_names, outputs, params


def get_inputs_outputs_params_from_yaml(config_file):
    with open(config_file) as f:
        dnn_model = yaml.load(f, Loader=yaml.SafeLoader)
        output_tensors = dnn_model["graph_tensor_mapping"]
        output_tensor_names = [
            output_tensors[e] for e in dnn_model["q_names"].split(" ")
        ]
        outputs = [e.split(":")[0] for e in output_tensor_names]

        slots_config = dnn_model["embedding"]["slots_config"]
        vec_inputs = dnn_model["vec_input"]
        inputs = [c["input_name"] for c in slots_config] + [
            c["name"] for c in vec_inputs
        ]
        params = [c["name"] for c in dnn_model["param"]]

        print(f"\noutputs num: {len(outputs)}, ", output_tensor_names)
        print(f"\ninputs num: {len(inputs)}")
        print(f"\nparams num: {len(params)}")

        return inputs, output_tensor_names, outputs, params
