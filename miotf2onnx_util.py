import sys, os, copy
import json
import base64
from typing import Dict, List, Any
import tf2onnx
import onnx

import tensorflow as tf
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import node_def_pb2
from tensorflow.core.framework import tensor_shape_pb2
from .graph_transforms import TransformGraph
from .tensorrt_optimizer import SpecialRuleRemoveReshape
from .miotf_util import (
    IsCompressIndexName,
    FindCompressIndexNode,
    GetContainerName,
    GetRealName,
    ChangeInputsAll,
    ChangeInputs,
    BuildTFConstNode,
)

extra_opset = []
if os.environ.get("USE_OPTIMIZE", True):
    from .tensorflow_custom_ops.tf2onnx_custom import KS_RECO_OPSET

    extra_opset = [KS_RECO_OPSET]

CUSTOM_ONNX_DOMAIN = "com.kuaishou.reco_arch"
CUSTOM_ONNX_OPSET = 1


def read_unipredict_fused_from_json(json_path: str) -> List[Dict[str, Any]]:
    """递归读取 json 配置中所有 `uni_predict_fused*` 条目，并抽取模型信息。

    你的动态 json 通常是多层嵌套的（例如在 pipeline->processor 下），所以这里会递归遍历整个 json 树。

    返回 list，每个元素包含：
    - onnx_name: 使用 uni_predict_fused_* 对象内部的 `obj["key"]` 作为最终 onnx 命名
    - mio_graph_b64: obj["graph"]
    - inputs: obj["inputs"][*]["tensor_name"]
    - output_tensor_names: obj["outputs"][*]["tensor_name"]
    - params: obj["param"][*]["name"]

    说明：
    - 该函数只做字段抽取，不做 mio->onnx 转换。
    - 若匹配到的条目缺少关键字段，会抛出 KeyError/TypeError，方便你尽早发现配置问题。
    """

    with open(json_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    results: List[Dict[str, Any]] = []

    def _walk(obj: Any) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(k, str) and k.startswith("uni_predict_fused"):
                    if not isinstance(v, dict):
                        raise TypeError(
                            f"json key '{k}' 对应的 value 不是 obj(dict)，实际是: {type(v)}"
                        )

                    # 按你的需求：使用内部字段 `key` 作为最终 onnx 名称
                    onnx_name = v["key"]
                    mio_graph_b64 = v["graph"]
                    inputs = [e["tensor_name"] for e in v.get("inputs", [])]
                    output_tensor_names = [e["tensor_name"] for e in v.get("outputs", [])]
                    params = [e["name"] for e in v.get("param", [])]

                    results.append(
                        {
                            "onnx_name": onnx_name,
                            "mio_graph_b64": mio_graph_b64,
                            "inputs": inputs,
                            "output_tensor_names": output_tensor_names,
                            "params": params,
                        }
                    )

                _walk(v)
        elif isinstance(obj, list):
            for it in obj:
                _walk(it)

    _walk(cfg)
    return results


def convert_mio_variable(mio_graph_def, inputs):
    lookup_table = {}
    for idx, node in enumerate(mio_graph_def.node):
        real_name = ""
        if node.op == "VariableFromMioComponentTable":  # V1
            real_name = GetContainerName(node)
        elif "VAR_SPLIT_" in node.name:  # V2
            real_name = GetRealName(node)

        if not real_name:
            continue
        if "dtype" not in node.attr:
            node.attr["dtype"].type = tf.float32.as_datatype_enum

        if real_name in inputs or IsCompressIndexName(real_name):
            lookup_table[node.name] = real_name
            node.op = "Placeholder"
            node.name = real_name
        else:
            print(
                f"unused node: {node.name} real_name: {real_name} op: {node.op}, unknown mio variable ..."
            )

        del node.input[:]
        if "container" in node.attr:
            del node.attr["container"]

    ChangeInputsAll(mio_graph_def, lookup_table)
    return mio_graph_def


def process_compress_index(graph_def):
    """
    trick
    mio tensorflow 框架导出的 COMPRESS_INDEX shape 初始为 [-1], 将其转换为[-1, 1], 因为 Unipredict trt_executor 需要二维的输入
    随后再将其 reshape 回 [-1]
    @param graph_def: The TensorFlow GraphDef to modify.
    """
    # 查找 COMPRESS_INDEX_NODE 节点
    compress_index_node = FindCompressIndexNode(graph_def)
    if compress_index_node is None:
        return graph_def
    consumers = [
        node for node in graph_def.node if compress_index_node.name in node.input
    ]
    if len(consumers) != 1:
        raise ValueError(
            f"COMPRESS_INDEX_NODE should have exactly 1 consumer, found {len(consumers)}"
        )
    if consumers[0].op != "Cast":
        raise ValueError("Consumer of COMPRESS_INDEX_NODE is not a Cast operation")
    cast_node = consumers[0]
    attrs = cast_node.attr
    if attrs["DstT"].type != tf.int32.as_datatype_enum:
        raise ValueError("Cast node DstT is not int32")
    if attrs["SrcT"].type != tf.float32.as_datatype_enum:
        raise ValueError("Cast node SrcT is not float32")

    # COMPRESS_INDEX shape 从 [-1] 修改为 [-1, 1]
    compress_index_node.attr["dtype"].type = tf.float32.as_datatype_enum
    compress_index_node.attr["shape"].shape.dim.add().size = 1

    reshape_const_shape_node = BuildTFConstNode(
        f"{compress_index_node.name}_const_shape", tf.int32.as_datatype_enum, [-1]
    )

    # 创建新的 Reshape 节点
    cast_reshape_node = node_def_pb2.NodeDef()
    cast_reshape_node.name = f"{cast_node.name}_reshape"
    cast_reshape_node.op = "Reshape"
    cast_reshape_node.input.extend([cast_node.name, reshape_const_shape_node.name])
    cast_reshape_node.attr["T"].type = tf.int32.as_datatype_enum
    cast_reshape_node.attr["Tshape"].type = tf.int32.as_datatype_enum

    # 先更新后续节点的输入, 防止新增节点的输入也被修改
    ChangeInputs(graph_def, cast_node.name, cast_reshape_node.name)

    # 将新节点添加到图中
    graph_def.node.extend([reshape_const_shape_node, cast_reshape_node])
    return graph_def


def remove_unused_inputs(graph_def, inputs):
    real_inputs = set([])
    for idx, node in enumerate(graph_def.node):
        if not node.input or len(node.input) == 0:
            real_inputs.add(node.name)

    ret_inputs = list()
    for e in inputs:
        if e in real_inputs:
            ret_inputs.append(e)

    return ret_inputs


def generate_random_initializer(initializer):
    """为 ONNX initializer 生成合理的随机数据。

    之前的实现对所有类型都用 `np.random.rand`：
    - 对 int/bool 会先生成 float 再 cast，分布不直观；
    - 对空 shape/标量 shape 等边界情况不够明确；

    这里做了更稳妥的处理：
    - float/float16: 正态分布（均值0，方差较小），更接近权重初始化
    - int32/int64: small range 随机整数
    - bool: 0/1 伯努利
    """

    import numpy as np

    data_type = initializer.data_type
    dims = list(initializer.dims)

    # ONNX 里 initializer 允许标量（dims=[]）
    shape = tuple(int(d) for d in dims)

    if data_type == onnx.TensorProto.FLOAT:
        return np.random.normal(loc=0.0, scale=0.02, size=shape).astype(np.float32)
    if data_type == onnx.TensorProto.FLOAT16:
        return np.random.normal(loc=0.0, scale=0.02, size=shape).astype(np.float16)
    if data_type == onnx.TensorProto.INT32:
        return np.random.randint(low=-128, high=128, size=shape, dtype=np.int32)
    if data_type == onnx.TensorProto.INT64:
        # numpy randint 不支持 dtype=int64 的所有版本参数一致，这里先生成再 astype
        return np.random.randint(low=-128, high=128, size=shape).astype(np.int64)
    if data_type == onnx.TensorProto.BOOL:
        return (np.random.rand(*shape) > 0.5) if shape else (np.random.rand() > 0.5)

    raise NotImplementedError(f"Unsupported data type: {data_type}")


def fill_random_weights_to_onnx_model(
    empty_params_model,
    output_path: str,
    params,
    *,
    save_as_external_data: bool = True,
    all_tensors_to_one_file: bool = True,
    external_data_filename: str | None = None,
):
    """把指定 params 对应的 initializer 填充随机权重并保存。

    优化点：
    1) 之前 `initializer_data` 没有被使用，去掉。
    2) 使用 `params_set` 加速匹配。
    3) 支持自定义 external data 文件名（避免默认命名不稳定）。
    """

    from onnx import numpy_helper

    model = onnx.ModelProto()
    model.CopyFrom(empty_params_model)

    params_set = set(params)
    filled = 0

    for initializer in model.graph.initializer:
        if initializer.name not in params_set:
            continue
        random_array = generate_random_initializer(initializer)
        initializer.CopyFrom(numpy_helper.from_array(random_array, name=initializer.name))
        filled += 1

    if external_data_filename is None:
        # 默认使用输出 onnx 文件名 + .bin，避免多个模型共享同一个 weights.bin
        base = os.path.basename(output_path)
        base_no_ext = os.path.splitext(base)[0]
        external_data_filename = f"{base_no_ext}.bin"

    # `onnx.save` 在 save_as_external_data=True 时，会把大 tensor 写到 external data
    # 并且 `location` 默认可能变化；这里显式指定 all_tensors_to_one_file 和文件名
    onnx.save(
        model,
        output_path,
        save_as_external_data=save_as_external_data,
        all_tensors_to_one_file=all_tensors_to_one_file,
        location=external_data_filename,
    )
    print(f"✅ save random weight onnx to '{output_path}' success! filled={filled}")


def recover_params_as_initializer(onnx_model, params):
    params_set = set(params)
    left_inputs = list()
    for inv in onnx_model.graph.input:
        if inv.name.split(":")[0] in params_set:
            param = onnx.TensorProto()
            param.name = inv.name
            param.data_type = inv.type.tensor_type.elem_type
            param.dims.extend(dim.dim_value for dim in inv.type.tensor_type.shape.dim)
            # param.raw_data = b''
            # param.data_location = onnx.TensorProto.DataLocation.EXTERNAL;
            # kv = onnx.StringStringEntryProto()
            # kv.key = "location"
            # kv.value = ""
            # param.external_data.append(kv)
            onnx_model.graph.initializer.append(param)
            # print(f"delete param: {inv.name} from input, and add to initializer ... ")
            # print(param)
            # print(inv)
        else:
            left_inputs.append(inv)

    del onnx_model.graph.input[:]
    for inv in left_inputs:
        onnx_model.graph.input.append(inv)


def convert_to_no_idx_format(onnx_model):
    """
    tf2onnx 转换要求 tf 的模型输入输出名格式为 ".*:idx"
    导致转换出的 onnx graph 的输入名带有 "idx", 这里删掉它, 和 yaml 文件保持一致
    """
    lookup_table = {}
    for input in onnx_model.graph.input:
        lookup_table[input.name] = input.name.split(":")[0]
        input.name = input.name.split(":")[0]
    for node in onnx_model.graph.node:
        new_inputs = []
        for input_name in node.input:
            new_inputs.append(
                lookup_table[input_name] if input_name in lookup_table else input_name
            )
        del node.input[:]
        node.input.extend(new_inputs)


# message ModelProto
def tf2onnx_from_graph_def(
    tf_graph_def, inputs, params, output_tensor_names, opset=None, is_export_onnx=False
):
    """
    @param:
      tf_graph_def: pure tf graph def
      inputs: model's input node name (exclude ":idx")
      params: model's param node name (exclude ":idx")
      output_tensor_names: model's output tensor name (include "idx")
    """
    input_tensor_names = [e + ":0" for e in inputs] + [
        e + ":0" for e in params
    ]  # tf2onnx 需要 ":idx" 格式的输入输出
    onnx_model_def, external_tensor_storage = tf2onnx.convert.from_graph_def(
        tf_graph_def,
        input_names=input_tensor_names,
        output_names=output_tensor_names,
        opset=opset,
        extra_opset=extra_opset,
    )
    onnx.checker.check_model(onnx_model_def)

    convert_to_no_idx_format(onnx_model_def)
    onnx.checker.check_model(onnx_model_def)

    recover_params_as_initializer(onnx_model_def, params)

    if is_export_onnx:
        fill_random_weights_to_onnx_model(onnx_model_def, "random.onnx", params)

    print("✅ tf graph converted to onnx graph success!")
    return onnx_model_def


def miotf_to_tf(mio_graph_def, inputs, params, output_tensor_names, optimizer=None):
    """
    @param:
      mio_graph_def: mio tf model_proto
      inputs: model's input node name (exclude ":idx")
      params: model's param node name (exclude ":idx")
      output_tensor_names: model's output tensor name (include "idx")
      optimizer: list[str]
    """
    inputs = list(set(inputs))
    params = list(set(params))
    output_tensor_names = list(set(output_tensor_names))

    inputs = [e.split(":")[0] for e in inputs]
    params = [e.split(":")[0] for e in params]
    outputs = [e.split(":")[0] for e in output_tensor_names]

    # convert VariableFromMioComponentTable to placeholder
    tf_graph = convert_mio_variable(mio_graph_def, inputs + params)
    tf_graph = process_compress_index(tf_graph)

    # NOTE tricky problem
    # 原来的 TensorRT 图优化也使用了这个, 目前发现主站精排的两个小模型中存在这种模式,需要进行修改
    # 如果遇到问题可以尝试注释下面一行
    tf_graph = SpecialRuleRemoveReshape(tf_graph)

    # remove unused nodes
    tf_graph = TransformGraph(
        tf_graph,
        inputs + params,
        outputs,
        ["strip_unused_nodes", "remove_nodes(op=StopGradient)"],
    )

    # NOTE [Critical], 删去 inputs, params 中存在但 graph 中不存在的参数
    inputs = remove_unused_inputs(tf_graph, inputs)
    params = remove_unused_inputs(tf_graph, params)

    print("✅ miotf graph converted to tf graph success!")
    return tf_graph, inputs, params, output_tensor_names


def miotf_to_onnxb64(
    mio_graph_def, inputs, params, output_tensor_names, opset=None, is_export_onnx=False
):
    """
    return a base64 encoded onnx model_proto from mio tf model_proto.
    @param:
      mio_graph_def: mio tf graph def
      inputs: model's input node name (exclude ":idx")
      params: model's param node name (exclude ":idx")
      output_tensor_names: model's output tensor name (include "idx")
    """
    pure_tf_graph_def, inputs, params, output_tensor_names = miotf_to_tf(
        mio_graph_def, inputs, params, output_tensor_names
    )
    onnx_graph_def = tf2onnx_from_graph_def(
        pure_tf_graph_def,
        inputs,
        params,
        output_tensor_names,
        opset=opset,
        is_export_onnx=is_export_onnx,
    )
    return "base64://" + base64.b64encode(onnx_graph_def.SerializeToString()).decode(
        "ascii"
    )


def export_unipredict_fused_json_to_onnx(
    json_path: str,
    output_dir: str = "./onnx_out",
    opset=None,
    is_export_onnx: bool = False,
    overwrite: bool = True,
) -> Dict[str, str]:
    """从 dynamic json 中找到所有 uni_predict_fused_*，转换为 onnx base64 并可选导出 onnx 文件。

    - 返回：`{onnx_name(条目内部 key 字段) -> base64://...}`
    - 若 `is_export_onnx=True`：同时在 `output_dir` 下导出 `{onnx_name}.onnx`
    """

    if is_export_onnx:
        os.makedirs(output_dir, exist_ok=True)

    models = read_unipredict_fused_from_json(json_path)
    ret: Dict[str, str] = {}

    for m in models:
        onnx_name = m["onnx_name"]

        onnx_b64 = miotfb64_to_onnxb64(
            m["mio_graph_b64"],
            m["inputs"],
            m["params"],
            m["output_tensor_names"],
            opset=opset,
            is_export_onnx=is_export_onnx,
        )
        ret[onnx_name] = onnx_b64

        if is_export_onnx:
            out_path = os.path.join(output_dir, f"{onnx_name}.onnx")
            if (not overwrite) and os.path.exists(out_path):
                continue

            raw_b64 = onnx_b64[9:] if onnx_b64.startswith("base64://") else onnx_b64
            onnx_bytes = base64.b64decode(raw_b64.encode("ascii"))
            with open(out_path, "wb") as f:
                f.write(onnx_bytes)

    return ret


def miotfb64_to_onnxb64(
    mio_graph_b64, inputs, params, output_tensor_names, opset=None, is_export_onnx=False
):
    """
    return a base64 encoded onnx model_proto from base64 encode mio tf model_proto.
    @param:
      mio_graph_b64: a base64 encoded mio tf graph def
      inputs: model's input node name (exclude ":idx")
      params: model's param node name (exclude ":idx")
      output_tensor_names: model's output tensor name (include "idx")
    """
    real_graph_b64 = mio_graph_b64
    if mio_graph_b64[:9] == "base64://":
        real_graph_b64 = mio_graph_b64[9:]

    mio_graph = graph_pb2.GraphDef()
    mio_graph.ParseFromString(base64.b64decode(real_graph_b64.encode("ascii")))
    return miotf_to_onnxb64(
        mio_graph,
        inputs,
        params,
        output_tensor_names,
        opset=opset,
        is_export_onnx=is_export_onnx,
    )
