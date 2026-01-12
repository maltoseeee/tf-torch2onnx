from dataclasses import dataclass
from typing import Dict, List, Union

import onnx
import torch
import torch.nn as nn
from torch.onnx.symbolic_helper import _get_tensor_sizes

from .constants import KS_RECO_ONNX_DOMAIN, KS_RECO_ONNX_VERSION
from .util import dtype_to_onnx_tensor_type_str, dtype_to_string, dtype_string_to_tensor_proto_dtype


class RagByDragonDelegate(torch.autograd.Function):
  supported_dtypes = (torch.int64,
                      torch.float32,
                      )
  OP_NAME = "RagByDragonDelegate"

  @dataclass
  class OutputMeta:
    res_attr: str
    dtype: torch.dtype
    shape: torch.Size

  class Config:
    def __init__(self,
                 req_attr: Union[str | List[str]],
                 output_metas: Union["OutputMeta" | List["OutputMeta"]],
                 max_batch_size: int,
                 model_key: str,
                 kess_service: str,
                 kess_cluster: str = "PRODUCTION",
                 kess_group: str = "",
                 timeout_ms: int = 300,
                 request_type: str = "",
                 dry_run: int = 0,
                 ) -> None:
      """
      req_attr: name for each of `input_tensors` as request attr to downstream rpc server
      output_metas: meta info of output tensors
      max_batch_size: max value for batch dim, i.e. axis = 0
      model_key: model key, used for perf
      kess_service: kess name of downstream rpc server
      kess_cluster: kess cluster
      kess_group: kess group
      timeout_ms: timeout
      request_type: request type
      default: default value for output tensor if error
      dry_run: set 1 to skip rag, such as when used in trtexec
      """
      self.req_attr = req_attr if isinstance(req_attr, list) else [req_attr]
      self.output_metas = output_metas if isinstance(output_metas, list) else [output_metas]
      assert max_batch_size > 0, f"invalid max_batch_size: {max_batch_size}"
      self.max_batch_size = max_batch_size
      self.model_key = model_key
      self.kess_service = kess_service
      self.kess_cluster = kess_cluster
      self.kess_group = kess_group
      self.timeout_ms = timeout_ms
      self.request_type = request_type
      self.dry_run = dry_run

  @staticmethod
  def check_tensor(i, tensor: torch.Tensor, max_batch_size):
    assert isinstance(tensor, torch.Tensor), f"the {i} input tensor is not a tensor"
    assert len(tensor.shape) == 2, f"rank of the {i} input tensor should be 2: {len(tensor.shape)}"
    assert tensor.shape[0] <= max_batch_size, f"batch size(axis=0) of the {i} input tensor vs max_batch_size: {tensor.shape[0]} vs {max_batch_size}"
    assert tensor.shape[1] > 0, f"dim(1) of the {i} input tensor should be positive: {tensor.shape}"
    assert tensor.dtype in __class__.supported_dtypes, f"dtype of the {i} input tensor not supported_dtypes: {tensor.dtype}"

  @staticmethod
  def check_input_tensors(input_tensors, config):
    assert isinstance(input_tensors, tuple) and len(input_tensors) >= 1
    t0 = None
    for i, tensor in enumerate(input_tensors):
      __class__.check_tensor(i, tensor, config.max_batch_size)
      if t0 is None:
        t0 = tensor
      else:
        assert t0.shape[0] == tensor.shape[
            0], f"dim(0) changed: the 0 input tensor({t0.shape}) vs the {i} input tensor({tensor.shape}) "

  @staticmethod
  def check_meta(i, meta: OutputMeta, max_batch_size):
    assert isinstance(meta, __class__.OutputMeta), f"the {i} output meta is not a OutputMeta"
    assert meta.res_attr, f"`res_attr` the {i} output meta should not be emtpy"
    assert len(meta.shape) == 2, f"rank of the {i} output meta should be 2: {len(meta.shape)}"
    assert meta.shape[0] <= max_batch_size, f"batch size(axis=0) of the {i} out meta vs max_batch_size: {meta.shape[0]} vs {max_batch_size}"
    assert meta.shape[1] > 0, f"dim(1) of the {i} output meta should be positive: {meta.shape}"
    assert meta.dtype in __class__.supported_dtypes, f"dtype of the {i} output meta not supported_dtypes: {meta.dtype}"

  @staticmethod
  def check_onput_metas(output_metas, config):
    assert isinstance(output_metas, list) and len(output_metas) >= 1
    m0 = None
    for i, meta in enumerate(output_metas):
      __class__.check_meta(i, meta, config.max_batch_size)
      if m0 is None:
        m0 = meta
      else:
        assert m0.shape[0] == meta.shape[0], f"dim(0) changed: the 0 output meta({m0.shape}) vs the {i} output meta({meta.shape}) "

  @staticmethod
  def deregister_schema_op():
    onnx.defs.deregister_schema(__class__.OP_NAME, KS_RECO_ONNX_VERSION, KS_RECO_ONNX_DOMAIN)

  @staticmethod
  def register_schema_op():
    from onnx.defs import OpSchema
    inputs = [OpSchema.FormalParameter(name="inputs", type_str="T", description="",
                                       param_option=OpSchema.FormalParameterOption.Variadic, is_homogeneous=False)]
    outputs = [OpSchema.FormalParameter(name="outputs", type_str="T", description="",
                                        param_option=OpSchema.FormalParameterOption.Variadic, is_homogeneous=False)]
    type_constraints = [("T", [dtype_to_onnx_tensor_type_str(i)
                         for i in __class__.supported_dtypes], "supported types")]
    attributes = [
        OpSchema.Attribute("req_attr", OpSchema.AttrType.STRINGS),
        OpSchema.Attribute("req_dtype", OpSchema.AttrType.STRINGS),
        OpSchema.Attribute("req_dim1", OpSchema.AttrType.INTS),
        OpSchema.Attribute("res_attr", OpSchema.AttrType.STRINGS),
        OpSchema.Attribute("res_dtype", OpSchema.AttrType.STRINGS),
        OpSchema.Attribute("res_dim1", OpSchema.AttrType.INTS),
        OpSchema.Attribute("model_key", OpSchema.AttrType.STRING),
        OpSchema.Attribute("kess_service", OpSchema.AttrType.STRING),
        OpSchema.Attribute("kess_cluster", OpSchema.AttrType.STRING),
        OpSchema.Attribute("kess_group", OpSchema.AttrType.STRING),
        OpSchema.Attribute("timeout_ms", OpSchema.AttrType.INT),
        OpSchema.Attribute("request_type", OpSchema.AttrType.STRING),
        OpSchema.Attribute("max_batch_size", OpSchema.AttrType.INT),
        OpSchema.Attribute("dry_run", OpSchema.AttrType.INT),
    ]
    schema = OpSchema(__class__.OP_NAME, KS_RECO_ONNX_DOMAIN, KS_RECO_ONNX_VERSION, inputs=inputs,
                      outputs=outputs, type_constraints=type_constraints, attributes=attributes)
    schema.set_type_and_shape_inference_function(__class__.shape_inference)
    onnx.defs.register_schema(schema)

  @staticmethod
  def shape_inference(ctx: onnx.shape_inference.InferenceContext):
    num_inputs = ctx.get_num_inputs()
    assert num_inputs >= 1, f"num_inputs({num_inputs}) >= 1"
    rank = 2
    batch_size = -1
    for i in range(num_inputs):
      local_rank = len(ctx.get_input_type(i).tensor_type.shape.dim)
      assert rank == local_rank, f"rank of the `{i}` input != {rank}: {local_rank}"
      batch_dim = ctx.get_input_type(i).tensor_type.shape.dim[0]
      if not batch_dim.HasField("dim_value"):
        continue
      local_batch_size = batch_dim.dim_value
      if batch_size <= 0:
        batch_size = local_batch_size
      assert batch_size == local_batch_size, f"batch dim(axis=0) of the `0` input should equal the `{i}` input: {batch_size} vs {local_batch_size}"

    num_outputs = ctx.get_num_outputs()
    assert num_outputs >= 1, f"num_outputs({num_outputs}) >= 1"
    res_dtype = ctx.get_attribute("res_dtype")
    assert num_outputs == len(
        res_dtype.strings), f"len(res_dtype) vs num_outputs: {len(res_dtype.strings)} vs {num_outputs}"
    res_dim1 = ctx.get_attribute("res_dim1")
    assert num_outputs == len(res_dim1.ints), f"len(res_dim1) vs num_outputs: {len(res_dim1.ints)} vs {num_outputs}"
    for i in range(num_outputs):
      output = ctx.get_output_type(i)
      output.tensor_type.elem_type = dtype_string_to_tensor_proto_dtype(res_dtype.strings[i])
      del output.tensor_type.shape.dim[:]
      dim0 = output.tensor_type.shape.dim.add()
      if batch_size <= 0:
        dim0.dim_param = "-1"
      else:
        dim0.dim_value = batch_size
      dim1 = output.tensor_type.shape.dim.add()
      dim1.dim_value = res_dim1.ints[i]
      local_rank = len(output.tensor_type.shape.dim)
      assert rank == local_rank, f"rank of the `{i}` output != {rank}: {local_rank}"
      ctx.set_output_type(i, output)

  @staticmethod
  def symbolic(g, config: Config, *input_tensors: List[torch.Tensor]):
    args = input_tensors
    req_dim1 = []
    rank = 2
    for i, input_tensor in enumerate(input_tensors):
      sizes = input_tensor.type().sizes() or input_tensor.type().varyingSizes()
      assert len(sizes) == 2, f"rank of the `{i}` input != {rank}: {len(sizes)}"
      if sizes[0]:
        assert sizes[0] <= config.max_batch_size, f"batch size(axis=0) of the {i} input tensor vs max_batch_size: {sizes[0]} vs {config.max_batch_size}"
      req_dim1.append(sizes[1])

    kwargs = {
        "req_attr_s": config.req_attr,
        "req_dtype_s": [dtype_to_string(input_tensor.type().dtype()) for input_tensor in input_tensors],
        "req_dim1_i": req_dim1,
        "res_attr_s": [meta.res_attr for meta in config.output_metas],
        "res_dtype_s": [dtype_to_string(meta.dtype) for meta in config.output_metas],
        "res_dim1_i": [meta.shape[1] for meta in config.output_metas],
        "model_key_s": config.model_key,
        "kess_service_s": config.kess_service,
        "kess_cluster_s": config.kess_cluster,
        "kess_group_s": config.kess_group,
        "timeout_ms_i": config.timeout_ms,
        "request_type_s": config.request_type,
        "max_batch_size_i": config.max_batch_size,
        "dry_run_i": config.dry_run,
    }
    num_outputs = len(config.output_metas)
    outputs = g.op(f"{KS_RECO_ONNX_DOMAIN}::{__class__.OP_NAME}", *args, outputs=num_outputs, **kwargs)
    if num_outputs == 1:
      outputs = [outputs]
    for output, meta in zip(outputs, config.output_metas):
      output_type = torch.TensorType.get().with_dtype(meta.dtype).with_sizes(meta.shape)
      output.setType(output_type)
    if num_outputs == 1:
      outputs = outputs[0]
    return outputs

  @staticmethod
  def forward(ctx, config: Config, *input_tensors: List[torch.Tensor]):
    """
    config: Config
    input_tensors: input tensors
    """
    # We don't have to actually implement the correct forward pass,
    # if the downstream graph is not data dependent,
    # as long as the shape of the output is correct.
    __class__.check_input_tensors(input_tensors, config)
    __class__.check_onput_metas(config.output_metas, config)
    return tuple(torch.zeros(*meta.shape, dtype=meta.dtype) for meta in config.output_metas)
