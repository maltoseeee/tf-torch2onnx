import onnx
import torch


def dtype_to_onnx_tensor_type_str(dtype: torch.dtype):
  """all valid onnx tensor type strs reference `onnx/defs/schema.cc`
  """
  if dtype == torch.int8:
    return "tensor(int8)"
  if dtype == torch.int16:
    return "tensor(int16)"
  if dtype == torch.int32:
    return "tensor(int32)"
  if dtype == torch.int64:
    return "tensor(int64)"
  if dtype == torch.float16:
    return "tensor(float16)"
  if dtype == torch.float32:
    return "tensor(float)"
  if dtype == torch.bfloat16:
    return "tensor(bfloat16)"
  assert False, f"dtype not supported: {dtype}"

def dtype_to_string(dtype: torch.dtype):
  if dtype == torch.int8:
    return "int8"
  if dtype == torch.int16:
    return "int16"
  if dtype == torch.int32:
    return "int32"
  if dtype == torch.int64:
    return "int64"
  if dtype == torch.float16:
    return "float16"
  if dtype == torch.float32:
    return "float32"
  if dtype == torch.bfloat16:
    return "bfloat16"
  assert False, f"dtype not supported: {dtype}"

def dtype_string_to_tensor_proto_dtype(dtype: str):
  if isinstance(dtype, bytes):
    dtype = dtype.decode()
  if dtype == "int8":
    return onnx.TensorProto.INT8
  if dtype == "int16":
    return onnx.TensorProto.INT16
  if dtype == "int32":
    return onnx.TensorProto.INT32
  if dtype == "int64":
    return onnx.TensorProto.INT64
  if dtype == "float16":
    return onnx.TensorProto.FLOAT16
  if dtype == "float32":
    return onnx.TensorProto.FLOAT
  if dtype == "bfloat16":
    return onnx.TensorProto.BFLOAT16
  assert False, f"dtype not supported: {dtype}"
