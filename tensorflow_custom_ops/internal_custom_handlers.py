from tf2onnx.handler import tf_op
from onnx import helper

KS_RECO_ONNX_DOMAIN = "com.kuaishou.reco_arch"
KS_RECO_OPSET = helper.make_opsetid(KS_RECO_ONNX_DOMAIN, 1)


op_mapping = {
    "FusedCompressGather": "fused_compress_gather",
    "FusedGatherMatmul": "fused_gather_batch_matmul",
    "FusedMultiheadAttention": "fused_multihead_attention_dynamic",
    "FusedMultiheadAttentionWithBias": "fused_multihead_attention_dynamic",
    "UniGather": "gather_dynamic",
    "FusedSlotGate": "fused_slot_gate_dynamic",
    "FusedGatedSum": "fused_gated_sum_dynamic",
    "FusedLayerNorm": "fused_layer_norm_dynamic",
    "MixtureOfExperts": "mixture_of_experts",
    "RadixTopK": "radix_topk",
}


@tf_op(
    [
        "FusedCompressGather",
        "FusedGatherMatmul",
        "FusedMultiheadAttention",
        "FusedMultiheadAttentionWithBias",
        "UniGather",
        "FusedSlotGate",
        "FusedGatedSum",
        "FusedLayerNorm",
        "MixtureOfExperts",
    ]
)
class LegacyUniCustomTfOperator:
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        if op_mapping.get(node.type) == None:
            raise RuntimeError(f"custom op {node.type} not supported currently.")
        node.type = op_mapping.get(node.type) # 根据 unipredict trt plugin 的实现指定
        node.domain = KS_RECO_ONNX_DOMAIN
        node.set_attr("type", 0)
        node.set_attr("plugin_version", "1.0")
        # node.set_attr("plugin_namespace", "")

@tf_op(
    [
        "RadixTopK",
    ]
)
class UniCustomTfOperator:
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        if op_mapping.get(node.type) == None:
            raise RuntimeError(f"custom op {node.type} not supported currently.")
        node.type = op_mapping.get(node.type) # 根据 unipredict trt plugin 的实现指定
        node.domain = KS_RECO_ONNX_DOMAIN
        node.set_attr("plugin_version", "1")
        #node.set_attr("plugin_namespace", "")
