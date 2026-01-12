import tensorflow as tf
from tensorflow.python.platform import resource_loader
from .internal_custom_handlers import *

major, minor, _ = map(int, tf.__version__.split('.'))

if major == 1 and minor == 15:
    custom_tf = tf.load_op_library(
        resource_loader.get_path_to_datafile("custom_tf_op_declare_tf15.so")
    )
elif major == 2 and minor <= 8:
    custom_tf = tf.load_op_library(
        resource_loader.get_path_to_datafile("custom_tf_op_declare_tf24.so")
    )
else:
    custom_tf = tf.load_op_library(
        resource_loader.get_path_to_datafile("custom_tf_op_declare_tf215.so")
    )

fused_multihead_attention = custom_tf.fused_multihead_attention
fused_multihead_attention_with_bias = custom_tf.fused_multihead_attention_with_bias
fuse_slot_gate = custom_tf.fused_slot_gate
fuse_gated_sum = custom_tf.fused_gated_sum
fused_layer_norm = custom_tf.fused_layer_norm

mixture_of_experts = custom_tf.mixture_of_experts
radix_top_k = custom_tf.radix_top_k
