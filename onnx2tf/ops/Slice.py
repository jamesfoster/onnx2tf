import random
random.seed(0)
import numpy as np
np.random.seed(0)
import tensorflow as tf
from tensorflow.python.keras.layers import Lambda
import onnx_graphsurgeon as gs
from utils.common_functions import (
    get_constant_or_variable,
    print_node_info,
    inverted_operation_enable_disable,
)
from utils.enums import NUMPY_DTYPES_TO_TF_DTYPES


@print_node_info
@inverted_operation_enable_disable
def make_node(
    *,
    graph_node: gs.Node,
    tf_layers_dict: dict,
    **kwargs: dict,
):
    """Slice

    Parameters
    ----------
    graph_node: gs.Node
        graph_surgeon Node

    tf_layers_dict: dict
        optype, shape, dtype, tensorflow graph
    """
    before_op_output_shape_trans_1 = \
        tf_layers_dict.get(graph_node.inputs[0].name, {}).get('before_op_output_shape_trans', True)
    before_op_output_shape_trans_2 = \
        tf_layers_dict.get(graph_node.inputs[1].name, {}).get('before_op_output_shape_trans', True)
    before_op_output_shape_trans_3 = \
        tf_layers_dict.get(graph_node.inputs[2].name, {}).get('before_op_output_shape_trans', True)
    before_op_output_shape_trans_4 = \
        tf_layers_dict.get(graph_node.inputs[3].name, {}).get('before_op_output_shape_trans', True)
    before_op_output_shape_trans_5 = \
        tf_layers_dict.get(graph_node.inputs[4].name, {}).get('before_op_output_shape_trans', True)
    before_op_output_shape_trans = \
        before_op_output_shape_trans_1 \
        and before_op_output_shape_trans_2 \
        and before_op_output_shape_trans_3 \
        and before_op_output_shape_trans_4 \
        and before_op_output_shape_trans_5

    input_tensor = get_constant_or_variable(
        graph_node.inputs[0],
        before_op_output_shape_trans,
    )
    input_tensor = tf_layers_dict[input_tensor.name]['tf_node'] \
        if isinstance(input_tensor, gs.Variable) else input_tensor

    starts = get_constant_or_variable(
        graph_node.inputs[1],
        before_op_output_shape_trans,
    )
    starts = tf_layers_dict[starts.name]['tf_node'] \
        if isinstance(starts, gs.Variable) else starts
    # if isinstance(starts, np.ndarray):
    #     starts = tf.constant(starts, dtype=NUMPY_DTYPES_TO_TF_DTYPES[starts.dtype])

    ends = get_constant_or_variable(
        graph_node.inputs[2],
        before_op_output_shape_trans,
    )
    ends = tf_layers_dict[ends.name]['tf_node'] \
        if isinstance(ends, gs.Variable) else ends
    # if isinstance(ends, np.ndarray):
    #     ends = tf.constant(ends, dtype=NUMPY_DTYPES_TO_TF_DTYPES[ends.dtype])

    # input_tensor_shape = tf.shape(
    #     input=input_tensor,
    #     out_type=ends.dtype,
    # )
    # input_tensor_rank = tf.rank(input_tensor)

    axes = None
    if len(graph_node.inputs) >= 4:
        axes = get_constant_or_variable(
            graph_node.inputs[3],
            before_op_output_shape_trans,
        )
    axes = tf_layers_dict[axes.name]['tf_node'] \
        if isinstance(axes, gs.Variable) else axes
    if isinstance(axes, np.ndarray):
        axes = tf.constant(axes, dtype=NUMPY_DTYPES_TO_TF_DTYPES[axes.dtype]) \
            if len(graph_node.inputs) >= 4 else tf.range(tf.shape(starts)[0], dtype=ends.dtype)
    else:
        axes = axes if len(graph_node.inputs) >= 4 else tf.range(tf.shape(starts)[0], dtype=ends.dtype)

    steps = None
    if len(graph_node.inputs) >= 5:
        steps = get_constant_or_variable(
            graph_node.inputs[4],
            before_op_output_shape_trans,
        )
    steps = tf_layers_dict[steps.name]['tf_node'] \
        if isinstance(steps, gs.Variable) else steps
    if isinstance(steps, np.ndarray):
        steps = tf.constant(steps, dtype=NUMPY_DTYPES_TO_TF_DTYPES[steps.dtype])

    graph_node_output: gs.Variable = graph_node.outputs[0]
    shape = graph_node_output.shape
    dtype = graph_node_output.dtype



    # Preserving Graph Structure (Dict)
    tf_layers_dict[graph_node_output.name] = {
        'optype': graph_node.op,
        'shape': shape,
        'dtype': dtype,
    }

    # Generation of TF OP
    slice_len = len(starts)

    if slice_len == 1:
        # Shape output as int64 since the spec implicitly allows int64
        full_sizes = np.asarray(input_tensor.shape, dtype=np.int64)

        updated_full_sizes = [0] * len(input_tensor.get_shape())
        updated_full_begin = [0] * len(input_tensor.get_shape())
        updated_starts = [0] * slice_len
        updated_ends = [0] * slice_len

        for axis in range(input_tensor.shape.rank):
            if axis not in axes:
                # Update the sizes for axes that are not in the axes attribute
                # No need to change the default of 0 in begins
                updated_full_sizes[axis] = full_sizes[axis]
            else:
                # Update the begins and sizes for each axis in the axes attribute
                for i in range(slice_len):
                    if axis == axes[i]:
                        updated_starts[i] = full_sizes[axis] + starts[i] if starts[i] < 0 else starts[i]
                        updated_ends[i] = full_sizes[axis] + ends[i] if ends[i] < 0 else ends[i]
                        if full_sizes[axis] is not None:
                            updated_ends[i] = min(full_sizes[axis], updated_ends[i])
                            updated_starts[i] = min(full_sizes[axis], updated_starts[i])

                        updated_full_begin[axis] = updated_starts[i]
                        updated_full_sizes[axis] = updated_ends[i]

        tf_layers_dict[graph_node_output.name]['tf_node'] = \
            tf.strided_slice(
                input_=input_tensor,
                begin=updated_full_begin,
                end=updated_full_sizes,
                name=graph_node.name,
            )
    else:
        tf_layers_dict[graph_node_output.name]['tf_node'] = \
            tf.strided_slice(
                input_=input_tensor,
                begin=starts,
                end=ends,
                strides=steps,
                name=graph_node.name,
            )
