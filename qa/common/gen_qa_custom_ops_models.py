# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import argparse
from builtins import range
import os
import sys

FLAGS = None

def create_zeroout_modelfile(create_savedmodel, models_dir, model_version):
    # Load the zero-out custom operator
    _zero_out_module = tf.load_op_library(os.path.join(FLAGS.zero_out_lib_path))
    zero_out = _zero_out_module.zero_out

    # Create the model that uses custom operator.
    tf.reset_default_graph()
    zin = tf.placeholder(tf.int32, [ None, ], "to_zero")
    zout = zero_out(zin, name="zeroed")

    model_name = "savedmodel_zeroout" if create_savedmodel else "graphdef_zeroout"
    model_version_dir = models_dir + "/" + model_name + "/" + str(model_version)

    try:
        os.makedirs(model_version_dir)
    except OSError as ex:
        pass # ignore existing dir

    if create_savedmodel:
        with tf.Session() as sess:
            input_name = "to_zero"
            output_name = "zeroed"
            input_tensor = tf.get_default_graph().get_tensor_by_name(input_name + ":0")
            output_tensor = tf.get_default_graph().get_tensor_by_name(output_name + ":0")
            input_dict = dict()
            output_dict = dict()
            input_dict[input_name] = input_tensor
            output_dict[output_name] = output_tensor
            tf.saved_model.simple_save(sess, model_version_dir + "/model.savedmodel",
                                       inputs=input_dict, outputs=output_dict)
    else:
        with tf.Session() as sess:
            graph_io.write_graph(sess.graph.as_graph_def(), model_version_dir,
                                 "model.graphdef", as_text=False)

def create_zeroout_modelconfig(create_savedmodel, models_dir, model_version):
    model_name = "savedmodel_zeroout" if create_savedmodel else "graphdef_zeroout"
    config_dir = models_dir + "/" + model_name
    config = '''
name: "{}"
platform: "{}"
max_batch_size: 0
input [
  {{
    name: "to_zero"
    data_type: TYPE_INT32
    dims: [ -1 ]
  }}
]
output [
  {{
    name: "zeroed"
    data_type: TYPE_INT32
    dims: [ -1 ]
  }}
]
'''.format(model_name,
           "tensorflow_savedmodel" if create_savedmodel else "tensorflow_graphdef")

    try:
        os.makedirs(config_dir)
    except OSError as ex:
        pass # ignore existing dir

    with open(config_dir + "/config.pbtxt", "w") as cfile:
        cfile.write(config)

def create_cudaop_modelfile(create_savedmodel, models_dir, model_version):
    # Load the add_one custom operator
    _cuda_op_module = tf.load_op_library(os.path.join(FLAGS.cuda_op_lib_path))
    add_one = _cuda_op_module.add_one

    # Create the model that uses custom operator.
    tf.reset_default_graph()
    zin = tf.placeholder(tf.int32, [ None, ], "in")
    zout = add_one(zin, name="out")

    model_name = "savedmodel_cudaop" if create_savedmodel else "graphdef_cudaop"
    model_version_dir = models_dir + "/" + model_name + "/" + str(model_version)

    try:
        os.makedirs(model_version_dir)
    except OSError as ex:
        pass # ignore existing dir

    if create_savedmodel:
        with tf.Session() as sess:
            input_name = "in"
            output_name = "out"
            input_tensor = tf.get_default_graph().get_tensor_by_name(input_name + ":0")
            output_tensor = tf.get_default_graph().get_tensor_by_name(output_name + ":0")
            input_dict = dict()
            output_dict = dict()
            input_dict[input_name] = input_tensor
            output_dict[output_name] = output_tensor
            tf.saved_model.simple_save(sess, model_version_dir + "/model.savedmodel",
                                       inputs=input_dict, outputs=output_dict)
    else:
        with tf.Session() as sess:
            graph_io.write_graph(sess.graph.as_graph_def(), model_version_dir,
                                 "model.graphdef", as_text=False)

def create_cudaop_modelconfig(create_savedmodel, models_dir, model_version):
    model_name = "savedmodel_cudaop" if create_savedmodel else "graphdef_cudaop"
    config_dir = models_dir + "/" + model_name
    config = '''
name: "{}"
platform: "{}"
max_batch_size: 0
input [
  {{
    name: "in"
    data_type: TYPE_INT32
    dims: [ -1 ]
  }}
]
output [
  {{
    name: "out"
    data_type: TYPE_INT32
    dims: [ -1 ]
  }}
]
'''.format(model_name,
           "tensorflow_savedmodel" if create_savedmodel else "tensorflow_graphdef")

    try:
        os.makedirs(config_dir)
    except OSError as ex:
        pass # ignore existing dir

    with open(config_dir + "/config.pbtxt", "w") as cfile:
        cfile.write(config)


def create_zero_out_models(models_dir):
    model_version = 1

    if FLAGS.graphdef:
        create_zeroout_modelconfig(False, models_dir, model_version)
        create_zeroout_modelfile(False, models_dir, model_version)

    if FLAGS.savedmodel:
        create_zeroout_modelconfig(True, models_dir, model_version)
        create_zeroout_modelfile(True, models_dir, model_version)

def create_cuda_op_models(models_dir):
    model_version = 1

    if FLAGS.graphdef:
        create_cudaop_modelconfig(False, models_dir, model_version)
        create_cudaop_modelfile(False, models_dir, model_version)

    if FLAGS.savedmodel:
        create_cudaop_modelconfig(True, models_dir, model_version)
        create_cudaop_modelfile(True, models_dir, model_version)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--models_dir', type=str, required=True,
                        help='Top-level model directory')
    parser.add_argument('--zero_out_lib_path', type=str, required=False,
                        default="./libzeroout.so",
                        help='Fullpath to libzeroout.so')
    parser.add_argument('--cuda_op_lib_path', type=str, required=False,
                        default="./libcudaop.so",
                        help='Fullpath to libcudaop.so')
    parser.add_argument('--graphdef', required=False, action='store_true',
                        help='Generate GraphDef models')
    parser.add_argument('--savedmodel', required=False, action='store_true',
                        help='Generate SavedModel models')
    FLAGS, unparsed = parser.parse_known_args()

    if FLAGS.graphdef or FLAGS.savedmodel:
        import tensorflow as tf
        from tensorflow.python.framework import graph_io, graph_util

    create_zero_out_models(FLAGS.models_dir)
    create_cuda_op_models(FLAGS.models_dir)
