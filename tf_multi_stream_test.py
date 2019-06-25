from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import sys
import time

import tensorflow as tf
import numpy as np
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.framework import graph_pb2


from threading import Thread, Barrier

def inference(config, gpu_num, model_path, barrier, num_iters=100):
    with tf.Session(graph=tf.Graph(), config=config) as sess:
        with tf.device("/gpu:%d"%gpu_num):
            
            # Import a graph def model
            graph_def = graph_pb2.GraphDef()
            with open(model_path, 'rb') as f:
                graph_def.ParseFromString(f.read());
            tf.import_graph_def(graph_def)
            graph = tf.get_default_graph()

            # Get tensor handles for inception graph
            input_tensor = graph.get_tensor_by_name("import/input:0")
            output_tensor = graph.get_tensor_by_name('import/InceptionV3/Predictions/Reshape_1:0')
            
            # Collect threads
            barrier.wait()

            # Run iterations of inference
            for i in range(num_iters):   
                sess.run(output_tensor, feed_dict={input_tensor:np.zeros(input_tensor.shape)})

	       
if __name__ == '__main__':

    if (len(sys.argv) < 4):
        print("Usage python test.py <graph_def_path> <num_instances> <vGPU [ON|OFF]>")
        sys.exit(1)

    # Initialize arguments to sessions
    num_instances = int(sys.argv[2])
    model_path = sys.argv[1]
    batch_size = int(sys.argv[4])
    barrier = Barrier(num_instances)
    threads = []

    # Create config
    if sys.argv[3] == "ON":
        gpu_options = config_pb2.GPUOptions(
            visible_device_list='0',
            experimental=config_pb2.GPUOptions.Experimental(virtual_devices=[
                config_pb2.GPUOptions.Experimental.VirtualDevices(
                    memory_limit_mb=[10000/num_instances]*num_instances)]) )   
        
        config = config_pb2.ConfigProto(gpu_options=gpu_options)
        gpus = range(num_instances)
    else:
        config = config_pb2.ConfigProto()
        gpus = [0]*num_instances

    # start threads
    for i in gpus:
        threads.append(Thread(target=inference, args=(config, i, model_path, barrier)))
        threads[-1].start()

    # join threads
    for t in threads:
        t.join()






    
