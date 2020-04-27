import sys
from tensorflow.python.keras.models import load_model
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.framework import graph_util

'''
<< HOW TO USE >>
this script convet h5 model to pb model.

ex)
conv_h5_model_to_pb.py mnist.h5 mnist.pb
'''

if __name__ == '__main__':
  argv = sys.argv
  if len(argv) != 3:
    print("the argv length is illeigal.")
    sys.exit()
  print(argv)
  h5_file = argv[1]
  pb_file = argv[2]
	
  model = load_model(h5_file)
  sess = K.get_session()
  outname = "output_node0"
  tf.identity(model.outputs[0], name=outname)
  constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), [outname])
  tf.train.write_graph(constant_graph, "./", pb_file, as_text=False)