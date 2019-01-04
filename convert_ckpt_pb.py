#-*- coding:utf:8 -*-

import tensorflow as tf

flags = tf.app.flags

flags.DEFINE_string("meta_path",None,"Path to the meta file")
flags.DEFINE_string("output_pb",None,"Path to the meta file")
flags.DEFINE_string("checkpoint_dir",None,"Path to the meta file")
flags.DEFINE_string("graph_proto",None,"Path to the meta file")

FLAGS = flags.FLAGS

def main(_):
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(FLAGS.meta_path)
        saver.restore(sess, tf.train.latest_checkpoint(FLAGS.checkpoint_dir))
        with open(FLAGS.graph_proto) as f:
            graph = tf.get_default_graph().as_graph_def(add_shape=True)
            f.write(graph.SerializeToString())
        init = tf.global_variables_initializer()
        sess.run(init)
        output_node_name = [n.name for n in tf.get_default_graph().as_graph_def().node]
        frozen_graph_def = tf.graph_util.convert_variable_to_constants(sess, sess.graph_def, output_node_name)
        with open(FLAGS.output_pb) as f:
            f.write(frozen_graph_def.SeriablizeToString())


if __name__ == "__main__":
    tf.app.run()

