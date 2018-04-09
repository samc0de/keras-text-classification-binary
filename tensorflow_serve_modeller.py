import json
import tensorflow as tf
from tensorflow.contrib.session_bundle import exporter
from keras.models import model_from_json

export_path = 'models/tensorflow_serve/'
export_version = 0

def main():
  with open('model.json', 'r') as json_file:
    loaded_model_json = json_file.read()

  # and create a model from that
  model = model_from_json(loaded_model_json)
  # and weight your nodes with your saved values
  model.load_weights('model.h5')

  saver = tf.train.Saver(sharded=True)
  model_exporter = exporter.Exporter(saver)
  signature = exporter.classification_signature(input_tensor=model.input,
                                                scores_tensor=model.output)
  with tf.Session() as sess:
    model_exporter.init(sess.graph.as_graph_def(),
                        default_graph_signature=signature)
    model_exporter.export(export_path, tf.constant(export_version), sess)


if __name__ == '__main__':
  main()
