import json

import tensorflow as tf
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import utils
from tensorflow.python.saved_model import tag_constants, signature_constants
from tensorflow.python.saved_model.signature_def_utils_impl import build_signature_def, predict_signature_def
from tensorflow.contrib.session_bundle import exporter

import keras
from keras.models import model_from_json

export_path = 'models/tensorflow_serve/exported'
export_version = 0

def main():
  with open('model.json', 'r') as json_file:
    loaded_model_json = json_file.read()

  # and create a model from that
  model = model_from_json(loaded_model_json)
  # and weight your nodes with your saved values
  model.load_weights('model.h5')

#   saver = tf.train.Saver(sharded=True)
#   model_exporter = exporter.Exporter(saver)
#   signature = exporter.classification_signature(input_tensor=model.input,
#                                                 scores_tensor=model.output)
#   with keras.backend.get_session() as sess:
#     model_exporter.init(sess.graph.as_graph_def(),
#                         default_graph_signature=signature)
#     model_exporter.export(export_path, tf.constant(export_version), sess)
#

  builder = saved_model_builder.SavedModelBuilder(export_path)

  signature = predict_signature_def(inputs={'images': model.input},
                                    outputs={'scores': model.output})

  with keras.backend.get_session() as sess:
      builder.add_meta_graph_and_variables(
          sess=sess, tags=[tag_constants.SERVING], signature_def_map={
              'predict': signature})
      builder.save()

if __name__ == '__main__':
  main()
