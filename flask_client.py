"""
Flask client for binary text classifier demo.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags

import sys
import json
import flask
import numpy as np

import keras.preprocessing.text as kpt
from keras.preprocessing.text import Tokenizer
from grpc.beta import implementations
import tensorflow as tf

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2

app = flask.Flask(__name__)

FLAGS = flags.FLAGS

# we're still going to use a Tokenizer here, but we don't need to fit it
tokenizer = Tokenizer(num_words=3000)
# for human-friendly printing
labels = ['negative', 'positive']

# read in our saved dictionary
with open('dictionary.json', 'r') as dictionary_file:
  dictionary = json.load(dictionary_file)

# this utility makes sure that all the words in your input
# are registered in the dictionary
# before trying to turn them into a matrix.
def convert_text_to_index_array(text):
  words = kpt.text_to_word_sequence(text)
  wordIndices = []
  for word in words:
    if word in dictionary:
      wordIndices.append(dictionary[word])
    else:
      continue
      # print("'%s' not in training corpus; ignoring." %(word))
  return wordIndices


@app.route('/', methods=["GET", "POST"])
def index():
  return flask.render_template('index.html')

@app.route('/predict', methods=["POST"])
def predict():
  input_text = flask.request.form['test_str']
  app.logger.debug(input_text)
  # print >> sys.stderr, input_text
  channel = implementations.insecure_channel('0.0.0.0', 9000)
  stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

  testArr = convert_text_to_index_array(input_text)
  seq = np.zeros((1,3000), dtype=np.float32)
  sequence = tokenizer.sequences_to_matrix([testArr], mode='binary')
  seq[0] = sequence[0]

  request = predict_pb2.PredictRequest()
  request.model_spec.name = 'binary_text_classifier'
  request.model_spec.signature_name = 'predict'

  request.inputs['images'].CopyFrom(tf.contrib.util.make_tensor_proto(seq, shape=[1, 3000]))
  result = stub.Predict(request, 60.0)
  pred = result.outputs['scores'].float_val
  # print >> sys.stderr, pred
  return flask.render_template('index.html', input_text=input_text, sentiment=labels[np.argmax(pred)], confidence=pred[np.argmax(pred)] * 100)



# if __name__ == '__main__':
  # app.run(main)
#   app.run(host='0.0.0.0', port=8080)
