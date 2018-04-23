# keras-text-classification-binary
Binary text classification using keras.

Using open source databases.

Twitter sentiment analysis: http://thinknook.com/twitter-sentiment-analysis-training-corpus-dataset-2012-09-22/ 
Based on: https://github.com/keras-team/keras/blob/master/examples/reuters_mlp.py

Installing the Model Server:
https://www.tensorflow.org/serving/setup#installing_the_modelserver

Doc for Tensorflow Serving:  
https://www.tensorflow.org/serving/serving_basic#load_exported_model_with_standard_tensorflow_modelserver  
I used a locally compiled ModelServer:  
bazel build -c opt //tensorflow_serving/model_servers:tensorflow_model_server  
To run the server:  
bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server --port=9000 --model_name=binary_text_classifier --model_base_path=/tmp/model

Contents of /tmp/model:  
srinathsarma@srinathspad:~/my_projects/keras-text-classification-binary$ ls /tmp/model  
0001
srinathsarma@srinathspad:~/my_projects/keras-text-classification-binary$ ls /tmp/model/0001/  
saved_model.pb  variables

