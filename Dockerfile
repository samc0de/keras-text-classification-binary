FROM ubuntu:16.04
MAINTAINER Sameer Mahabole <sameer.mahabole@gmail.com>

RUN apt-get update && yes | apt-get upgrade && apt-get install --yes python-pip

RUN pip install --upgrade pip

ADD . /text-classification

# Also support python3.
RUN pip install --trusted-host pypi.python.org -r /text-classification/requirements.txt

RUN set -e; \
        apt-get install --yes\
                vim \
                git \
        ;

# For exporting models to tensorflow-serving.
RUN pip install tensorflow-serving-api


# Maybe make one repo for jupyter notebooks, clone these model repos there and
# serve them.

RUN jupyter notebook --generate-config --allow-root
RUN echo "c.NotebookApp.password = u'sha1:6a3f528eec40:6e896b6e4828f525a6e20e5411cd1c8075d68619'" >> /root/.jupyter/jupyter_notebook_config.py

EXPOSE 8888
CMD ["jupyter", "notebook", "--allow-root", "--notebook-dir=/text-classification", "--ip='*'", "--port=8888", "--no-browser"]

# Run like:
# docker build -t binary-text-classification .
# docker run -it --name binary_classification -p 9999:8888 binary-text-classification
