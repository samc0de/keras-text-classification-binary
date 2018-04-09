FROM ubuntu:16.04
MAINTAINER Sameer Mahabole <sameer.mahabole@gmail.com>

RUN apt-get update && yes | apt-get upgrade

RUN apt-get install --yes python-pip

RUN pip install --upgrade pip

# Also support python3.
RUN pip install --trusted-host pypi.python.org -r requirements.txt

RUN set -e; \
        apt-get install --yes\
                vim \
                git \
        ;

