
FROM quay.io/fenicsproject/stable:2017.2.0

USER root

RUN sudo apt-get update && sudo apt-get -y install git python3-setuptools

RUN git clone https://github.com/akdiem/poroelastic.git
RUN cd poroelastic && sudo python3 setup.py install && cd ..
