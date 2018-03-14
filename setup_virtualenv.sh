#!/bin/sh

# Not necessarily functional since getting the right version of python3.6 on CentOS is a PITA
# and there appear to be multiple options.  x_X

virtualenv-3 env --python=python36
source env/bin/activate
pip install --upgrade pip
pip install --process-dependency-links git+https://gitlab.com/datadrivendiscovery/primitive-interfaces.git
pip install docker grpcio-tools grpcio d3m