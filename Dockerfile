# JPL base image
from registry.datadrivendiscovery.org/jpl/docker_images/complete:ubuntu-artful-python36-v2018.6.5-20180620-120449

maintainer "Donghan Wang<donghanw@cs.cmu.edu>, Simon Heath <sheath@andrew.cmu.edu>"

user root

# libcurl4-openssl-dev for pycurl
# fortran for bayesian_optimization
RUN apt-get update && apt-get install -y \
    libcurl4-openssl-dev \
    gfortran

# install d3m and grpc, a D3M dependency
RUN pip3 install --upgrade pip \
    && pip3 install --process-dependency-links git+https://gitlab.com/datadrivendiscovery/primitive-interfaces.git \
    && pip3 install --process-dependency-links git+https://gitlab.com/datadrivendiscovery/d3m.git \
    && pip3 install --upgrade grpcio grpcio-tools

# Install bayesian_optimiaztion
COPY bayesian_optimization /tmp/bayesian_optimization
RUN cd /tmp/bayesian_optimization/bo/utils/direct_fortran; \
    bash make_direct.sh; \
    cd /tmp/bayesian_optimization; \
    python3 setup.py bdist_wheel; \
    pip3 install ./dist/bo*.whl

expose 45042

run mkdir /d3m
add src/ /d3m/src
#add primitives_repo /d3m/primitives_repo
#add test_output /d3m/test_output

cmd /d3m/src/main.py
