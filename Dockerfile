# JPL base image
FROM registry.datadrivendiscovery.org/jpl/docker_images/complete:ubuntu-bionic-python36-v2019.4.4

maintainer "Donghan Wang<donghanw@cs.cmu.edu>"

user root

# libcurl4-openssl-dev for pycurl
# fortran for bayesian_optimization
# python3-tk for d3m.index
RUN sudo apt-get update && apt-get install -y \
    libcurl4-openssl-dev \
    gfortran \
    python3-tk

## install d3m and grpc, a D3M dependency
RUN pip3 install --upgrade pip \
    && python3 -m pip install --upgrade grpcio grpcio-tools

# Create static dir for Image weights file
RUN mkdir /static
COPY resnet50_weights_tf_dim_ordering_tf_kernels.h5 /static

# Install bayesian_optimiaztion
COPY bayesian_optimization /tmp/bayesian_optimization
RUN cd /tmp/bayesian_optimization/bo/utils/direct_fortran; \
    bash make_direct.sh; \
    cd /tmp/bayesian_optimization; \
    pip3 install --upgrade pip \
    && python3 setup.py bdist_wheel \
    && pip3 install ./dist/bo*.whl

EXPOSE 45042

RUN mkdir /d3m
ADD src/ /d3m/src

CMD /d3m/src/main.py ${D3MRUN}
