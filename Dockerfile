# JPL base image
#FROM registry.datadrivendiscovery.org/jpl/docker_images/complete:ubuntu-bionic-python36-v2019.4.4-20190507-022543
#FROM registry.datadrivendiscovery.org/jpl/docker_images/complete:ubuntu-bionic-python36-v2019.5.8-20190602-081936
#FROM registry.datadrivendiscovery.org/jpl/docker_images/complete:ubuntu-bionic-python36-v2019.6.7-20190611-060317
#FROM registry.datadrivendiscovery.org/jpl/docker_images/complete:ubuntu-bionic-python36-v2019.6.7-20190611-205824
#FROM registry.datadrivendiscovery.org/jpl/docker_images/complete:ubuntu-bionic-python36-v2019.6.7
#FROM registry.datadrivendiscovery.org/jpl/docker_images/complete:ubuntu-bionic-python36-v2019.6.7-20190622-073225
#FROM registry.gitlab.com/datadrivendiscovery/images/primitives:ubuntu-bionic-python36-v2019.11.10-20191127-050901
FROM registry.gitlab.com/datadrivendiscovery/images/primitives:ubuntu-bionic-python36-v2020.1.9-20200125-073913

maintainer "Donghan Wang<donghanw@cs.cmu.edu>"

user root

# add git-lfs gpg and return exit code 0
RUN apt-get update || (apt-get install dirmngr && apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 762E3157)

# libcurl4-openssl-dev for pycurl
# fortran for bayesian_optimization
# python3-tk for d3m.index
RUN sudo apt-get update && apt-get install -y \
    libcurl4-openssl-dev \
    gfortran \
    python3-tk

## install d3m and grpc, a D3M dependency
##
## We use pip==18.1 because pip 19+ removed --process-dependency-links
##
RUN pip3 install --upgrade pip==18.1 \
#    && python3 -m pip install --process-dependency-links d3m \
    && python3 -m pip install --upgrade grpcio grpcio-tools

# Create static dir for Image weights file
#RUN mkdir /static
#COPY resnet50_weights_tf_dim_ordering_tf_kernels.h5 /static

EXPOSE 45042

RUN mkdir /d3m
ADD src/ /d3m/src

CMD /d3m/src/main.py ${D3MRUN}
