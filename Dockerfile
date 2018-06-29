# JPL base image
from registry.datadrivendiscovery.org/jpl/docker_images/complete:ubuntu-artful-python36-v2018.6.5-20180626-120232

maintainer "Donghan Wang<donghanw@cs.cmu.edu>, Simon Heath <sheath@andrew.cmu.edu>"

user root

# libcurl4-openssl-dev for pycurl
# fortran for bayesian_optimization
RUN apt-get update && apt-get install -y \
    libcurl4-openssl-dev \
    gfortran

# install d3m and grpc, a D3M dependency
RUN pip3 install --upgrade pip \
    && python3 -m pip install --process-dependency-links d3m \
    && python3 -m pip install --upgrade grpcio grpcio-tools

# Install bayesian_optimiaztion
COPY bayesian_optimization /tmp/bayesian_optimization
RUN cd /tmp/bayesian_optimization/bo/utils/direct_fortran; \
    bash make_direct.sh; \
    cd /tmp/bayesian_optimization; \
    python3 setup.py bdist_wheel; \
    pip3 install ./dist/bo*.whl

# Install old sklearn_wrap
RUN pip3 install -U -e git+https://gitlab.com/datadrivendiscovery/sklearn-wrap.git@8d8a53ec71dff5906c4e3f206967a6ec63b66bab#egg=sklearn_wrap

expose 45042

run mkdir /d3m
add src/ /d3m/src
#add primitives_repo /d3m/primitives_repo
#add test_output /d3m/test_output

cmd /d3m/src/main.py ${D3MRUN}
