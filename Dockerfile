# JPL base image
FROM registry.datadrivendiscovery.org/jpl/docker_images/complete:ubuntu-artful-python36-v2018.7.10-20180801-215033

maintainer "Donghan Wang<donghanw@cs.cmu.edu>, Simon Heath <sheath@andrew.cmu.edu>"

user root

# Silently resolve the GPG error so the CI doesn't fail
# The error is caused by
# W: GPG error: https://packagecloud.io/github/git-lfs/ubuntu artful InRelease: The following signatures couldn't be verified because the public key is not available: NO_PUBKEY 6B05F25D762E3157
RUN sudo 2>/dev/null 1>/dev/null apt-get update
# add git-lfs gpg
RUN sudo apt-get install dirmngr && \
    apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 762E3157

# libcurl4-openssl-dev for pycurl
# fortran for bayesian_optimization
# python3-tk for d3m.index
RUN sudo apt-get update && apt-get install -y \
    libcurl4-openssl-dev \
    gfortran \
    python3-tk

# install d3m and grpc, a D3M dependency
RUN pip3 install --upgrade pip \
    && python3 -m pip install --process-dependency-links d3m \
    && python3 -m pip install --upgrade grpcio grpcio-tools

RUN pip3 install -U pandas==0.22.0

# Install bayesian_optimiaztion
COPY bayesian_optimization /tmp/bayesian_optimization
RUN cd /tmp/bayesian_optimization/bo/utils/direct_fortran; \
    bash make_direct.sh; \
    cd /tmp/bayesian_optimization; \
    python3 setup.py bdist_wheel; \
    pip3 install ./dist/bo*.whl

EXPOSE 45042

RUN mkdir /d3m
ADD src/ /d3m/src

CMD /d3m/src/main.py ${D3MRUN}
