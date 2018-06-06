from registry.datadrivendiscovery.org/jpl/docker_images/complete:ubuntu-xenial-python36-v2018.1.26-20180515-235715
maintainer "Donghan Wang<donghanw@cs.cmu.edu>, Simon Heath <sheath@andrew.cmu.edu>"

user root

RUN apt-get update
RUN apt-get install -y libcurl4-openssl-dev # for pycurl

run pip3 install --upgrade pip
run pip3 install --process-dependency-links git+https://gitlab.com/datadrivendiscovery/primitive-interfaces.git
run pip3 install --process-dependency-links git+https://gitlab.com/datadrivendiscovery/d3m.git

# Install grpc, a D3M dependency
RUN pip3 install --upgrade grpcio grpcio-tools

# Install fortran, a bayesian_optimization dependency
RUN apt-get install -y gfortran

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
add primitives_repo /d3m/primitives_repo
#add test_output /d3m/test_output

cmd /d3m/src/main.py
