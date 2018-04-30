from registry.gitlab.com/datadrivendiscovery/images/base:ubuntu-artful-python36
maintainer Simon Heath <sheath@andrew.cmu.edu>

user root

run pip3 install --upgrade pip
run pip3 install --process-dependency-links git+https://gitlab.com/datadrivendiscovery/primitive-interfaces.git
run pip3 install --process-dependency-links git+https://gitlab.com/datadrivendiscovery/d3m.git
run pip3 install docker grpcio-tools grpcio celery sphinx
run pip3 install http://download.pytorch.org/whl/cpu/torch-0.3.1-cp36-cp36m-linux_x86_64.whl
run pip3 install torchvision

expose 45042


run mkdir /d3m
add src/ /d3m/src
add primitives_repo /d3m/primitives_repo
add test_output /d3m/test_output

cmd /d3m/src/main.py
