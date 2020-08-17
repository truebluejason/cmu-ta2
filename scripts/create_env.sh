#!/usr/bin/env bash

# NOTE: Source-enable the latest /opt/rh/devtools first

# We assume we're running from the ta2/scripts dir. Cd to ta2.
cd ..

conda deactivate

# Remove the d3m directory if it already exists (prompt user)
while true; do
    read -p "Must delete d3m and ta3ta2-apifolder (if they exist). Ok to proceed? " yn
    case $yn in
        [Yy]* ) rm -rf d3m; rm -rf ta3ta2-api; break;;
        [Nn]* ) return;;
        * ) echo "Please answer yes or no (y or n).";;
    esac
done

# Remove the existing d3m environment (prompt user)
while true; do
    read -p "Delete and recreate 'd3m' conda environment (if no, existing 'd3m' conda environment will be updated)? " yn
    case $yn in
        [Yy]* ) conda env remove -n d3m --yes; conda create -y --name d3m python=3.6 libcurl; break;;
        [Nn]* ) break;;
        * ) echo "Please answer yes or no.";;
    esac
done

# Activate the new conda d3m environment
conda activate d3m

# For command-line json parsing (used in test.sh)
conda install -c conda-forge jq

# Install PIP dependencies
pip install docker grpcio-tools grpcio d3m sri-d3m

# Here, we start installing primitives. Update these links with the
# installation->package_uri value from the primitive.json file inside any one
# of the primitives inside the organization folder (e.g. ISI, Distil, etc.) of
# the latest release folder (e.g. v2020.1.9).

# Create the d3m directory where the primitives will be installed.
mkdir d3m
cd d3m

# common_primitives
pip install -U -e git+https://gitlab.com/datadrivendiscovery/common-primitives.git@807e6a5e353b41c37d3da13f7fd81386aee9eb07#egg=common_primitives

# JPL/SKLearn
pip install -U -e git+https://gitlab.com/datadrivendiscovery/sklearn-wrap.git@1c12b1b9e6d138a1936389a892b02a98db9f4691#egg=sklearn_wrap

# Other primitives to install...
#ISI
pip install -U -e git+https://github.com/usc-isi-i2/dsbox-primitives@07ba9325e47b79b2fabef5a53c38cee25b28ee5d#egg=dsbox-primitives
#Distil
pip install -U -e git+https://github.com/uncharted-distil/distil-primitives.git@487f4d91434f5bf938faae4f03ef09a76c2878fa#egg=distil-primitives
pip install -U -e git+https://github.com/kungfuai/d3m-primitives.git@67e436e383c59dccb63bf683edb6f6a122b5c1e7#egg=kf-d3m-primitives
#RPI
pip install -U rpi_d3m_primitives rpi_d3m_primitives_part2
#SRI
pip install -U sri-d3m
#JHU
pip install -U -e git+https://github.com/neurodata/primitives-interfaces.git@f6f5fcb577804231037e8a4f2d83638532212d14#egg=jhu_primitives
#CMU
pip install -U -e git+https://github.com/autonlab/fastlvm.git@aa52ced29537aa545d91c3f7df90608756981ff4#egg=fastlvm
pip install -U -e git+https://github.com/autonlab/find_projections.git@6bd85718e86632c891f4eb872088d02c18d3e5e0#egg=find_projections
pip install -U -e git+https://github.com/autonlab/autonbox.git@83a048b3b30f8a3e17479d5d05cfaa569fc8e843#egg=autonbox
pip install -U -e git+https://github.com/autonlab/esrnn.git@49d18e545f1605ca6c5ce71277c51463581bd382#egg=esrnn

# We are done with the primitives installation, so leave the d3m folder
cd ..

# Clone CMU TA2 repo
# NOTE: Uncomment this line to also clone the CMU TA2 repo.
# git clone git@gitlab.datadrivendiscovery.org:sheath/cmu-ta2.git

# Clone ta3ta2-api repo
git clone https://gitlab.com/datadrivendiscovery/ta3ta2-api.git

# Switch to the latest stable tag (update this to the latest dated tag you see in the repo)
cd ta3ta2-apic
git checkout -b v2020.6.2
cd ..

# Rebuild protocol buffers
cd ta2
./rebuild_grpc.sh
cd ..