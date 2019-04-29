TA2 implementation for CMU

# Overview

Our job is to take requests for analysis from the TA3, turn it into an analysis pipeline using the available library of TA1 primitives, and run it.

TA1 is by python function classes and method calls, basically, defined by a JSON schema.  We have the `primitive-interfaces` repo which defines the interfaces.  Then we have the `primitives_repo` repo which contains the schemas.  Then the schemas have the install source and method for the primitive, generally through pip.

The TA3 requests take the form of a grpc protobuf call.  https://grpc.io/docs/quickstart/python.html has basics.

Goals: Pipeline and API, primitives, TA3 interface, Bayes optimization.


# Setup instructions

Setup virtualenv

```
# On Centos 7:
sudo yum install rh-python36-python rh-python36-python-pip rh-python36-python-virtualenv
source /opt/rh/rh-python36/enable

virtualenv env --python=python36
source env/bin/activate
pip install docker grpcio-tools grpcio d3m
```

# Related things

## Interfaces

 * https://gitlab.datadrivendiscovery.org/jpl/primitives_repo.git
 * https://gitlab.com/datadrivendiscovery/ta3ta2-api 

We start our TA2:

```
cd /home/sheath/projects/D3M/cmu-ta2
./src/main.py ta2ta3
```

Now we can hit "start session" in the thing and, lo and behold, it actually talks to our TA2!  Magic.

# Building the docker image
The repository automatically builds the docker image after each commit.
Images are stored in the [registry](https://gitlab.datadrivendiscovery.org/sheath/cmu-ta2/container_registry).

The Gitlab continuous integration (CI) manages the building process.
There are three configuration files:
1. ```.gitlab-ci.yml```
1. ```.gitmodules```
1. ```Dockerfile```

# Building the docker image manually
Prerequisites:
1. [Install docker](https://docs.docker.com/install/).
    1. Check docker ```docker --version```
1. ```git clone  git@gitlab.datadrivendiscovery.org:sray/bayesian_optimization.git```
1. The use of submodule is **deprecated** since submodule itself doesn’t update unless you manually update it.
    1. We keep ta3ta2-api-v2 for reference only
    1. ta3ta2-api-v2 is used in ```rebuild_grpc.sh```
        1. You probably want to create a local copy of the shell script in which it points to the latest version of the api repo instead.
    1. To clone into submodules, run following command at repository top level as a normal user
    ```bash
    git submodule sync --recursive
    git submodule update --init --recursive
    ```

## Create a docker image
Run all commands as root.
Create a docker image by running the following commands
```bash
docker build -t registry.datadrivendiscovery.org/sheath/cmu-ta2 . 
```
which builds an image with tag **latest**.

This creates an image--```cmu-ta2```--in your machine’s local Docker image registry.

You can build an image with a named tag
```bash
docker build -t registry.datadrivendiscovery.org/sheath/cmu-ta2:<tag> . 
```
where ```<tag>``` is the tag name.

List docker images:
```bash
docker image ls
```

Finally, push the image to the D3M registry
```bash
docker push registry.datadrivendiscovery.org/sheath/cmu-ta2
```

# Using the docker image
## Pull the docker image from the D3M registry
```bash
# pull the image with "live" tag
# used by summer 2018 eval
docker pull registry.datadrivendiscovery.org/sheath/cmu-ta2:live

# pull the "lasted" image 
docker pull registry.datadrivendiscovery.org/sheath/cmu-ta2:latest
```

## Run the docker image
The following commands map 
Run the docker image, mapping your machine’s port 45042 to the container’s published port 45042 using ```-p```

You can run TA2 image in one of three modes: ```search``` and ```ta2ta3```.
This is controlled by the environment variable ```D3MRUN```.
If you don't specify ```D3MRUN``` while ```docker run...```, it runs in ```ta2ta3``` mode.

### Run in ta2ta3 mode
```bash
docker run -i -t \
    --name d3m-cmu-ta2 \
    --mount type=bind,source=</path/to/seed_dataset/on/host>,target=/input \
    --mount type=bind,source=</path/to/output/on/host>,target=/output \
    -p 45042:45042  \
    -e D3MINPUTDIR="</path/to/dataset>"  \
    -e D3MOUTPUTDIR="</path/to/output_folder>" \
    -e D3MCPU=8 \
    -e D3MTIMEOUT=5 \
    -e D3MRUN="ta2ta3" \
    registry.datadrivendiscovery.org/sheath/cmu-ta2:live
```

Below is an example:
```bash
docker run -i -t \
    --rm \
    --name d3m-cmu-ta2 \
    -p 45042:45042 \
    --mount type=bind,source=/data/data/d3m/dryrun2018summer/input,target=/input \
    --mount type=bind,source=/data/data/d3m/dryrun2018summer/output,target=/output \
    -e D3MINPUTDIR=/input  \
    -e D3MOUTPUTDIR=/output  \
    -e D3MCPU=8 \
    -e D3MTIMEOUT=5 \
    -e D3MRUN="ta2ta3" \
    registry.datadrivendiscovery.org/sheath/cmu-ta2:live
```

### Run in search mode
```bash
docker run -i -t \
    --rm \
    -p 45042:45042  \
    --name d3m-cmu-ta2-search \
    --mount type=bind,source=</path/to/seed_dataset/on/host>,target=/input \
    --mount type=bind,source=</path/to/output/on/host>,target=/output \
    -e D3MINPUTDIR="</path/to/input_folder>"  \
    -e D3MOUTPUTDIR="</path/to/output_folder>" \
    -e D3MCPU=8 \
    -e D3MTIMEOUT=5 \
    -e D3MRUN="search" \
    registry.datadrivendiscovery.org/sheath/cmu-ta2:live
```

Below is an example:
```bash
docker run -i -t \
    --rm \
    --name d3m-cmu-ta2-search \
    -p 45042:45042 \
    --mount type=bind,source=/data/data/d3m/dryrun2018summer/input/LL0_1100_popularkids,target=/input \
    --mount type=bind,source=/data/data/d3m/dryrun2018summer/output,target=/output \
    -e D3MINPUTDIR=/input  \
    -e D3MOUTPUTDIR=/output  \
    -e D3MCPU=8 \
    -e D3MTIMEOUT=5 \
    -e D3MRUN="search" \
    registry.datadrivendiscovery.org/sheath/cmu-ta2:live
```

### Run in test mode
WIP

```bash
docker run -i -t \
    --rm \
    --name d3m-cmu-ta2-test \
    -p 45042:45042  \
    --mount type=bind,source=</path/to/seed_dataset/on/host>,target=/input \
    --mount type=bind,source=</path/to/output/on/host>,target=/output \
    -e D3MTESTOPT="</path/to/executable/file>"  \
    -e D3MINPUTDIR="</path/to/input_folder>"  \
    -e D3MOUTPUTDIR="</path/to/output_folder>" \
    -e D3MCPU=8 \
    -e D3MTIMEOUT=5 \
    -e D3MRUN="test" \
    registry.datadrivendiscovery.org/sheath/cmu-ta2:live
```

#### An example:
```bash
docker run -i -t \
    --rm \
    -p 45042:45042  \
    --name d3m-cmu-ta2-test \
    --mount type=bind,source=/input,target=/input \
    --mount type=bind,source=/output,target=/output \
    -e D3MTESTOPT="/output/executables/dcebb3c9-6889-41e2-82b9-8d2d2484d5d5_1.sh"  \
    -e D3MINPUTDIR="/input/185_baseball"  \
    -e D3MOUTPUTDIR="/output" \
    -e D3MCPU=8 \
    -e D3MTIMEOUT=5 \
    -e D3MRUN="test" \
    registry.datadrivendiscovery.org/sheath/cmu-ta2:live
```

Run the command to produce the score from the executable of pipeline that ranked #1
```bash
python evaluate_script.py ./185_baseball/SCORE/targets.csv ../output/predictions/dcebb3c9-6889-41e2-82b9-8d2d2484d5d5_1/predictions.csv Hall_of_Fame F1
```
