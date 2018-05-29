TA2 implementation for CMU

# Overview

Our job is to take requests for analysis from the TA3, turn it into an analysis pipeline using the available library of TA1 primitives, and run it.

TA1 is by python function classes and method calls, basically, defined by a JSON schema.  We have the `primitive-interfaces` repo which defines the interfaces.  Then we have the `primitives_repo` repo which contains the schemas.  Then the schemas have the install source and method for the primitive, generally through pip.

The TA3 requests take the form of a grpc protobuf call.  https://grpc.io/docs/quickstart/python.html has basics.  Matthias's TA3 includes a little server to translate JSON<->gRPC so his stuff can just speak JSON for now.  

Easy primitives for testing would be the JPL repo, which just wraps sklearn.  


Goals: Pipeline and API, primitives, TA3 interface, Bayes optimization.


# Setup instructions

Setup virtualenv

```
# On Centos 7:
sudo yum install rh-python36-python rh-python36-python-pip rh-python36-python-virtualenv
source /opt/rh/rh-python36/enable

virtualenv env --python=python36
source env/bin/activate
pip install --upgrade pip
pip install --process-dependency-links git+https://gitlab.com/datadrivendiscovery/primitive-interfaces.git
pip install docker grpcio-tools grpcio d3m
```



# Related things

## Interfaces

 * https://gitlab.com/datadrivendiscovery/primitive-interfaces
  * https://gitlab.com/datadrivendiscovery/tests-data.git
 * https://gitlab.datadrivendiscovery.org/jpl/primitives_repo.git
 * https://gitlab.datadrivendiscovery.org/nyu/ta3ta2-api
  * https://gitlab.com/datadrivendiscovery/ta3ta2-api -- newer than above???  There's a v2017.12.20 branch, sigh
 * https://gitlab.datadrivendiscovery.org/nyu/ta2ta3-stubs
 * New proposed TA2-TA3 interface: https://gitlab.com/datadrivendiscovery/ta3ta2-api/blob/preprocessing_api2/


## TA2 impl's

 * https://gitlab.datadrivendiscovery.org/MIT-FeatureLabs/TA2-TA3-Example
 * https://gitlab.datadrivendiscovery.org/uncharted/ta2-server

## TA3 impl's

 * https://gitlab.datadrivendiscovery.org/mgrabmair/cmu-ta3-webclient


# A bit more guts

Requests from the TA3 include a SessionContext which just contains a (string) ID.  Then the session ID is included with each request, so the server needs to remember that state.

Each session has N pipelines defined, which are basically trained models.  You can create, delete, alter and run pipelines.  Pipelines must also be able to be saved and exported to be re-run later.

Currently pipelines are 100% opaque.  Mitar's proposal, which we are going to need to do sooner or later, basically opens the black box and lets the user look inside, breaking pipelines into steps.  The steps can be defined, tuned, etc.  We need to build the system on the assumption that we're going to do either/both.

## Tech

 * plasma for object store?  https://arrow.apache.org/docs/python/plasma.html
 * Celery for task distribution and communication, almost certainly.  Alternatives to look at: rq, huey, maybe kuyruk.  Question is basically whether to use redis or rabbitmq as the message broker.
 * 


# Running (very ad-hoc)

So after some hacking of paths and ports, we can start Matthias's TA3 interface like so (having installed all the deps in our virtualenv and activating it):

```
env CONFIG_JSON_PATH=/home/sheath/projects/D3M/cmu-ta3/test-configs/test_config_185_local_mg.json python ta3ta2-proxy.py 
```

Now we can browse to `localhost:8088` (or whatever port set in `ta3ta2-proxy.py`) and should see "D3M January blah blah System Evaluation".

We start our TA2:

```
cd /home/sheath/projects/D3M/cmu-ta2
./src/main.py
```

Now we can hit "start session" in the thing and, lo and behold, it actually talks to our TA2!  Magic.

# Building docker image
Prerequisites:
1. [Install docker](https://docs.docker.com/install/).
    1. Check docker ```docker --version```
1. [For building docker] Create a [Gitlab access token](https://docs.gitlab.com/ee/user/profile/personal_access_tokens.html) for your Gitlab account
    1. Use ```read_repository``` for ```scope```
    1. This is used to clone the repository--[bayesian_optimization](https://gitlab.datadrivendiscovery.org/sray/bayesian_optimization) during the docker image creation

Run all commands as root.

## Create a docker image
Create a docker image by running the following command
```bash
docker build -t registry.datadrivendiscovery.org/sheath/cmu-ta2 . --build-arg gitlab_token=<access_token> --build-arg gitlab_user=<user>
```
where ```<access_token>``` is the Gitlab access token, 
and ```<user>``` is your Gitlab account.

This creates an image--```cmu-ta2```--in your machine’s local Docker image registry.

List docker images:
```bash
docker image ls
```

Finally, push the image to the D3M registory
```bash
docker push registry.datadrivendiscovery.org/sheath/cmu-ta2
```

## Pull the docker image from the D3M registory
```bash
docker push registry.datadrivendiscovery.org/sheath/cmu-ta2
```

## Run the docker image
Run the docker image, mapping your machine’s port 45042 to the container’s published port 45042 using ```-p```
```bash
docker run -i -t \
-p 45042:45042
docker push registry.datadrivendiscovery.org/sheath/cmu-ta2
```
