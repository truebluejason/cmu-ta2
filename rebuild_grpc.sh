#!/bin/sh
# This script assumes that the ta3ta2-api repo is located at ../ta3ta2-api
# (i.e. side-by-side with your cmu-ta2 repo).
#
# Gernerate python files from the protobuf spec
# The paths and such here are *super fiddly* to get the protobuf paths
# and python modules to line up right.  Sigh.
# See https://github.com/google/protobuf/issues/1491 and related issues,
# which don't really fix anything, just tell you how to work around them
# if you get things EXACTLY RIGHT.

python -m grpc_tools.protoc -I ../ta3ta2-api --python_out=src --grpc_python_out=src ../ta3ta2-api/*.proto
