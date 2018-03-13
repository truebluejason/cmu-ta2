#!/bin/sh

for protofile in ta3ta2-api/*.proto; do
    echo "Building" $protofile
    python -m grpc_tools.protoc -Ita3ta2-api --python_out=src/proto --grpc_python_out=src/proto $protofile
done