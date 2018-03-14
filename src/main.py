#!/usr/bin/env python3

import logging
from concurrent import futures
import time

import grpc
from api_v1 import core, data_ext, dataflow_ext

logging.basicConfig(level=logging.WARNING)

__version__ = "0.1.0"

def main():
    threadpool = futures.ThreadPoolExecutor
    server = grpc.server(threadpool)
    core.add_to_server(server)
    data_ext.add_to_server(server)
    dataflow_ext.add_to_server(server)
    server.add_insecure_port('[::]:50051')
    server.start()
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        server.stop(0)

if __name__ == '__main__':
    main()