#!/usr/bin/env python3

import logging
from concurrent import futures
import time
import util
import primitive_lib

import grpc
from api_v3 import core

TA2_API_HOST = '[::]'
TA2_API_PORT = 45042

def main():
    threadpool = futures.ThreadPoolExecutor(max_workers=4)
    server = grpc.server(threadpool)
    core.add_to_server(server)
    server_string = '{}:{}'.format(TA2_API_HOST, TA2_API_PORT)
    server.add_insecure_port(server_string)
    logging.info("Starting server on %s", server_string)
    #prims = primitive_lib.list_primitives()
    server.start()
    logging.info("Server started, waiting.")
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        server.stop(0)

if __name__ == '__main__':
    main()
