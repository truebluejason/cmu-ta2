#!/usr/bin/env python3

import logging
from concurrent import futures
import time
import sys

import grpc
from api_v3 import core

TA2_API_HOST = '[::]'
TA2_API_PORT = 45042

def main(argv):
    mode = argv[0]
    logging.info("Running in mode %s", mode)

    if mode == "search":
        core.search_phase()
    elif mode == "test":
        core.test_phase()
    else:
        threadpool = futures.ThreadPoolExecutor(max_workers=4)
        server = grpc.server(threadpool)
        core.add_to_server(server)
        server_string = '{}:{}'.format(TA2_API_HOST, TA2_API_PORT)
        server.add_insecure_port(server_string)
        logging.info("Starting server on %s", server_string)
        server.start()
        logging.info("Server started, waiting.")
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            server.stop(0)

if __name__ == '__main__':
    main(sys.argv[1:])
