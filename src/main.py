#!/usr/bin/env python3

__author__ = "Saswati Ray"
__email__ = "sray@cs.cmu.edu"

import logging
from multiprocessing import set_start_method
set_start_method("spawn", force=True)
from concurrent import futures
import time
import sys
import warnings

warnings.filterwarnings("ignore")

import grpc
import search
from api_v3 import core
from multiprocessing import cpu_count

TA2_API_HOST = '[::]'
TA2_API_PORT = 45042

def main(argv):
    mode = argv[0]
    logging.info("Running in mode %s", mode)

    if mode == "search":
        search.search_phase()
    else:
        threadpool = futures.ThreadPoolExecutor(max_workers=cpu_count())
        server = grpc.server(threadpool)
        core.add_to_server(server)
        server_string = '{}:{}'.format(TA2_API_HOST, TA2_API_PORT)
        server.add_insecure_port(server_string)
        logging.critical("Starting server on %s", server_string)
        server.start()
        logging.critical("Server started, waiting.")
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            server.stop(0)

if __name__ == '__main__':
    main(sys.argv[1:])
