import unittest
from api_v1 import core, data_ext, dataflow_ext


import core_pb2 as core_pb2
import core_pb2_grpc as core_pb2_grpc

from concurrent import futures
import grpc



class TestCore(unittest.TestCase):
    def setUp(self):
        threadpool = futures.ThreadPoolExecutor(max_workers=4)
        self.__server__ = grpc.server(threadpool)
        core.add_to_server(self.__server__)
        data_ext.add_to_server(self.__server__)
        dataflow_ext.add_to_server(self.__server__)
        self.__server__.add_insecure_port('localhost:50051')
        self.__server__.start()

    def tearDown(self):
        self.__server__.stop(0)

    def test_session(self):
        channel = grpc.insecure_channel('localhost:50051')
        stub = core_pb2_grpc.CoreStub(channel)
        msg = core_pb2.SessionRequest(user_agent="unittest", version="Foo")
        session = stub.StartSession(msg)
        print(session)

        session_end_response = stub.EndSession(session.context)

if __name__ == '__main__':
    unittest.main()