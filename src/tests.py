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
        self.assertTrue(session.response_info.status.code == core_pb2.OK)

        session_end_response = stub.EndSession(session.context)
        self.assertTrue(session_end_response.status.code == core_pb2.OK)

        # Try to end a session that does not exist
        fake_context = core_pb2.SessionContext(session_id="fake context")
        session_end_response = stub.EndSession(fake_context)
        self.assertTrue(session_end_response.status.code == core_pb2.SESSION_UNKNOWN)


    def test_pipeline(self):
        "Tries setting up a new pipeline"
        channel = grpc.insecure_channel('localhost:50051')
        stub = core_pb2_grpc.CoreStub(channel)
        msg = core_pb2.SessionRequest(user_agent="unittest", version="Foo")
        session = stub.StartSession(msg)
        self.assertTrue(session.response_info.status.code == core_pb2.OK)

        pipeline_request = core_pb2.PipelineCreateRequest(
            context=session.context,
            dataset_uri="file:///foo",
            task=core_pb2.TASK_TYPE_UNDEFINED,
            task_subtype=core_pb2.TASK_SUBTYPE_UNDEFINED,
            task_description="",
            output=core_pb2.OUTPUT_TYPE_UNDEFINED,
            metrics=[],
            target_features=[],
            predict_features=[],
            max_pipelines=10
        )
        p = stub.CreatePipelines(pipeline_request)
        for response in p:
            self.assertTrue(response.response_info.status.code == core_pb2.OK)


if __name__ == '__main__':
    unittest.main()