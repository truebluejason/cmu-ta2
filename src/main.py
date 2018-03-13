#!/usr/bin/env python3

import logging
from concurrent import futures
import time

import grpc
import core_pb2 as core_pb2
import core_pb2_grpc as core_pb2_grpc

logging.basicConfig(level=logging.WARNING)

__version__ = "0.1.0"

class Core(core_pb2_grpc.CoreServicer):
    def CreatePipelines(self, request, context):
        pipeline_id = 1
        progress = 1
        msg = core_pb2.PipelineCreateResult(
            response_info=core_pb2.Response(
                status=core_pb2.Status(code=core_pb2.OK),
            ),
            progress_info=progress,
            pipeline_id=pipeline_id,
        )
        yield msg


    def ExecutePipeline(self, request, context):
        yield core_pb2.PipelineExecuteResult(
            response_info=core_pb2.Response(
                status=core_pb2.Status(code=core_pb2.OK),
            ),
            progress_info=core_pb2.COMPLETED,
            pipeline_id=1,
            result_uri=1,
        )
        

    def ListPipelines(self, request, context):
        return core_pb2.PipelineListResult(
            response_info = core_pb2.Response(
                status=core_pb2.Status(code=core_pb2.OK)),
            pipeline_ids = [
                "pipeline_1",
            ]
        )


    def DeletePipelines(self, request, context):
        return core_pb2.PipelineListResult(
            response_info = core_pb2.Response(
                status=core_pb2.Status(code=core_pb2.OK)),
            pipeline_ids = [
                
            ]
        )

    def GetCreatePipelineResults(self, request, context):
        return core_pb2.Response(core_pb2.Status(code=core_pb2.OK))
    def GetExecutePipelineResults(self, request, context):
        return core_pb2.Response(core_pb2.Status(code=core_pb2.OK))

    def ExportPipeline(self, request, context):
        return core_pb2.Response(core_pb2.Status(code=core_pb2.OK))

    def UpdateProblemSchema(self, request, context):
        return core_pb2.Response(core_pb2.Status(code=core_pb2.OK))

    def StartSession(self, request, context):
        version = core_pb2.DESCRIPTOR.GetOptions().Extensions[
            core_pb2.protocol_version]
        session = 1
        # session = "session_%d" % len(self.sessions)
        # self.sessions.add(session)
        logging.info("Session started: 1 (protocol version %s)", version)
        return core_pb2.SessionResponse(
            response_info=core_pb2.Response(
                status=core_pb2.Status(code=core_pb2.OK)
            ),
            user_agent="cmu_ta2 %s" % __version__,
            version=version,
            context=core_pb2.SessionContext(session_id=session),
        )

    def EndSession(self, request, context):
        logging.info("Session terminated: %s", request.session_id)
        return core_pb2.Response(
            status=core_pb2.Status(code=core_pb2.OK),
        )

def main():
    threadpool = futures.ThreadPoolExecutor
    server = grpc.server(threadpool)
    core_pb2_grpc.add_CoreServicer_to_server(Core(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        server.stop(0)

if __name__ == '__main__':
    main()