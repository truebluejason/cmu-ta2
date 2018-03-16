"""
Implementation of the ta2ta3 API v1 (preprocessing extensions) -- core.proto
"""


import core_pb2 as core_pb2
import core_pb2_grpc as core_pb2_grpc
import logging
import primitive_lib

logging.basicConfig(level=logging.INFO)

__version__ = "0.1.0"

__symbol_idx = 0
def gensym(id="gensym"):
    global __symbol_idx
    s = "{}_{}".format(id, __symbol_idx)
    __symbol_idx += 1
    return s

def run_pipeline_mockup():
    import time
    yield core_pb2.SUBMITTED
    time.sleep(1)
    yield core_pb2.RUNNING
    time.sleep(1)
    yield core_pb2.COMPLETED

class Core(core_pb2_grpc.CoreServicer):
    def __init__(self):
        self._sessions = set()

    def _new_session_id(self):
        "Returns an identifier string for a new session."
        return gensym("session")

    def _new_pipeline_id(self):
        "Returns an identifier string for a new pipeline."
        return gensym("pipeline")

    def CreatePipelines(self, request, context):
        logging.info("Message received: CreatePipelines: %s", request)
        pipeline_id = self._new_pipeline_id()
        pipeline = core_pb2.Pipeline(
            predict_result_uri = "",
            output = core_pb2.OUTPUT_TYPE_UNDEFINED,
            scores = []
        )
        for status in run_pipeline_mockup():
            msg = core_pb2.PipelineCreateResult(
                response_info=core_pb2.Response(
                    status=core_pb2.Status(code=core_pb2.OK),
                ),
                progress_info=status,
                pipeline_id=pipeline_id,
                pipeline_info=pipeline
            )
            yield msg


    def ExecutePipeline(self, request, context):
        logging.info("Message received: ExecutePipelines")
        yield core_pb2.PipelineExecuteResult(
            response_info=core_pb2.Response(
                status=core_pb2.Status(code=core_pb2.OK),
            ),
            progress_info=core_pb2.COMPLETED,
            pipeline_id=1,
            result_uri=1,
        )
        

    def ListPipelines(self, request, context):
        logging.info("Message received: ListPipelines")
        return core_pb2.PipelineListResult(
            response_info = core_pb2.Response(
                status=core_pb2.Status(code=core_pb2.OK)),
            pipeline_ids = [
                "pipeline_1",
            ]
        )


    def DeletePipelines(self, request, context):
        logging.info("Message received: DeletePipelines")
        return core_pb2.PipelineListResult(
            response_info = core_pb2.Response(
                status=core_pb2.Status(code=core_pb2.OK)),
            pipeline_ids = [
                
            ]
        )

    def GetCreatePipelineResults(self, request, context):
        logging.info("Message received: GetCreatePipelineResults")
        return core_pb2.Response(core_pb2.Status(code=core_pb2.OK))
    def GetExecutePipelineResults(self, request, context):
        logging.info("Message received: GetExecutePipelineResults")
        return core_pb2.Response(core_pb2.Status(code=core_pb2.OK))

    def ExportPipeline(self, request, context):
        logging.info("Message received: ExportPipeline")
        return core_pb2.Response(core_pb2.Status(code=core_pb2.OK))

    def UpdateProblemSchema(self, request, context):
        logging.info("Message received: UpdateProblemSchema")
        return core_pb2.Response(core_pb2.Status(code=core_pb2.OK))

    def StartSession(self, request, context):
        logging.info("Message received: StartSession %s", request)
        version = core_pb2.DESCRIPTOR.GetOptions().Extensions[
            core_pb2.protocol_version]
        session_id = self._new_session_id()
        self._sessions.add(session_id)
        # session = "session_%d" % len(self.sessions)
        # self.sessions.add(session)
        logging.info("Session started: %s (protocol version %s)", session_id, version)
        return core_pb2.SessionResponse(
            response_info=core_pb2.Response(
                status=core_pb2.Status(code=core_pb2.OK)
            ),
            user_agent="cmu_ta2 %s" % __version__,
            version=version,
            context=core_pb2.SessionContext(session_id=session_id),
        )

    def EndSession(self, request, context):
        logging.info("Message received: EndSession")
        if request.session_id in self._sessions:
            self._sessions.remove(request.session_id)
            logging.info("Session terminated: %s", request.session_id)
            return core_pb2.Response(
                status=core_pb2.Status(code=core_pb2.OK),
            )
        else:
            logging.warn("Client tried to end session %s which does not exist", request.session_id)
            return core_pb2.Response(
                status=core_pb2.Status(code=core_pb2.SESSION_UNKNOWN),
            )

def add_to_server(server):
    core_pb2_grpc.add_CoreServicer_to_server(Core(), server)