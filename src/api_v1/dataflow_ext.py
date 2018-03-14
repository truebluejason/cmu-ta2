
import dataflow_ext_pb2 as dataflow_ext_pb2
import dataflow_ext_pb2_grpc as dataflow_ext_pb2_grpc

class DataflowExt(dataflow_ext_pb2_grpc.DataflowExtServicer):
    pass


def add_to_server(server):
    dataflow_ext_pb2_grpc.add_DataflowExtServicer_to_server(DataflowExt(), server)