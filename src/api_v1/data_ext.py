
import data_ext_pb2 as data_ext_pb2
import data_ext_pb2_grpc as data_ext_pb2_grpc

class DataExt(data_ext_pb2_grpc.DataExtServicer):
    pass


def add_to_server(server):
    data_ext_pb2_grpc.add_DataExtServicer_to_server(DataExt(), server)