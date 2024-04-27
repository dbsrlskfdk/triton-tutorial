import argparse

import grpc
import numpy as np
from tritonclient.grpc import service_pb2, service_pb2_grpc
from tritonclient.utils import triton_to_np_dtype

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="add_net")
    parser.add_argument("--model_version", type=str, default="1")
    parser.add_argument("--url", type=str, default="0.0.0.0:8101")
    args = parser.parse_args()

    # Create gRPC client
    channel = grpc.insecure_channel(args.url)
    grpc_stub = service_pb2_grpc.GRPCInferenceServiceStub(channel)

    # Get model metadata
    model_meta_request = service_pb2.ModelMetadataRequest(name=args.model_name, version=args.model_version)
    model_meta_resp = grpc_stub.ModelMetadata(model_meta_request)

    # Prepare input, output meta
    input_names = [input_tensor.name for input_tensor in model_meta_resp.inputs]
    output_names = [output_tensor.name for output_tensor in model_meta_resp.outputs]
    input_dtypes = [input_tensor.datatype for input_tensor in model_meta_resp.inputs]
    output_dtypes = [output_tensor.datatype for output_tensor in model_meta_resp.outputs]

    # Sample input data
    input_datas = [
        np.array([[1, 2, 3], [10, 11, 23]], dtype=triton_to_np_dtype(input_dtypes[0])),
        np.array([[4, 5, 6], [1, 2, 3]], dtype=triton_to_np_dtype(input_dtypes[1])),
    ]
    print("Input Shape : ", list(input_datas[0].shape))

    for i in range(len(input_datas)):
        print("Input Data :\n", input_datas[i])
        # Prepare input Request
        request = service_pb2.ModelInferRequest(model_name=args.model_name, model_version=args.model_version)
        for j, name in enumerate(input_names):
            input_tensor = service_pb2.ModelInferRequest().InferInputTensor(
                name=name, datatype=input_dtypes[j], shape=[len(input_datas[i][j])]
            )
            request.inputs.extend([input_tensor])
            request.raw_input_contents.extend([input_datas[i][j].tobytes()])

        # Prepare output Request
        output_tensor = service_pb2.ModelInferRequest().InferRequestedOutputTensor(name=output_names[0])
        request.outputs.extend([output_tensor])

        # Send request
        try:
            response = grpc_stub.ModelInfer(request)
            response = np.frombuffer(response.raw_output_contents[0], dtype=triton_to_np_dtype(output_dtypes[0]))
            print("Output Data : ", response)
        except Exception as e:
            print("Inference failed: {}".format(e))
            exit(1)
