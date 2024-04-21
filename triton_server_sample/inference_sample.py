import numpy as np
from tritony import InferenceClient

if __name__ == "__main__":
    client = InferenceClient.create_with(
        model="add_net", url="0.0.0.0:8101", model_version="1", protocol="grpc", run_async=True
    )

    input_data = {
        "INPUT__0": np.array([[1, 2, 3], [10, 11, 23]], dtype=np.float32),
        "INPUT__1": np.array([[4, 5, 6], [1, 2, 3]], dtype=np.float32),
    }
    response = client(input_data)
    print(response)
