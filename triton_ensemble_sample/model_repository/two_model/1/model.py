import json

import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        self.model_config = model_config = json.loads(args["model_config"])
        output_configs = model_config["output"]

        self.output_names = [output_config["name"] for output_config in output_configs]
        self.output_dtypes = [
            pb_utils.triton_string_to_numpy(output_config["data_type"]) for output_config in output_configs
        ]

    def execute(self, requests):
        responses = []
        for request in requests:
            input_1_tensor = pb_utils.get_input_tensor_by_name(request, "two_input")
            input_1_data = input_1_tensor.as_numpy()

            output_data = np.add(input_1_data, 100)
            output_tensor = pb_utils.Tensor("two_output", output_data.astype(self.output_dtypes[0]))
            response = pb_utils.InferenceResponse(output_tensors=[output_tensor])
            responses.append(response)
        return responses

    def finalize(self):
        pass
