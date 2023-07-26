import numpy as np
import onnxruntime


class ONNXModel():
    def __init__(self, onnx_file):
        self.onnx_session = onnxruntime.InferenceSession(onnx_file, providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider',
                                                           'CPUExecutionProvider'])

        self.input_name = self.get_input_name(self.onnx_session)
        self.output_name = self.get_output_name(self.onnx_session)
        print("input_name:{}".format(self.input_name))
        print("output_name:{}".format(self.output_name))

    def get_output_name(self, onnx_session):
        output_name = []
        for node in onnx_session.get_outputs():
            output_name.append(node.name)
        return output_name

    def get_input_name(self, onnx_session):
        input_name = []
        for node in onnx_session.get_inputs():
            input_name.append(node.name)
        return input_name

    def get_input_feed(self, input_name, image:np.ndarray):
        input_feed = {}
        for name in input_name:
            input_feed[name] = image
        return input_feed

    def forward(self, image:np.ndarray):
        input_feed = self.get_input_feed(self.input_name, image)
        result = self.onnx_session.run(self.output_name, input_feed=input_feed)
        return result



def to_numpy(tensor):
    print(tensor.device)
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
