import json

import cv2 as cv
import numpy as np
import triton_python_backend_utils as pb_utils



def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    shape = im.shape[:2]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    
    dw /= 2
    dh /= 2
    
    if shape[::-1] != new_unpad:
        im = cv.resize(im, new_unpad, interpolation=cv.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv.copyMakeBorder(im, top, bottom, left, right, cv.BORDER_CONSTANT, value=color)
    return im



class TritonPythonModel:
    def initialize(self, args):
        model_config = json.loads(args['model_config'])
        self.output0_config = pb_utils.get_output_config_by_name(model_config, "OUTPUT_0")

    def execute(self, requests):
        responses = []
        for request in requests:
            img = pb_utils.get_input_tensor_by_name(request, "INPUT_0").as_numpy()

            img = img[:, :, ::-1]
            img = letterbox(img, auto=False)
            img = img.transpose((2, 0, 1)).astype('float32')
            img /= 255

            out_tensor_0 = pb_utils.Tensor("OUTPUT_0", img)
            inference_response = pb_utils.InferenceResponse(output_tensors=[out_tensor_0])
            responses.append(inference_response)              
          
        return responses

    def finalize(self):
        print('Cleaning up...')