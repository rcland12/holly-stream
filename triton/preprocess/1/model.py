import cv2
import json
import numpy
import triton_python_backend_utils as pb_utils



def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, stride=32):
    shape = im.shape[:2]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]

    if auto:
        dw, dh = numpy.mod(dw, stride), numpy.mod(dh, stride)
    
    dw /= 2
    dh /= 2
    
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return im



class TritonPythonModel:
    def initialize(self, args):
        model_config = json.loads(args['model_config'])
        self.output0_config = pb_utils.get_output_config_by_name(model_config, "OUTPUT_0")

    def execute(self, requests):
        output0_dtype = pb_utils.triton_string_to_numpy(self.output0_config['data_type'])
        print(output0_dtype)

        responses = []
        for request in requests:
            image = pb_utils.get_input_tensor_by_name(request, "INPUT_0").as_numpy()

            img = letterbox(image, auto=False)
            img = img.transpose((2, 0, 1)).astype(output0_dtype)
            img /= 255

            responses.append(
                pb_utils.InferenceResponse(
                    output_tensors=[
                        pb_utils.Tensor(
                            "OUTPUT_0",
                            img
                        )
                    ]
                )
            )              
          
        return responses

    def finalize(self):
        print('Cleaning up...')