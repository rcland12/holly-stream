import cv2
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
        self.new_shape = (640, 640)

    def execute(self, requests):
        responses = []
        for request in requests:
            img = letterbox(
                pb_utils.get_input_tensor_by_name(request, "INPUT_0").as_numpy(),
                self.new_shape,
                auto=False
            )
            img = img.transpose((2, 0, 1)).astype('float16')
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