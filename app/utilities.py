import os
from ast import literal_eval



class EnvArgumentParser():
    def __init__(self):
        self.dict = {}

    class _define_dict(dict):
        __getattr__ = dict.get
        __setattr__ = dict.__setitem__
        __delattr__ = dict.__delitem__

    def add_arg(self, variable, default=None, type=str):
        env = os.environ.get(variable)

        if env is None:
            value = default
        else:
            value = self.cast_type(env, type)

        self.dict[variable] = value

    def cast_type(self, arg, d_type):
        if d_type == list or d_type == tuple or d_type == bool:
            try:
                cast_value = literal_eval(arg)
                return cast_value
            except (ValueError, SyntaxError):
                raise ValueError(f"Argument {arg} does not match given data type or is not supported.")
        else:
            try:
                cast_value = d_type(arg)
                return cast_value
            except (ValueError, SyntaxError):
                raise ValueError(f"Argument {arg} does not match given data type or is not supported.")
    
    def parse_args(self):
        return self._define_dict(self.dict)


def create_warmup_file(save_path, np_array=None, string=False, img_path=None):
    # For Triton model warmup, typical model input is a np.array, string, or image
    # The warmup files must be in bytes, so use this function to create them.
    if hasattr(np_array, 'shape'):
        np_array.tofile(save_path)
    elif isinstance(img_path, str):
        import cv2
        img = cv2.imread(img_path)
        img.tofile(save_path)
    elif isinstance(string, str):
        from tritonclient.utils import serialize_byte_tensor
        serialized = serialize_byte_tensor(np.array([string.encode("utf-8")], dtype=object))
        with open(save_path, "wb") as f:
            f.write(serialized.item())
    else:
        print("Invalid input. Input a numpy array, string, or image path")


if __name__ == "__main__":
    import numpy as np
    create_warmup_file("../triton/preprocess/warmup/INPUT_0", img_path="./images/pic_228.png")
    create_warmup_file("../triton/postprocess/warmup/input_1", np_array=np.array([1280, 720]))