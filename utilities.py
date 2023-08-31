import os

from ast import literal_eval



class EnvArgumentParser():
    def __init__(self):
        self.dict = {}
    
    class define_dict(dict):
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
        return self.define_dict(self.dict)