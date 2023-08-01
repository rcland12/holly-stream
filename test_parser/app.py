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
        if d_type == list or d_type == tuple:
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


if __name__ == "__main__":
    parser = EnvArgumentParser()
    parser.add_arg("ARG_A", default="a", type=str)
    parser.add_arg("ARG_B", default=[1, 2, 3], type=list)
    parser.add_arg("ARG_C", default=4.0, type=float)
    parser.add_arg("ARG_D", default=100, type=int)
    parser.add_arg("ARG_E", default=(1,), type=tuple)
    args = parser.parse_args()

    print(args.ARG_A)
    print(type(args.ARG_A))
    print(args.ARG_B)
    print(type(args.ARG_B))
    print(args.ARG_C)
    print(type(args.ARG_C))
    print(args.ARG_D)
    print(type(args.ARG_D))
    print(args.ARG_E)
    print(type(args.ARG_E))
