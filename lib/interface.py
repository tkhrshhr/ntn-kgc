import json


def fname_make(args):
    args_dict = vars(args)
    args_str = '#'.join('{}:{}'.format(key, val) for key, val in args_dict.items())
    return args_str
