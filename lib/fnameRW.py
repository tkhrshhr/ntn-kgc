def fname_make(args):
    args_dict = vars(args)
    args_str = '#'.join('{}:{}'.format(key, val) for key, val in args_dict.items())
    return args_str


def tlname_make(fname):
    tlname_list = fname.split('#')[1:]
    tlname = '#'.join(tlname_list)
    return tlname


def fname_read(fname):
    args_list = fname.split('#')
    args_dict = dict()
    for key_val in args_list:
        key, val = key_val.split(':')
        if key in ['kg_choice', 'model', 'folder']:
            args_dict[key] = val
        elif key in ['weightdecay', 'learn_rate']:
            args_dict[key] = float(val)
        else:
            args_dict[key] = int(val)
    return args_dict


if __name__ == '__main__':
    fname_read('gpu:-1#kg_choice:w#model:d#weightdecay:0.0001#learn_rate:0.1#batchsize:1000#epoch:10#matpro:1#p_dim:1#dimension:100#slice_size:4#n_nsamp:10#save_period:1#folder:test0404#c_epoch:1')
