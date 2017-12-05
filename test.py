import argparse
import numpy as np

from joblib import Parallel, delayed

from chainer import serializers

import reader

from models.nn import NN
from models.ntn import NTN
from models.ntnd import NTNd
from models.ntnc import NTNc
from models.ntns import NTNs


def parse_hyparams(model_name):
    hp_dict = {}

    hps = model_name.split('-')
    print(hps)
    hp_dict['dataset'] = hps[0]
    hp_dict['model'] = hps[1]
    for hp in hps[2:]:
        key = hp[0]
        value = hp[1:]
        if key == 'w':
            hp_dict[key] = float(value)
        else:
            print(key, value)
            hp_dict[key] = int(value)

    return hp_dict


def get_mrr_and_hits(tri, score_dict, all_ent_ids, gs, corrupt_s=True):
    r, s, o = tri
    if corrupt_s is True:
        index = s
        ro2s[(r, o)].remove(s)
        f_inds = ro2s[(r, o)]
    elif corrupt_s is False:
        index = o
        rs2o[(r, s)].remove(o)
        f_inds = rs2o[(r, s)]
    # Raw
    descending = all_ent_ids[np.argsort(gs)[::-1]]
    rank = np.where(descending == index)[0]
    # - MRR
    score_dict['mrr'] += 1 / rank
    # - Hits
    if rank < 1:
        score_dict['hits1'] += 1
    elif rank < 3:
        score_dict['hits3'] += 1
    elif rank < 10:
        score_dict['hits10'] += 1

    # Filtered
    print(f_inds)
    f_gs = np.delete(gs, f_inds)
    f_all_ent_ids = np.delete(all_ent_ids, f_inds)

    descending = f_all_ent_ids[np.argsort(f_gs)[::-1]]
    rank = np.where(descending == index)[0]
    # - MRR
    score_dict['mrrf'] += 1 / rank
    # - Hits
    if rank < 1:
        score_dict['hits1f'] += 1
    elif rank < 3:
        score_dict['hits3f'] += 1
    elif rank < 10:
        score_dict['hits10f'] += 1

    return 0


def process(model, tri, score_dict):
    r, s, o = tri
    r_ids = np.repeat(r, n_ent)
    s_ids = np.repeat(s, n_ent)
    o_ids = np.repeat(o, n_ent)
    all_ent_ids = np.arange(n_ent).astype(np.int32)

    gs = model.get_g(r_ids, all_ent_ids, o_ids).reshape(n_ent,).data
    get_mrr_and_hits(tri, score_dict, all_ent_ids, gs, corrupt_s=True)

    gs = model.get_g(r_ids, s_ids, all_ent_ids).reshape(n_ent,).data
    get_mrr_and_hits(tri, score_dict, all_ent_ids, gs, corrupt_s=False)


def get_all_metrics(data, model):
    # Prepare score dictionary
    score_dict = {}
    score_dict['hits10'] = 0
    score_dict['hits3'] = 0
    score_dict['hits1'] = 0
    score_dict['hits10f'] = 0
    score_dict['hits3f'] = 0
    score_dict['hits1f'] = 0
    score_dict['mrr'] = 0
    score_dict['mrrf'] = 0

    # Prepare data
    data = np.array(data, np.int32)
    Parallel(n_jobs=-1)([delayed(process)(model, tri, score_dict) for tri in data])

    return score_dict


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--save', '-s', type=str, default='',
                        help='save name to test')
    args = parser.parse_args()

    # Parse save file name
    hp_dict = parse_hyparams(args.save)

    # Read dataset
    global n_ent
    global rs2o
    global ro2s
    train, dev, test, n_ent, n_rel, rs2o, ro2s = reader.read(hp_dict['dataset'])

    # Prepare model
    params = {}
    params['n_ent'] = n_ent
    params['n_rel'] = n_rel
    params['n_nsamp'] = hp_dict['s']
    params['d'] = hp_dict['d']
    params['k'] = hp_dict['k']
    if hp_dict['model'] == 'n':
        model = NN(**params)
    elif hp_dict['model'] == 't':
        model = NTN(**params)
    elif hp_dict['model'] == 'd':
        model = NTNd(**params)
    elif hp_dict['model'] == 'c':
        model = NTNc(**params)
    elif hp_dict['model'] == 's':
        params['p'] = hp_dict['p']
        model = NTNs(**params)

    serializers.load_hdf5("trained_model/" + args.save, model)
    model.to_cpu()
    score_dict = get_all_metrics(dev, model)
    for key in score_dict.keys():
        print(key, score_dict[key])


if __name__ == '__main__':
    main()
