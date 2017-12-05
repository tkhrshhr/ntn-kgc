import argparse
import numpy as np
import logging

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


def get_mrr_and_hits(tri, all_ent_ids, gs, scores, corrupt_s=True):
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
    rank = np.where(descending == index)[0][0] + 1
    # - MRR
    scores[6] += 1 / rank
    # - Hits
    if rank <= 1:
        scores[0] += 1
    if rank <= 3:
        scores[1] += 1
    if rank <= 10:
        scores[2] += 1

    # Filtered
    f_gs = np.delete(gs, f_inds)
    f_all_ent_ids = np.delete(all_ent_ids, f_inds)

    descending = f_all_ent_ids[np.argsort(f_gs)[::-1]]
    rank = np.where(descending == index)[0][0] + 1
    # - MRR
    scores[7] += 1 / rank
    # - Hits
    if rank <= 1:
        scores[3] += 1
    if rank <= 3:
        scores[4] += 1
    if rank <= 10:
        scores[5] += 1

    return scores


def process(model, tri):
    # hits1, hits3, hits10, hits1f, hits3f, hits10f, mrr, mrrf
    scores = np.zeros(8)

    r, s, o = tri
    print(tri)
    r_ids = np.repeat(r, n_ent)
    s_ids = np.repeat(s, n_ent)
    o_ids = np.repeat(o, n_ent)
    all_ent_ids = np.arange(n_ent).astype(np.int32)

    gs = model.get_g(r_ids, all_ent_ids, o_ids).reshape(n_ent,).data
    scores_s = get_mrr_and_hits(tri, all_ent_ids, gs, scores, corrupt_s=True)

    gs = model.get_g(r_ids, s_ids, all_ent_ids).reshape(n_ent,).data
    scores_o = get_mrr_and_hits(tri, all_ent_ids, gs, scores, corrupt_s=False)

    return np.stack((scores_s, scores_o))


def get_all_metrics(data, model):
    # Prepare score dictionary
    score_dict = {}

    # Prepare data
    data = np.array(data, np.int32)
    scores = Parallel(n_jobs=-1)([delayed(process)(model, tri) for tri in data])
    scores = np.concatenate(scores, axis=0)
    example_length = len(scores)
    scores = np.sum(scores, axis=0)
    scores = scores / example_length
    score_dict['hits1'] = scores[0]
    score_dict['hits3'] = scores[1]
    score_dict['hits10'] = scores[2]
    score_dict['hits1f'] = scores[3]
    score_dict['hits3f'] = scores[4]
    score_dict['hits10f'] = scores[5]
    score_dict['mrr'] = scores[6]
    score_dict['mrrf'] = scores[7]
    return score_dict


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--save', '-s', type=str, default='',
                        help='save name to test')
    args = parser.parse_args()

    # create logger with 'spam_application'
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    # create file handler which logs even debug messages
    fh = logging.FileHandler('test_result/{}.log'.format(args.save), mode='w')
    fh.setLevel(logging.INFO)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)

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

    # Dev
    logger.info('---dev---')
    score_dict = get_all_metrics(dev, model)
    for key in score_dict.keys():
        logger.info('{}: {}'.format(key, score_dict[key]))

    # Test
    logger.info('---test---')
    score_dict = get_all_metrics(test, model)
    for key in score_dict.keys():
        logger.info('{}: {}'.format(key, score_dict[key]))


if __name__ == '__main__':
    main()
