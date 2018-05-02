import argparse
import numpy as np
import logging
import sys

from joblib import Parallel, delayed
from chainer import serializers

from models.nn import NN
from models.ntn import NTN
from models.ntnd import NTNd
from models.ntnc import NTNc
from models.ntns import NTNs

from lib import reader, fnameRW


"""Functions for ranking"""


def get_mrr_and_hits(tri, all_ent_ids, gs, corrupt_s=True):
    r, s, o = tri
    if corrupt_s is True:
        index = s
        ro2s[(r, o)].remove(s)
        f_inds = ro2s[(r, o)]
    elif corrupt_s is False:
        index = o
        rs2o[(r, s)].remove(o)
        f_inds = rs2o[(r, s)]

    target_score = gs[index]
    scores = np.zeros(8)
    # Raw
    rank = np.sum(gs >= target_score)
#    descending = all_ent_ids[np.argsort(gs)[::-1]]
#    rank = np.where(descending == index)[0][0] + 1
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
#    f_all_ent_ids = np.delete(all_ent_ids, f_inds)
#    descending = f_all_ent_ids[np.argsort(f_gs)[::-1]]
#    rank = np.where(descending == index)[0][0] + 1
    rank = np.sum(f_gs >= target_score)
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
    r, s, o = tri
    r_ids = np.repeat(r, n_ent)
    s_ids = np.repeat(s, n_ent)
    o_ids = np.repeat(o, n_ent)
    all_ent_ids = np.arange(n_ent).astype(np.int32)

    gs = model.get_g(r_ids, all_ent_ids, o_ids).reshape(n_ent,).data
    scores_s = get_mrr_and_hits(tri, all_ent_ids, gs, corrupt_s=True)

    gs = model.get_g(r_ids, s_ids, all_ent_ids).reshape(n_ent,).data
    scores_o = get_mrr_and_hits(tri, all_ent_ids, gs, corrupt_s=False)

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
    print(example_length)
    score_dict['hits1'] = scores[0]
    score_dict['hits3'] = scores[1]
    score_dict['hits10'] = scores[2]
    score_dict['hits1f'] = scores[3]
    score_dict['hits3f'] = scores[4]
    score_dict['hits10f'] = scores[5]
    score_dict['mrr'] = scores[6]
    score_dict['mrrf'] = scores[7]
    return score_dict


"""Functions for classification"""


def get_scores(data, model):
    # Prepare data
    data_length = len(data)
    r_ids = data[:, 0]
    s_ids = data[:, 1]
    o_ids = data[:, 2]

    # Return scores
    return model.get_g(r_ids, s_ids, o_ids).reshape(data_length,).data


def get_accuracy(labels, scores, threshold):
    return np.sum(labels * (scores > threshold)) / len(labels)


def get_threshold(data, scores):
    # Prepare label
    labels = data[:, 3]
    n_intervals = len(data) - 1

    # Get the threshold
    max = np.max(scores)
    min = np.min(scores)
    increment = (max - min) / n_intervals
    threshold = min
    accuracy = 0
    new_accuracy = 0
    while accuracy <= new_accuracy:
        new_accuracy = get_accuracy(labels, scores, threshold)
        if accuracy <= new_accuracy:
            accuracy = new_accuracy
        threshold += increment
    return accuracy, threshold


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', '-fl', type=str, default='',
                        help='file to be tested')
    parser.add_argument('--classORrank', '-cr', type=str, default='rank',
                        help='file to be tested')
    test_args = parser.parse_args()

    # Read the trained model name and set the test log file and folder name
    train_args_dict = fnameRW.fname_read(test_args.file)
    tl_name = fnameRW.tlname_make(test_args.file)
    tl_folder = train_args_dict['folder']

    # Create logger with 'spam_application'
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    # Create file handler which logs even debug messages
    fh = logging.FileHandler('test_result/{}/{}.log'.format(tl_folder, 'class' + tl_name), mode='w')
    fh.setLevel(logging.INFO)
    # Create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    # Create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # Add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    # Read dataset
    global n_ent
    global rs2o
    global ro2s
    train, dev, test, n_ent, n_rel, rs2o, ro2s = reader.read(train_args_dict['kg_choice'])

    # Prepare model
    params = {}
    params['n_ent'] = n_ent
    params['n_rel'] = n_rel
    params['n_nsamp'] = train_args_dict['n_nsamp']
    params['d'] = train_args_dict['dimension']
    params['k'] = train_args_dict['slice_size']
    if train_args_dict['model'] == 'n':
        model = NN(**params)
    elif train_args_dict['model'] == 't':
        params['mp'] = train_args_dict['matpro']
        model = NTN(**params)
    elif train_args_dict['model'] == 'd':
        params['mp'] = train_args_dict['matpro']
        model = NTNd(**params)
    elif train_args_dict['model'] == 'c':
        params['mp'] = train_args_dict['matpro']
        model = NTNc(**params)
    elif train_args_dict['model'] == 's':
        params['p'] = train_args_dict['p_dim']
        model = NTNs(**params)

    serializers.load_hdf5("trained_model/{}/{}".format(train_args_dict['folder'], test_args.file), model)
    model.to_cpu()

    if test_args.classORrank == 'rank':
        """
        # Dev
        logger.info('---dev---')
        score_dict = get_all_metrics(dev[:100], model)
        for key in score_dict.keys():
            logger.info('{}: {}'.format(key, score_dict[key]))
        """

        # Test
        logger.info('---test---')
        score_dict = get_all_metrics(test, model)
        for key in score_dict.keys():
            logger.info('{}: {}'.format(key, score_dict[key]))

    elif test_args.classORrank == 'class':
        # Dev
        dev = np.array(dev, np.int32)
        logger.info('---dev---')
        dev_scores = get_scores(dev, model)
        dev_accuracy, threshold = get_threshold(dev, dev_scores)
        logger.info('dev_accuracy: {}'.format(dev_accuracy))

        # Test
        test = np.array(test, np.int32)
        logger.info('---test---')
        test_scores = get_scores(test, model)
        test_accuracy = get_accuracy(test[:, 3], test_scores, threshold)
        logger.info('test_accuracy: {}'.format(test_accuracy))


if __name__ == '__main__':
    main()
