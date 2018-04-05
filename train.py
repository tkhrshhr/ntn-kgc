import numpy
import argparse
import os
import chainer
from chainer import cuda
from chainer import training
from chainer.training import extensions

import matplotlib as mpl
mpl.use('Agg')

from models.nn import NN
from models.ntn import NTN
from models.ntnd import NTNd
from models.ntnc import NTNc
from models.ntns import NTNs

from lib import reader, fnameRW


def main():
    parser = argparse.ArgumentParser()

    # CPU or GPU
    parser.add_argument('--gpu', '-g', default=-1, type=int,
                        help='GPU ID (negative value indicates CPU)')

    # Hyper parameters for optimization
    parser.add_argument('--kg_choice', '-c', type=str, default='w',
                        help='KG to train')

    parser.add_argument('--model', '-m', type=str, default='n',
                        help='Model to train')

    parser.add_argument('--weightdecay', '-w', type=float, default=0.0001,
                        help='Coefficient of weight decay')

    parser.add_argument('--learn_rate', '-l', type=float, default=0.1,
                        help='Learning rate')

    parser.add_argument('--batchsize', '-b', type=int, default=1000,
                        help='minibatch size')

    parser.add_argument('--epoch', '-e', default=100, type=int,
                        help='number of epochs to learn')

    # Hyperparameters for a model
    parser.add_argument('--matpro', '-v', type=int, default=1,
                        help='if matrix product is in the model')

    parser.add_argument('--p_dim', '-p', type=int, default=1,
                        help='Dimension of low rank matrices')

    parser.add_argument('--dimension', '-d', type=int, default=100,
                        help='Dimension of embeddings')

    parser.add_argument('--slice_size', '-k', type=int, default=4,
                        help='Dimension of output vector')

    parser.add_argument('--n_nsamp', '-s', type=int, default=10,
                        help='Number of negative samples')

    # Others
    parser.add_argument('--save_period', '-sp', default='50', type=int,
                        help='a number of epochs to save models')

    parser.add_argument('--folder', '-f', default='', type=str,
                        help='Prefix name of an experiment for chainer-logging and trained_model')

    args = parser.parse_args()

    # Numpy or Cupy
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        xp = cuda.cupy
    else:
        xp = numpy

    # Set random seed
    xp.random.seed(100)

    # Data setup
    train, dev, test, n_ent, n_rel, rs2o, ro2s = reader.read(args.kg_choice)

    # Model setup
    params = {'n_ent': n_ent,
              'n_rel': n_rel,
              'n_nsamp': args.n_nsamp,
              'd': args.dimension,
              'k': args.slice_size}

    if args.model == 'n':
        result_dir = 'result_nn'
        model = NN(**params)
    elif args.model == 't':
        if args.matpro == 1:
            result_dir = 'result_ntn'
        else:
            result_dir = 'result_ntn_t'
        params['mp'] = args.matpro
        model = NTN(**params)
    elif args.model == 'd':
        if args.matpro == 1:
            result_dir = 'result_ntnd'
        else:
            result_dir = 'result_ntnd_t'
        params['mp'] = args.matpro
        model = NTNd(**params)
    elif args.model == 'c':
        if args.matpro == 1:
            result_dir = 'result_ntnd'
        else:
            result_dir = 'result_ntnd_t'
        params['mp'] = args.matpro
        model = NTNc(**params)
    elif args.model == 's':
        result_dir = 'result_ntns'
        params['p'] = args.p_dim
        model = NTNs(**params)

    if args.gpu >= 0:
        model.to_gpu()

    # File name setting
    name = fnameRW.fname_make(args)

    # Prepare train iterators
    train = xp.array(train, xp.int32)
    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)

    dev = xp.array(dev, xp.int32)
    dev_iter = chainer.iterators.SerialIterator(dev, args.batchsize, repeat=False, shuffle=False)

    # Optimizer setup
    optimizer = chainer.optimizers.AdaGrad(lr=args.learn_rate)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(args.weightdecay))

    # Trainer setup
    # - Set up a trainer
    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=result_dir)

    # - Add extensions
    dir_name = "trained_model/{}".format(args.folder)
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

    def save_models(n_epoch):
        @training.make_extension(trigger=(n_epoch, 'epoch'))
        def _save_models(trainer):
            chainer.serializers.save_hdf5("{}/{}#c_epoch:{}".format(dir_name, name, updater.epoch), model)
        return _save_models

    def normalize():
        @training.make_extension(trigger=(1, 'epoch'))
        def _normalize(trainer):
            model._normalize()
        return _normalize

    trainer.extend(save_models(args.save_period))
    trainer.extend(normalize())

    # - Evaluate the model with the dev dataset for each epoch
    trainer.extend(extensions.Evaluator(dev_iter, model, device=args.gpu))

    # - Write a log of evaluation statistics for each epoch
    trainer.extend(extensions.LogReport(log_name="{}.log".format(name)))

    # - Save two plot images to the result dir
    if extensions.PlotReport.available():
        trainer.extend(extensions.PlotReport(
                                            ['main/loss', 'validation/main/loss'],
                                            'epoch',
                                            file_name='{}loss.png'.format(name)
                                            ))

    # - Print a progress bar to stdout
    trainer.extend(extensions.ProgressBar())

    # - Run trainer
    trainer.run()


if __name__ == '__main__':
    main()
