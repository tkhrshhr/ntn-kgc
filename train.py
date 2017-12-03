import numpy
import argparse
import datetime
import chainer
from chainer import cuda
from chainer import training
from chainer.training import extensions
from chainer import links as L

import matplotlib as mpl
mpl.use('Agg')

import reader

from models.nn import NN
from models.ntn import NTN
from models.ntnd import NTNd
from models.ntnc import NTNc
from models.ntns import NTNs


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--kg_choice', '-c', type=str, default='w',
                        help='KG to train')

    parser.add_argument('--model', '-m', type=str, default='n',
                        help='Model to train')

    parser.add_argument('--p_dim', '-p', type=int, default=1,
                        help='Dimension p')

    parser.add_argument('--epoch', '-e', default=100, type=int,
                        help='number of epochs to learn')

    parser.add_argument('--dimension', '-d', type=int, default=100,
                        help='Dimension of embeddings')

    parser.add_argument('--slice_size', '-k', type=int, default=4,
                        help='Dimension of output vector')

    parser.add_argument('--n_nsamp', '-s', type=int, default=5,
                        help='Number of negative samples')

    parser.add_argument('--weightdecay', '-w', type=float, default=0.0001,
                        help='Coefficient of weight decay')

    parser.add_argument('--learn_rate', '-l', type=float, default=0.1,
                        help='Learning rate')

    parser.add_argument('--batchsize', '-b', type=int, default=25,
                        help='minibatch size')

    parser.add_argument('--gpu', '-g', default=-1, type=int,
                        help='GPU ID (negative value indicates CPU)')

    parser.add_argument('--nmodifier', '-n', default='',
                        help='Result name modifier')

    parser.add_argument('--test', dest='test', action='store_true')

    parser.set_defaults(test=False)
    args = parser.parse_args()

    # Log file name setting
    today = datetime.date.today()
    month = today.month
    day = today.day
    resultname = "{}{}{}-b{}d{}w{}".format(month, day, args.nmodifier, args.batchsize, args.dimension, args.weightdecay)

    # Data setup
    train, dev, test, n_ent, n_rel = reader.read(args.kg_choice)

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        xp = cuda.cupy
    else:
        xp = numpy

    # - Prepare train iterators
    train = xp.array(train, xp.int32)
    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)

    # Model setup
    params = {'n_ent': n_ent, 'n_rel': n_rel, 'n_nsamp': args.n_nsamp, 'd': args.dimension, 'k': args.slice_size}
    if args.model == 'n':
        result_dir = 'result_rnn'
        model = NN(**params)
    elif args.model == 't':
        result_dir = 'result_ntn'
        model = NTN(**params)
    elif args.model == 'd':
        result_dir = 'result_ntnd'
        model = NTNd(**params)
    elif args.model == 'c':
        result_dir = 'result_ntnc'
        model = NTNc(**params)
    elif args.model == 's':
        result_dir = 'result_ntns'
        params['p'] = args.p_dim
        model = NTNs(**params)

    if args.gpu >= 0:
        model.to_gpu()

    # Optimizer setup
    optimizer = chainer.optimizers.AdaGrad(lr=args.learn_rate)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(args.weightdecay))

    # Trainer setup
    # - Set up a trainer
    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=result_dir)

    # - Evaluate the model with the test dataset for each epoch

    # - Write a log of evaluation statistics for each epoch
    trainer.extend(extensions.LogReport(log_name="{}log".format(resultname)))

    # - Save two plot images to the result dir
    if extensions.PlotReport.available():
        trainer.extend(
            extensions.PlotReport(['main/loss', 'validation/main/loss'],
                                  'epoch', file_name='{}loss.png'.format(resultname)))
        trainer.extend(
            extensions.PlotReport(
                ['main/accuracy', 'validation/main/accuracy'],
                'epoch', file_name='{}accuracy.png'.format(resultname)))

    # - Print a progress bar to stdout
    trainer.extend(extensions.ProgressBar())

    # - Run trainer
    trainer.run()

    # - Save model
    if args.model != 's':
        model_name = '{}_d{}k{}s{}'.format(args.model, args.dimension, args.slice_size, args.n_nsamp)
    elif args.model == 's':
        model_name = '{}_d{}k{}s{}p{}'.format(args.model, args.dimension, args.slice_size, args.n_nsamp, args.p_dim)

    chainer.serializers.save_hdf5("trained_model/{}".format(model_name), model)


if __name__ == '__main__':
    main()
