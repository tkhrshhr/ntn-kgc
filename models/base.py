import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L

from chainer import Variable as chVar


class Base(chainer.Chain):
    def __init__(self, n_ent, n_rel, n_nsamp=1, d=100, k=4):
        super().__init__()
        with self.init_scope():
            # Set initializer
            u_initializer = chainer.initializers.Uniform(dtype=self.xp.float32)

            # Entity vectors
            self.embed = L.EmbedID(n_ent, d)

            # Neural Layer
            # - Vr
            self.Vr = chainer.Parameter(shape=(n_rel, k, 2 * d), initializer=u_initializer)

            # - br
            self.br = chainer.Parameter(shape=(n_rel, k), initializer=u_initializer)

            # - ur
            self.ur = chainer.Parameter(shape=(n_rel, k), initializer=u_initializer)

        # Hyper Parameters
        self.n_ent = n_ent
        self.n_rel = n_rel
        self.n_nsamp = n_nsamp
        self.d = d
        self.k = k

    def _normalize(self):
        norm = self.xp.linalg.norm(self.embed.W.data)
        norm = self.xp.expand_dims(norm, axis=1)
        self.embed.W.data = self.embed.W.data / norm

    def get_g(self, r_ids, s_ids, o_ids):
        raise NotImplementedError

    def _mold_for_neg(self, s_ids, o_ids, cs_ids, co_ids):
        raise NotImplementedError

    def _get_neg_g(self, r_ids, *inputs):
        raise NotImplementedError

    def _get_loss(self, batch):
        # Get batch size
        self.s_batch = len(batch)
        # Get negative sample size of S and O
        self.n_cs = np.random.binomial(n=self.n_nsamp - 2, p=0.5) + 1
        self.n_co = self.n_nsamp - self.n_cs
        # Get IDs
        r_ids = batch[:, 0]
        s_ids = batch[:, 1]
        o_ids = batch[:, 2]
        cs_ids = self.xp.random.randint(0, self.n_ent, self.s_batch * self.n_cs).astype(self.xp.int32)
        co_ids = self.xp.random.randint(0, self.n_ent, self.s_batch * self.n_co).astype(self.xp.int32)

        # Get positive g scores and tile them
        pos_g = self.get_g(r_ids, s_ids, o_ids)
        pos_g_t = F.tile(pos_g, (self.n_nsamp, 1))

        # Get negative g scores
        molds = self._mold_for_neg(s_ids, o_ids, cs_ids, co_ids)
        neg_g = self._get_neg_g(r_ids, *molds)

        # Set mergins 1 as chainer.Variable
        margin = chVar(self.xp.ones((self.s_batch * self.n_nsamp, 1), dtype=self.xp.float32))
        # Set 0 as chainer.Variable
        zero = chVar(self.xp.zeros((self.s_batch * self.n_nsamp, 1), dtype=self.xp.float32))

        # Calculate loss
        loss_ = F.maximum(zero, margin - pos_g_t + neg_g)
        loss = F.sum(loss_)

        return loss

    def __call__(self, batch):
        return self._get_loss(batch)
