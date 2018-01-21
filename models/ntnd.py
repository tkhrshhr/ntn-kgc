import chainer
import chainer.functions as F
from models.ntn import NTN


class NTNd(NTN):
    def __init__(self, n_ent, n_rel, n_nsamp=1, d=100, k=4, mp=1):
        NTN.__init__(self, n_ent, n_rel, n_nsamp, d, k)
        with self.init_scope():
            # Set initializer
            u_initializer = chainer.initializers.Uniform(dtype=self.xp.float32)
            # Wr
            del self.Wr
            self.wr = chainer.Parameter(shape=(n_rel, k, 1, d), initializer=u_initializer)

        self.mp = mp

    def get_g(self, r_ids, s_ids, o_ids):
        s_batch = len(r_ids)
        # Get embeddings
        s = self.embed(s_ids)
        s_r = F.reshape(F.tile(s, (1, self.k)), (s_batch * self.k, 1, self.d))
        o = self.embed(o_ids)
        o_r = F.reshape(F.tile(o, (1, self.k)), (s_batch * self.k, 1, self.d))
        so = F.concat((s, o), axis=1)
        # W
        w = F.reshape(self.wr[r_ids], (s_batch * self.k, 1, self.d))
        # V
        V = self.Vr[r_ids]
        # b
        b = self.br[r_ids]
        # u
        u = self.ur[r_ids]

        # calculate each term
        # sWo
        sWo_ = F.sum(s_r * w * o_r, axis=(1, 2))
        sWo = F.reshape(sWo_, (s_batch, self.k))
        # - Vso
        Vso_ = F.matmul(V, F.expand_dims(so, axis=1), transb=True)
        Vso = F.reshape(Vso_, (s_batch, self.k))

        # sum up terms
        if self.mp == 1:
            preact = sWo + Vso + b
        elif self.mp == 0:
            preact = sWo + b

        activated = F.tanh(preact)

        g_score_ = F.sum(u * activated, axis=1)
        g_score = F.reshape(g_score_, (s_batch, 1))

        return g_score

    def _get_neg_g(self, r_ids, s_r, o_r, cs_r, co_r, csco):
        # W
        w = F.reshape(self.wr[r_ids], (self.s_batch * self.k, 1, self.d))
        # V
        V = self.Vr[r_ids]
        V_t = F.tile(V, (self.n_nsamp, 1, 1))
        # b
        b = self.br[r_ids]
        b_t = F.tile(b, (self.n_nsamp, 1))
        # u
        u = self.ur[r_ids]
        u_t = F.tile(u, (self.n_nsamp, 1))

        # calculate each term
        sW = s_r * w
        Wo = w * o_r

        sW_t = F.tile(sW, (self.n_co, 1, 1))
        Wo_t = F.tile(Wo, (self.n_cs, 1, 1))

        csWo_ = F.sum(cs_r * Wo_t, axis=(1, 2))
        csWo = F.reshape(csWo_, (self.s_batch * self.n_cs, self.k))
        sWco_ = F.sum(sW_t * co_r, axis=(1, 2))
        sWco = F.reshape(sWco_, (self.s_batch * self.n_co, self.k))

        sWo = F.concat((csWo, sWco), axis=0)

        # - Vso
        Vso_ = F.matmul(V_t, F.expand_dims(csco, axis=1), transb=True)
        Vso = F.reshape(Vso_, (self.s_batch * self.n_nsamp, self.k))

        # sum up terms
        if self.mp == 1:
            preact = sWo + Vso + b_t
        if self.mp == 0:
            preact = sWo + b_t

        activated = F.tanh(preact)

        g_score_ = F.sum(u_t * activated, axis=1)
        g_score = F.reshape(g_score_, (self.s_batch * self.n_nsamp, 1))

        return g_score
