import chainer
import chainer.functions as F
from models.ntn import NTN


class NTNs(NTN):
    def __init__(self, n_ent, n_rel, n_nsamp=1, d=100, k=4, p=1):
        NTN.__init__(self, n_ent, n_rel, n_nsamp, d, k)
        with self.init_scope():
            # Set initializer
            u_initializer = chainer.initializers.Uniform(dtype=self.xp.float32)
            # Wr
            del self.Wr
            self.Sr = chainer.Parameter(shape=(n_rel, k, d, p), initializer=u_initializer)
            self.Tr = chainer.Parameter(shape=(n_rel, k, p, d), initializer=u_initializer)
        self.p = p

    def get_g(self, r_ids, s_ids, o_ids):
        s_batch = len(r_ids)
        # Get embeddings
        s = self.embed(s_ids)
        s_r = F.reshape(F.tile(s, (1, self.k)), (s_batch * self.k, 1, self.d))
        o = self.embed(o_ids)
        o_r = F.reshape(F.tile(o, (1, self.k)), (s_batch * self.k, 1, self.d))
        so = F.concat((s, o), axis=1)
        # W
        S = F.reshape(self.Sr[r_ids], (s_batch * self.k, self.d, self.p))
        T = F.reshape(self.Tr[r_ids], (s_batch * self.k, self.p, self.d))
        # V
        V = self.Vr[r_ids]
        # b
        b = self.br[r_ids]
        # u
        u = self.ur[r_ids]

        # calculate each term
        # sWo
        sS = F.matmul(s_r, S)
        sST = F.matmul(sS, T)
        sWo_ = F.matmul(sST, o_r, transb=True)
        sWo = F.reshape(sWo_, (s_batch, self.k))

        # - Vso
        Vso_ = F.matmul(V, F.expand_dims(so, axis=1), transb=True)
        Vso = F.reshape(Vso_, (s_batch, self.k))

        # sum up terms
        preact = sWo + Vso + b

        activated = F.tanh(preact)

        g_score_ = F.sum(u * activated, axis=1)
        g_score = F.reshape(g_score_, (s_batch, 1))

        return g_score

    def _get_neg_g(self, r_ids, s_r, o_r, cs_r, co_r, csco):
        # W
        S = F.reshape(self.Sr[r_ids], (self.s_batch * self.k, self.d, self.p))
        T = F.reshape(self.Tr[r_ids], (self.s_batch * self.k, self.p, self.d))
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
        sS = F.matmul(s_r, S)
        sW = F.matmul(sS, T)
        To = F.matmul(T, o_r, transb=True)
        Wo = F.matmul(S, To)
        sW_t = F.tile(sW, (self.n_co, 1, 1))
        Wo_t = F.tile(Wo, (self.n_cs, 1, 1))

        csWo_ = F.matmul(cs_r, Wo_t)
        csWo = F.reshape(csWo_, (self.s_batch * self.n_cs, self.k))
        sWco_ = F.batch_matmul(sW_t, co_r, transb=True)
        sWco = F.reshape(sWco_, (self.s_batch * self.n_co, self.k))

        sWo = F.concat((csWo, sWco), axis=0)

        # - Vso
        Vso_ = F.matmul(V_t, F.expand_dims(csco, axis=1), transb=True)
        Vso = F.reshape(Vso_, (self.s_batch * self.n_nsamp, self.k))

        # sum up terms
        preact = sWo + Vso + b_t
        activated = F.tanh(preact)

        g_score_ = F.sum(u_t * activated, axis=1)
        g_score = F.reshape(g_score_, (self.s_batch * self.n_nsamp, 1))

        return g_score
