import chainer.functions as F
from models.base import Base


class NN(Base):
    def __init__(self, **argvs):
        Base.__init__(self, **argvs)

    def get_g(self, r_ids, s_ids, o_ids):
        s_batch = len(r_ids)
        # Get embeddings
        s = self.embed(s_ids)
        o = self.embed(o_ids)
        so = F.concat((s, o), axis=1)
        # V
        V = self.Vr[r_ids]
        # b
        b = self.br[r_ids]
        # u
        u = self.ur[r_ids]

        # calculate each term
        # - Vso
        Vso_ = F.matmul(V, F.expand_dims(so, axis=1), transb=True)
        Vso = F.reshape(Vso_, (s_batch, self.k))

        # sum up terms
        preact = Vso + b

        activated = F.tanh(preact)

        g_score_ = F.sum(u * activated, axis=1)
        g_score = F.reshape(g_score_, (s_batch, 1))

        return g_score

    def _mold_for_neg(self, s_ids, o_ids, cs_ids, co_ids):
        # Get embeddings
        s = self.embed(s_ids)
        o = self.embed(o_ids)
        cs = self.embed(cs_ids)
        co = self.embed(co_ids)
        # - concat ce1e2  and e1ce2
        s_t_half = F.tile(s, (self.n_co, 1))
        o_t_half = F.tile(o, (self.n_cs, 1))
        cso = F.concat((cs, o_t_half), axis=1)
        sco = F.concat((s_t_half, co), axis=1)
        csco = F.concat((cso, sco), axis=0)

        return (csco,)

    def _get_neg_g(self, r_ids, csco):
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
        # - Vso
        Vso_ = F.matmul(V_t, F.expand_dims(csco, axis=1), transb=True)
        Vso = F.reshape(Vso_, (self.s_batch * self.n_nsamp, self.k))

        # sum up terms
        preact = Vso + b_t
        activated = F.tanh(preact)

        g_score_ = F.sum(u_t * activated, axis=1)
        g_score = F.reshape(g_score_, (self.s_batch * self.n_nsamp, 1))

        return g_score
