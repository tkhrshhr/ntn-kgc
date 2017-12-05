import chainer
import chainer.functions as F
import chainer.links as L
from models.ntn import NTN


class NTNc(NTN):
    def __init__(self, n_ent, n_rel, n_nsamp=1, d=100, k=4):
        NTN.__init__(self, n_ent, n_rel, n_nsamp, d, k)
        with self.init_scope():
            # Set initializer
            u_initializer = chainer.initializers.Uniform(dtype=self.xp.float32)
            # Embeddings
            del self.embed
            self.embed_re = L.EmbedID(n_ent, d, initialW=u_initializer)
            self.embed_im = L.EmbedID(n_ent, d, initialW=u_initializer)
            # Wr
            del self.Wr
            self.wr_re = chainer.Parameter(shape=(n_rel, k, 1, d), initializer=u_initializer)
            self.wr_im = chainer.Parameter(shape=(n_rel, k, 1, d), initializer=u_initializer)
            # Vr
            del self.Vr
            self.Vr = chainer.Parameter(shape=(n_rel, k, 4 * d), initializer=u_initializer)

    def _normalize(self):
        concat = self.xp.concatenate((self.embed_re.W.data, self.embed_im.W.data), axis=1)
        norm = self.xp.linalg.norm(concat, axis=1)
        norm = self.xp.expand_dims(norm, axis=1)
        self.embed_re.W.data = self.embed_re.W.data / norm
        self.embed_im.W.data = self.embed_im.W.data / norm

    def get_g(self, r_ids, s_ids, o_ids):
        s_batch = len(r_ids)
        # Get embeddings
        s_re = self.embed_re(s_ids)
        s_im = self.embed_im(s_ids)
        s_re_r = F.reshape(F.tile(s_re, (1, self.k)), (s_batch * self.k, 1, self.d))
        s_im_r = F.reshape(F.tile(s_im, (1, self.k)), (s_batch * self.k, 1, self.d))

        o_re = self.embed_re(o_ids)
        o_im = self.embed_im(o_ids)
        o_re_r = F.reshape(F.tile(o_re, (1, self.k)), (s_batch * self.k, 1, self.d))
        o_im_r = F.reshape(F.tile(o_im, (1, self.k)), (s_batch * self.k, 1, self.d))

        so = F.concat((s_re, s_im, o_re, o_im), axis=1)
        # W
        w_re = F.reshape(self.wr_re[r_ids], (s_batch * self.k, 1, self.d))
        w_im = F.reshape(self.wr_im[r_ids], (s_batch * self.k, 1, self.d))
        # V
        V = self.Vr[r_ids]
        # b
        b = self.br[r_ids]
        # u
        u = self.ur[r_ids]

        # calculate each term
        # sWo
        s_riri = F.stack((s_re_r, s_im_r, s_re_r, s_im_r), axis=0)
        o_riir = F.stack((o_re_r, o_im_r, o_im_r, o_re_r), axis=0)
        w_rrii = F.stack((w_re, w_re, w_im, w_im), axis=0)
        sWo__ = F.sum(w_rrii * s_riri * o_riir, axis=(2, 3))
        sWo_ = sWo__[0] + sWo__[1] + sWo__[2] - sWo__[3]
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

    def _mold_for_neg(self, s_ids, o_ids, cs_ids, co_ids):
        # Get embeddings
        s_re = self.embed_re(s_ids)
        s_im = self.embed_im(s_ids)
        s_re_r = F.reshape(F.tile(s_re, (1, self.k)), (self.s_batch * self.k, 1, self.d))
        s_im_r = F.reshape(F.tile(s_im, (1, self.k)), (self.s_batch * self.k, 1, self.d))

        o_re = self.embed_re(o_ids)
        o_im = self.embed_im(o_ids)
        o_re_r = F.reshape(F.tile(o_re, (1, self.k)), (self.s_batch * self.k, 1, self.d))
        o_im_r = F.reshape(F.tile(o_im, (1, self.k)), (self.s_batch * self.k, 1, self.d))

        cs_re = self.embed_re(cs_ids)
        cs_im = self.embed_im(cs_ids)
        cs_re_r = F.reshape(F.tile(cs_re, (1, self.k)), (self.n_cs * self.s_batch * self.k, 1, self.d))
        cs_im_r = F.reshape(F.tile(cs_im, (1, self.k)), (self.n_cs * self.s_batch * self.k, 1, self.d))

        co_re = self.embed_re(co_ids)
        co_im = self.embed_im(co_ids)
        co_re_r = F.reshape(F.tile(co_re, (1, self.k)), (self.n_co * self.s_batch * self.k, 1, self.d))
        co_im_r = F.reshape(F.tile(co_im, (1, self.k)), (self.n_co * self.s_batch * self.k, 1, self.d))

        # - concat ce1e2  and e1ce2
        s_re_t_half = F.tile(s_re, (self.n_co, 1))
        s_im_t_half = F.tile(s_im, (self.n_co, 1))
        o_re_t_half = F.tile(o_re, (self.n_cs, 1))
        o_im_t_half = F.tile(o_im, (self.n_cs, 1))
        cso = F.concat((cs_re, cs_im, o_re_t_half, o_im_t_half), axis=1)
        sco = F.concat((s_re_t_half, s_im_t_half, co_re, co_im), axis=1)
        csco = F.concat((cso, sco), axis=0)

        return (s_re_r, s_im_r, o_re_r, o_im_r, cs_re_r, cs_im_r, co_re_r, co_im_r, csco)

    def _get_neg_g(self, r_ids, s_re_r, s_im_r, o_re_r, o_im_r, cs_re_r, cs_im_r, co_re_r, co_im_r, csco):
        # W
        w_re = F.reshape(self.wr_re[r_ids], (self.s_batch * self.k, 1, self.d))
        w_im = F.reshape(self.wr_re[r_ids], (self.s_batch * self.k, 1, self.d))
        # V
        V = self.Vr[r_ids]
        V_t = F.tile(V, (self.n_nsamp, 1, 1))
        # b
        b = self.br[r_ids]
        b_t = F.tile(b, (self.n_nsamp, 1))
        # u
        u = self.ur[r_ids]
        u_t = F.tile(u, (self.n_nsamp, 1))

        # Stack vectors
        s_riri = F.stack((s_re_r, s_im_r, o_re_r, o_im_r), axis=0)
        o_riir = F.stack((s_re_r, s_im_r, o_im_r, o_re_r), axis=0)
        w_rrii = F.stack((w_re, w_re, w_im, w_im), axis=0)

        cs_riri = F.stack((cs_re_r, cs_im_r, cs_re_r, cs_im_r), axis=0)
        co_riir = F.stack((co_re_r, co_im_r, co_im_r, co_re_r), axis=0)

        # calculate each term
        sW = s_riri * w_rrii
        Wo = w_rrii * o_riir

        sW_t = F.tile(sW, (1, self.n_co, 1, 1))
        Wo_t = F.tile(Wo, (1, self.n_cs, 1, 1))

        csWo__ = F.sum(cs_riri * Wo_t, axis=(2, 3))
        csWo_ = csWo__[0] + csWo__[1] + csWo__[2] - csWo__[3]
        csWo = F.reshape(csWo_, (self.s_batch * self.n_cs, self.k))

        sWco__ = F.sum(sW_t * co_riir, axis=(2, 3))
        sWco_ = sWco__[0] + sWco__[1] + sWco__[2] - sWco__[3]
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
