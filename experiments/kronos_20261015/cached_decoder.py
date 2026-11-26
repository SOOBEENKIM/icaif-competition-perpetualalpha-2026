"""Inference-only KV cache around the unchanged official pretrained modules.

The upstream autoregressive sampler and tokenizer stay unchanged. This adapter
reuses causal transformer keys/values within ONE forecast; caches never cross
origins. Full-reference equality is checked before performance evaluation.
"""
import torch
import torch.nn.functional as F


class CachedDecoder:
    def __init__(self, model, copies=10):
        self.model = model
        self.copies = copies
        self.original = model.decode_s1
        self.reset()

    def reset(self):
        self.keys = []
        self.values = []
        self.context = None
        self.tokens = None
        self.stamps = None

    @staticmethod
    def rope(rotary, x, offset):
        cos, sin = rotary._update_cos_sin_cache(x, offset + x.shape[-2])
        cos, sin = cos[..., offset:, :], sin[..., offset:, :]
        return x * cos + rotary._rotate_half(x) * sin

    def __call__(self, s1_ids, s2_ids, stamp=None, padding_mask=None):
        if self.model.training or padding_mask is not None or stamp is None:
            raise ValueError('Cache supports eval mode, complete unpadded inputs only')
        length = s1_ids.shape[1]
        if self.tokens is None:
            return self.prefill(s1_ids, s2_ids, stamp)
        previous = self.tokens[0].shape[1]
        if (length != previous + 1 or not torch.equal(s1_ids[:, :-1], self.tokens[0])
                or not torch.equal(s2_ids[:, :-1], self.tokens[1])
                or not torch.equal(stamp[:, :-1], self.stamps)):
            raise ValueError('Cache context changed; reset at each prediction origin')
        m = self.model
        x = m.embedding([s1_ids[:, -1:], s2_ids[:, -1:]]) + m.time_emb(stamp[:, -1:])
        for n, layer in enumerate(m.transformer):
            a = layer.self_attn
            xx = layer.norm1(x)
            shape = (len(x), 1, a.n_heads, a.head_dim)
            q = self.rope(a.rotary, a.q_proj(xx).view(shape).transpose(1, 2), previous)
            k = self.rope(a.rotary, a.k_proj(xx).view(shape).transpose(1, 2), previous)
            v = a.v_proj(xx).view(shape).transpose(1, 2)
            k = torch.cat([self.keys[n], k], dim=2)
            v = torch.cat([self.values[n], v], dim=2)
            self.keys[n], self.values[n] = k, v
            # A single last-token query may attend to every cached key.
            y = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
            x = x + a.out_proj(y.transpose(1, 2).contiguous().view(len(x), 1, m.d_model))
            x = x + layer.ffn(layer.norm2(x))
        x = m.norm(x)
        self.context = torch.cat([self.context, x], dim=1)
        self.tokens = (s1_ids.clone(), s2_ids.clone())
        self.stamps = stamp.clone()
        return m.head(x), self.context

    def prefill(self, s1, s2, stamp):
        m = self.model
        copies = self.copies
        if len(s1) % copies:
            raise ValueError('Unexpected Monte Carlo batch')
        for x in (s1, s2, stamp):
            if not torch.equal(x, x[::copies].repeat_interleave(copies, dim=0)):
                raise ValueError('Initial sample contexts are not identical')
        x = m.embedding([s1[::copies], s2[::copies]]) + m.time_emb(stamp[::copies])
        for layer in m.transformer:
            a = layer.self_attn
            xx = layer.norm1(x)
            shape = (len(x), x.shape[1], a.n_heads, a.head_dim)
            q = a.q_proj(xx).view(shape).transpose(1, 2)
            k = a.k_proj(xx).view(shape).transpose(1, 2)
            v = a.v_proj(xx).view(shape).transpose(1, 2)
            q, k = a.rotary(q, k)
            self.keys.append(k.repeat_interleave(copies, dim=0))
            self.values.append(v.repeat_interleave(copies, dim=0))
            y = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)
            x = x + a.out_proj(y.transpose(1, 2).contiguous().view(x.shape))
            x = x + layer.ffn(layer.norm2(x))
        x = m.norm(x).repeat_interleave(copies, dim=0)
        self.context = x
        self.tokens = (s1.clone(), s2.clone())
        self.stamps = stamp.clone()
        return m.head(x), x


def install(predictor):
    cache = CachedDecoder(predictor.model)
    predictor.model.decode_s1 = cache
    return cache
