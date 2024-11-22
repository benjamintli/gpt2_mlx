from dataclasses import dataclass
import math

import mlx.core as mx
import mlx.nn as nn


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head

        self.scale = self.head_dim**-0.5

    def __call__(self, x: mx.array, mask=None, cache=None):
        batch, sequence_length, _ = x.shape  # batch size, sequence length
        qkv = self.c_attn(x)
        q, k, v = mx.split(qkv, 3, axis=-1)
        k = k.reshape(batch, sequence_length, self.n_head, -1).transpose(0, 2, 1, 3)
        q = q.reshape(batch, sequence_length, self.n_head, -1).transpose(0, 2, 1, 3)
        v = v.reshape(batch, sequence_length, self.n_head, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            key_cache, value_cache = cache
            k = mx.concatenate([key_cache, k], axis=2)
            v = mx.concatenate([value_cache, v], axis=2)

        output = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(batch, sequence_length, -1)
        return self.c_proj(output), (k, v)


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()

        self.n_embd = config.n_embd
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def __call__(self, x: mx.array):
        return self.c_proj(nn.gelu_fast_approx(self.c_fc(x)))


class TransformerBlock(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.ln_2 = nn.LayerNorm(config.n_embd)

    def __call__(self, x: mx.array, mask=None, cache=None):
        r, cache = self.attn(self.ln_1(x), mask, cache)
        h = x + r
        r = self.mlp(self.ln_2(h))
        out = h + r
        return out, cache


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.h = [TransformerBlock(config) for _ in range(config.n_layer)]
        self.ln_f = nn.LayerNorm(config.n_embd, affine=config.bias)

    def _forward_block(
        self, x: mx.array, positions: mx.array, mask=None, cache=None, build_cache=False
    ):
        tok_emb = self.wte(x)  # token embeddings of shape (b, t, n_embd)
        pos_emb = self.wpe(positions)  # position embeddings of shape (t, n_embd)
        x = self.drop(tok_emb + pos_emb)

        kv_cache = []
        if cache is not None:
            for i in range(len(cache)):
                x, cache[i] = self.h[i](x, mask=None, cache=cache[i])
        else:
            for block in self.h:
                x, curr_cache = block(x, mask=mask)
                if build_cache:
                    kv_cache.append(curr_cache)

        x = self.ln_f(x)
        return x, kv_cache if build_cache else cache


    def _create_causal_mask(self, length: int):
        mask = nn.MultiHeadAttention.create_additive_causal_mask(length)
        return mask.astype(self.wte.weight.dtype)

    def _sample_next_token(self, x, temperature):
        logits = mx.expand_dims(x[:, -1], axis=0) @ self.wte.weight.T
        y = logits[:, -1, :]
        y = mx.random.categorical(y * (1 / temperature))
        return y

    def generate(self, x: mx.array, max_new_tokens=256, temperature=0.8):
        _, t = x.shape
        pos = mx.arange(0, t, 1, dtype=x.dtype)
        mask = self._create_causal_mask(t)
        x, cache = self._forward_block(x, pos, mask=mask, build_cache=True)
        y = self._sample_next_token(x, temperature)
        position = t
        yield y

        for _ in range(max_new_tokens):
            position += 1
            x = y[:, None]
            x, cache = self._forward_block(x, position, cache=cache)
            y = self._sample_next_token(x, temperature)
            yield y
