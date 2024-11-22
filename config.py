from dataclasses import dataclass

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True

GPT_MODELS = {
    "gpt2": dict(n_layer=12, n_head=12, n_embd=768),
    "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),
    "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),
    "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),
}