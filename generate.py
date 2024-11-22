import argparse
import tiktoken
import time
import mlx.core as mx

from mlx.utils import tree_unflatten, tree_flatten
from model import GPT, GPTConfig

GPT_MODELS = {
    "gpt2": dict(n_layer=12, n_head=12, n_embd=768),
    "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),
    "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),
    "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),
}


def load_weights(gpt_model: GPT, weights: dict):
    gpt_model.update(tree_unflatten(list(weights.items())))
    mx.eval(gpt_model.parameters())
    nparams = sum(x.size for k, x in tree_flatten(gpt_model.parameters()))
    print(f"Loaded GPT-2 with {nparams / 1e6:.3f} M parameters")


def load_model(model_path: str, model_name: str):
    config_args = GPT_MODELS[model_name]

    config_args["vocab_size"] = 50257
    config_args["block_size"] = 1024
    config_args["bias"] = True

    config = GPTConfig(**config_args)

    gpt_model = GPT(config)
    weights = mx.load(model_path, format="safetensors")
    load_weights(gpt_model, weights)

    return gpt_model


def generate_text(prompt: str, model: GPT):
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)
    start_ids = encode(prompt)

    x = mx.array([start_ids], dtype=mx.uint32)

    print(prompt, end="")
    tokens = []
    start = time.time()
    for token in model.generate(x, max_new_tokens=256):
        tok = token.item()
        tokens.append(tok)
        print(decode([tok]), end="", flush=True)
    end = time.time()
    print("")
    print("---------------")
    print(
        f"time: {end - start:.3f} s, tokens per second: {len(tokens) / (end - start)}"
    )
    print("---------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text from GPT-2")

    parser.add_argument(
        "--model_name",
        type=str,
        help="The name of a pre-trained GPT-2 model to use",
        required=True,
        choices=GPT_MODELS.keys()
    )

    parser.add_argument(
        "--model_path", type=str, help="path to safetensors model", required=True
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Kyle lowry ain't no spot up shooter",
        help="The prompt to generate text from",
    )

    args = parser.parse_args()
    model = load_model(args.model_path, args.model_name)
    generate_text(args.prompt, model)
