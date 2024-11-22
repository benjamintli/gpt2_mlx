import argparse
import os
from safetensors import safe_open
from safetensors.torch import save_file

from config import GPT_MODELS


def transpose_specific_layers(state_dict: dict) -> dict:
    print("transposing...")
    layers_to_transpose = [
        "attn.c_attn.weight",
        "attn.c_proj.weight",
        "mlp.c_fc.weight",
        "mlp.c_proj.weight",
    ]

    for key in state_dict.keys():
        if any(key.endswith(suffix) for suffix in layers_to_transpose):
            state_dict[key] = state_dict[key].T
    return state_dict


def transpose_and_save(input_path, model_name):
    state_dict = {}
    print("loading safetensors dict")
    with safe_open(input_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            state_dict[key] = f.get_tensor(key)
    state_dict_transposed = transpose_specific_layers(state_dict)
    for layer in state_dict_transposed.keys():
        state_dict_transposed[layer] = state_dict_transposed[layer].contiguous()
    input_dir = os.path.dirname(input_path)
    output_path = os.path.join(input_dir, f"{model_name}.safetensors")
    save_file(state_dict_transposed, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert GPT-2 safetensors weights")

    parser.add_argument(
        "--weights_path",
        type=str,
        default="model.safetensors",
        help="Path to safetensors weights",
        required=True,
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt2",
        help="The name of the model",
        required=True,
        choices=GPT_MODELS.keys(),
    )

    args = parser.parse_args()
    print("starting conversion...")
    transpose_and_save(args.weights_path, args.model_name)
    print("conversion finished")