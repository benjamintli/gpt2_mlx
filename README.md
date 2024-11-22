# gpt2_mlx
Running GPT2 on a Mac using Apple MLX

## Getting Started
1. Download the [safetensors model](https://huggingface.co/openai-community/gpt2/tree/main) for gpt2 from huggingface
2. Run the weight conversion script. For some reason we have to transpose some of these layers 🤪
```
python3 convert_weights.py --weights_path model.safetensors --model_name {gpt2, gpt2-xl, ...}     
```
3. run the generation script
```
python3 generate.py --model_name gpt2 --model_path gpt2.safetensors --prompt <text>
```