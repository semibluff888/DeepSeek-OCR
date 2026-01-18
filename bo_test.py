from transformers import AutoModel, AutoTokenizer
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
model_name = 'model'
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
# 如果安装了flash-attn，通过设置进行加速推理(wsl的linux已安装flash-attn)
# model = AutoModel.from_pretrained(model_name, _attn_implementation='flash_attention_2', trust_remote_code=True, use_safetensors=True)
# 如果没有安装flash-attn，无需设置(windows环境尚未安装flash-attn)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True, use_safetensors=True)
model = model.eval().cuda().to(torch.bfloat16)
prompt = "<image>\n<|grounding|>Convert the document to markdown. "
# image_file = 'test_data/p1.jpg'
# image_file = '身份证.jpg'
image_file = './input/npl.png' #  work on both Windows and Linux in Python! 但因为是相对路径,脚本必须在当前目录下运行.
output_path = 'output'
res = model.infer(tokenizer, prompt=prompt, image_file=image_file, output_path = output_path, base_size = 1024, image_size = 640, crop_mode=True, save_results = True, test_compress = True)
