#!/usr/bin/env bash

# 显式加载 Conda 的基础配置
# 注意：把下面的路径改成你自己的conda路径（用 which conda 找）
# source ~/anaconda3/etc/profile.d/conda.sh
# 如果是 miniconda
source ~/miniconda3/etc/profile.d/conda.sh

conda activate deepseek-ocr

# 进入目录
cd DeepSeek-OCR-master/DeepSeek-OCR-vllm || exit

# 运行 Python 脚本
python run_dpsk_ocr_pdf_batch_Bo.py