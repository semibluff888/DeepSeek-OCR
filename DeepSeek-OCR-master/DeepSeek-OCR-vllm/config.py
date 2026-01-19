from pathlib import Path

# Project root is 3 levels up from this config file
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# TODO: change modes
# Tiny: base_size = 512, image_size = 512, crop_mode = False
# Small: base_size = 640, image_size = 640, crop_mode = False
# Base: base_size = 1024, image_size = 1024, crop_mode = False
# Large: base_size = 1280, image_size = 1280, crop_mode = False
# Gundam: base_size = 1024, image_size = 640, crop_mode = True

BASE_SIZE = 1024
IMAGE_SIZE = 640
CROP_MODE = True
MIN_CROPS= 2
MAX_CROPS= 6 # max:9; If your GPU memory is small, it is recommended to set it to 6.
MAX_CONCURRENCY = 100 # If you have limited GPU memory, lower the concurrency count.
NUM_WORKERS = 64 # image pre-process (resize/padding) workers 
PRINT_NUM_VIS_TOKENS = False
SKIP_REPEAT = True
# MODEL_PATH = 'deepseek-ai/DeepSeek-OCR' # change to your model path
# MODEL_PATH = str(PROJECT_ROOT / 'model')  # Bo: 跨平台写法(绝对路径)
# Bo(2026-01-19): WSL(linux)环境下的模型路径. 之前发现wsl加载模型非常慢，问了AI后建议放到linux路径下，详细参考下面的注释
MODEL_PATH = "/home/bo/models/deepseek-ocr"

'''
问了claude opus 4.5，可以查看antigravity历史agent会话:
Optimize Model Loading Speed
或者gemini会话链接:https://gemini.google.com/app/7253b451ec8ce2e5
The Core Issue: WSL Filesystem Performance
You are storing your model on a Windows drive: model='/mnt/j/Project/DeepSeek-OCR/model'
In WSL2 (Windows Subsystem for Linux), there are two types of file systems:
1. The Native Linux System (e.g., /home/username/): Extremely fast.
2. The Windows Mounted System (e.g., /mnt/c/, /mnt/j/): Very slow when accessed from Linux.
Because your model is on the J: drive (Windows), WSL has to use a translation protocol (9P protocol) to read the data. 
DeepSeek-OCR (running on the vLLM engine shown in your logs) uses memory mapping to load weights. Memory mapping over
 this cross-OS translation layer is notoriously inefficient, causing the 6GB read to crawl.
'''

# Bo: 路径写法跨平台对比**********************************************************
# # ✅ 经过研究,下面这个写法在 Python 中，Windows/Linux 其实也都能用 (相对路径)：
# INPUT_PATH = '../../../npl.png'      # 正斜杠（推荐）

# # Windows only 上三层路径写法, 反斜杠（仅 Windows，需要转义）
# MODEL_PATH = '..\\..\\..\\model'
# INPUT_PATH = '..\\..\\..\\npl.png'
# OUTPUT_PATH = '..\\..\\..\\output'

# # ❌ 以下写法在 Windows 会报错：
# INPUT_PATH = '..\..\..\npl.png'      # 没转义，会出错！

# # Linux only 上三层路径写法
# MODEL_PATH = '../../../model'
# INPUT_PATH = '../../../npl.png'
# OUTPUT_PATH = '../../../output'
# Bo: 路径写法跨平台对比**********************************************************


# TODO: change INPUT_PATH
# .pdf: run_dpsk_ocr_pdf.py; 
# .jpg, .png, .jpeg: run_dpsk_ocr_image.py; 
# Omnidocbench images path: run_dpsk_ocr_eval_batch.py

# INPUT_PATH = str(PROJECT_ROOT / 'input'/ 'npl.png') # image path
# INPUT_PATH = str(PROJECT_ROOT / 'input'/ '惠元2024年第七期不良资产证券化信托受托机构报告2026年度第1期（总第16期）.pdf') # pdf path
INPUT_PATH = str(PROJECT_ROOT / 'input'/ '建欣2025年第十八期不良资产支持证券评级报告及跟踪评级安排（中债资信）.pdf') # pdf path
OUTPUT_PATH = str(PROJECT_ROOT / 'output')

PROMPT = '<image>\n<|grounding|>Convert the document to markdown.'
# PROMPT = '<image>\nFree OCR.'
# TODO commonly used prompts
# document: <image>\n<|grounding|>Convert the document to markdown.
# other image: <image>\n<|grounding|>OCR this image.
# without layouts: <image>\nFree OCR.
# figures in document: <image>\nParse the figure.
# general: <image>\nDescribe this image in detail.
# rec: <image>\nLocate <|ref|>xxxx<|/ref|> in the image.
# '先天下之忧而忧'
# .......


from transformers import AutoTokenizer

TOKENIZER = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
