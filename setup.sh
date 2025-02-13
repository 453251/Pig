#!/bin/bash

# 安装 paddlepaddle-gpu
python -m pip install paddlepaddle-gpu==2.4.2.post116 -f https://www.paddlepaddle.org.cn/whl/windows/mkl/avx/stable.html
# 安装 PaddleYOLO 的依赖
cd PaddleYOLO && pip install --user -r requirements.txt

# 安装 pycocotools
pip install pycocotools

# 升级 Pillow
pip install --upgrade pillow
