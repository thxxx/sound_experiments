#!/bin/bash

# 시스템 패키지 업데이트
apt-get update
apt-get install -y tmux
apt-get install -y zip
apt-get install -y ffmpeg

# Python 패키지 설치
pip install -r requirements.txt

# audiocraft 복제
if [ ! -d "audiocraft" ]; then
    git clone https://github.com/facebookresearch/audiocraft.git
fi

# wandb 로그인
wandb login 7aec073ce381d48833b28a64463928263ee70cc5
