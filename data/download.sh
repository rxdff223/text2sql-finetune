#!/bin/bash
set -e

DATA_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== Downloading Spider dataset ==="
if [ ! -d "$DATA_DIR/spider" ]; then
    wget -q https://drive.usercontent.google.com/download?id=1iRDVHLr6THDbL1wIJo5Fmkr7XL96OPl5 -O "$DATA_DIR/spider.zip"
    unzip -q "$DATA_DIR/spider.zip" -d "$DATA_DIR/spider"
    rm "$DATA_DIR/spider.zip"
    echo "Spider downloaded."
else
    echo "Spider already exists, skipping."
fi

echo "=== Downloading BIRD dataset ==="
if [ ! -d "$DATA_DIR/bird" ]; then
    wget -q https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip -O "$DATA_DIR/bird_dev.zip"
    wget -q https://bird-bench.oss-cn-beijing.aliyuncs.com/train.zip -O "$DATA_DIR/bird_train.zip"
    unzip -q "$DATA_DIR/bird_dev.zip" -d "$DATA_DIR/bird"
    unzip -q "$DATA_DIR/bird_train.zip" -d "$DATA_DIR/bird"
    rm "$DATA_DIR/bird_dev.zip" "$DATA_DIR/bird_train.zip"
    echo "BIRD downloaded."
else
    echo "BIRD already exists, skipping."
fi

echo "=== Downloading CSpider (Chinese) dataset ==="
if [ ! -d "$DATA_DIR/cspider" ]; then
    git clone https://github.com/taolusi/chisp.git "$DATA_DIR/cspider"
    echo "CSpider downloaded."
else
    echo "CSpider already exists, skipping."
fi

echo "=== All datasets downloaded ==="
