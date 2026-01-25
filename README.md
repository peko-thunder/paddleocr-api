# Memo

## Trainingコードのリポジトリをインストール

```bash
git clone https://github.com/PaddlePaddle/PaddleOCR.git
cd PaddleOCR
pip install -r requirements.txt
```

## Training準備

```bash
cd /workspace
cp PaddleOCR/configs/det/PP-OCRv5/PP-OCRv5_server_det.yml training/t01
cp PaddleOCR/configs/rec/PP-OCRv5/PP-OCRv5_server_rec.yml training/t01
```

## Training実行

```bash
cd PaddleOCR
python tools/train.py -c "/workspace/training/t01/PP-OCRv5_server_det.yml"
python tools/train.py -c "/workspace/training/t01/PP-OCRv5_server_rec.yml"
```
