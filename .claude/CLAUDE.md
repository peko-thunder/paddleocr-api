# PaddleOCR API

FastAPI + PaddleOCR + Dockerを使用した日本語対応OCR WebAPI

## プロジェクト構造

```
/workspace/
├── app/
│   ├── main.py          # FastAPIエンドポイント
│   ├── ocr_service.py   # PaddleOCRラッパー
│   └── models.py        # Pydanticモデル
├── tests/
│   └── test_api.py      # ユニットテスト
└── .devcontainer/
    ├── Dockerfile
    ├── docker-compose.yml
    ├── requirements.txt
    └── preload_models.py  # OCRモデルプリダウンロードスクリプト
```

## 技術スタック

- Python 3.12
- FastAPI + Uvicorn
- PaddleOCR (日本語対応)
- Docker

## 開発コマンド

### サーバー起動
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### テスト実行
```bash
pytest tests/ -v
```

### コンテナ再ビルド
```bash
cd /workspace/.devcontainer && docker compose up --build -d
```

## APIエンドポイント

| メソッド | パス | 説明 |
|---------|------|------|
| GET | `/` | ヘルスチェック |
| GET | `/health` | 詳細ヘルスチェック |
| POST | `/ocr` | 画像アップロードでOCR処理 |

### OCRリクエスト例
```bash
curl -X POST "http://localhost:8000/ocr" -F "file=@image.png"
```

## 重要な注意事項

- PaddleOCRはPython 3.12まで対応（3.13以降は非対応）
- OCRモデルはDockerビルド時に自動ダウンロードされる（イメージに含まれる）
- 対応画像形式: JPEG, PNG, GIF, BMP, WebP

## Dockerビルドについて

Dockerイメージのビルド時に、PaddleOCRの日本語モデルが自動的にダウンロードされます。
これにより、コンテナ起動後すぐにOCR処理が可能になります。

ビルド時のモデルダウンロードは`.devcontainer/preload_models.py`で実行されます。
