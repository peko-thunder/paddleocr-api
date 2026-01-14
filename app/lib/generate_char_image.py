"""文字リストから文字画像を生成するモジュール"""

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def parse_char_file(file_path: str) -> list[tuple[int, str]]:
    """文字ファイルを解析し、(インデックス, 文字)のリストを返す

    対応フォーマット:
    - 1行1文字形式（例: "亜\\n唖\\n娃"）
    - 番号→文字形式（例: "1→亜"）

    Args:
        file_path: 文字リストファイルのパス

    Returns:
        (インデックス, 文字)のタプルのリスト（インデックスは1始まり）
    """
    result = []
    with open(file_path, "r", encoding="utf-8") as f:
        index = 1
        for line in f:
            line = line.strip()
            if not line:
                continue

            if "→" in line:
                parts = line.split("→")
                if len(parts) == 2:
                    idx = int(parts[0].strip())
                    char = parts[1]
                    result.append((idx, char))
            else:
                result.append((index, line))
                index += 1
    return result


def generate_char_image(
    char: str,
    font_path: str,
    font_size: int,
    image_size: tuple[int, int],
) -> Image.Image:
    """単一文字の画像を生成

    Args:
        char: 描画する文字
        font_path: フォントファイルのパス
        font_size: フォントサイズ（px）
        image_size: 画像サイズ (width, height)

    Returns:
        生成された画像（PIL Image）
    """
    image = Image.new("RGB", image_size, color="white")
    draw = ImageDraw.Draw(image)

    font = ImageFont.truetype(font_path, font_size)

    bbox = draw.textbbox((0, 0), char, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    x = (image_size[0] - text_width) / 2 - bbox[0]
    y = (image_size[1] - text_height) / 2 - bbox[1]

    draw.text((x, y), char, font=font, fill="black")

    return image


def generate_images_from_file(
    text_file_path: str,
    font_path: str,
    font_size: int = 64,
    image_size: tuple[int, int] = (128, 128),
    extension: str = "png",
) -> list[dict]:
    """文字ファイルから全画像を生成し、ログを出力

    Args:
        text_file_path: 入力文字リストファイルのパス
        font_path: フォントファイルのパス
        font_size: フォントサイズ（px）
        image_size: 画像サイズ (width, height)
        extension: 出力画像の拡張子

    Returns:
        生成ログのリスト
    """
    chars = parse_char_file(text_file_path)

    file_stem = Path(text_file_path).stem
    output_dir = Path("public/char_image") / file_stem
    output_dir.mkdir(parents=True, exist_ok=True)

    log_dir = Path("public/gen_image_log")
    log_dir.mkdir(parents=True, exist_ok=True)

    logs = []
    image_size_str = f"{image_size[0]}x{image_size[1]}"

    for index, char in chars:
        image = generate_char_image(char, font_path, font_size, image_size)

        image_path = output_dir / f"{index}.{extension}"
        image.save(image_path)

        log_entry = {
            "text_file_path": text_file_path,
            "char": char,
            "image_path": str(image_path),
            "font_family": font_path,
            "font_size": str(font_size),
            "image_size": image_size_str,
        }
        logs.append(log_entry)

    log_file_path = log_dir / f"{file_stem}.json"
    with open(log_file_path, "w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)

    print(f"Generated {len(logs)} images to {output_dir}")
    print(f"Log saved to {log_file_path}")

    return logs


def parse_image_size(value: str) -> tuple[int, int]:
    """画像サイズ文字列をパース（例: "128x128" -> (128, 128)）"""
    parts = value.lower().split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"Invalid image size format: {value}. Use WxH (e.g., 128x128)")
    return (int(parts[0]), int(parts[1]))


def main():
    generate_images_from_file(
        text_file_path="doc/jis_x_0208_level1_kanji.txt",
        font_path="fonts/NotoSansJP-Regular.ttf",
        font_size=16,
        image_size=(30, 30),
        extension="jpg",
    )


if __name__ == "__main__":
    main()
