"""文字リストから学習リソースを生成するモジュール"""

import shutil
import argparse
import json
from pathlib import Path
from typing import List
from PIL import Image, ImageDraw, ImageFont


def generate(
    output_dir: str,
    text_file_paths: List[str],
    font_paths: List[str],
    font_size: int = 16,
    image_size: tuple[int, int] = (30, 30),
    extension: str = "jpg",
) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # cleanup
    if (Path(output_dir) / "images").exists():
        shutil.rmtree(Path(output_dir) / "images")
        (Path(output_dir) / "images").mkdir(parents=True, exist_ok=True)

    if (Path(output_dir) / "classification").exists():
        shutil.rmtree(Path(output_dir) / "classification")
        (Path(output_dir) / "classification").mkdir(parents=True, exist_ok=True)

    with open(Path(output_dir) / "config.json", "w", encoding="utf-8") as f:
        config = {
            "text_file_paths": text_file_paths,
            "font_paths": font_paths,
            "font_size": font_size,
            "image_size": f"{image_size[0]}x{image_size[1]}",
            "extension": extension,
        }
        json.dump(config, f, ensure_ascii=False, indent=2)

    label_texts: List[str] = []
    rec_gt_texts: List[str] = []
    unicode_char_map = {}

    for text_file_path in text_file_paths:
        stem = Path(text_file_path).stem
        (Path(output_dir) / "images" / stem).mkdir(parents=True, exist_ok=True)
        chars = parse_char_file(text_file_path)

        for font_path in font_paths:
            font_name = Path(font_path).stem
            for index, char in chars:
                image = generate_char_image(char, font_path, font_size, image_size)

                # PaddleOCR ソース
                label_key = f"images/{stem}/{index}.{extension}"
                label_value = f'[{{"transcription": "{char}", "points": [[0, 0], [{image_size[0]}, 0], [{image_size[0]}, {image_size[1]}], [0, {image_size[1]}]], "difficult": false}}]'
                label_texts.append(f"{label_key}\t{label_value}")
                rec_gt_texts.append(f"{label_key}\t{char}")
                image.save(Path(output_dir) / "images" / stem / f"{index}.{extension}")

                # Tensorflow ソース
                unicode_str = char_to_unicode(char)
                unicode_dir = Path(output_dir) / "classification" / unicode_str
                unicode_dir.mkdir(parents=True, exist_ok=True)
                unicode_char_map[unicode_str] = char
                image.save(unicode_dir / f"{font_name}.{extension}")


    with open(Path(output_dir) / "Label.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(label_texts))
    with open(Path(output_dir) / "rec_gt.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(rec_gt_texts))
    with open(Path(output_dir) / "unicode_char.json", "w", encoding="utf-8") as f:
        json.dump(unicode_char_map, f, ensure_ascii=False, indent=2)


def char_to_unicode(char: str) -> str:
    """文字をUnicodeコードポイント文字列に変換

    Args:
        char: 変換する文字

    Returns:
        Unicodeコードポイント文字列（例: "亜" -> "4E9C"）
    """
    return format(ord(char), "04X")


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


def parse_image_size(value: str) -> tuple[int, int]:
    """画像サイズ文字列をパース（例: "128x128" -> (128, 128)）"""
    parts = value.lower().split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"Invalid image size format: {value}. Use WxH (e.g., 128x128)")
    return (int(parts[0]), int(parts[1]))


def main():
    generate(
        output_dir="training/t02",
        text_file_paths=[
            "doc/jis_x_0208_level3_kanji.txt",
            "doc/jis_x_0208_level4_kanji.txt"
        ],
        font_paths = [
            "fonts/IPAexGothic-Regular.ttf",
            "fonts/IPAexMincho-Regular.ttf",
            "fonts/MPlus1c-Regular.ttf",
            "fonts/MPlus1p-Bold.ttf",
            "fonts/MPlus1p-Regular.ttf",
            "fonts/MPlus2p-Regular.ttf",
            "fonts/NotoSansJP-Bold.ttf",
            "fonts/NotoSansJP-Light.otf",
            "fonts/NotoSansJP-Regular.ttf",
            "fonts/NotoSerifJP-Regular.otf",
            "fonts/SourceHanSansJP-Regular.otf",
        ],
        font_size=16,
        image_size=(30, 30),
        extension="jpg",
    )


if __name__ == "__main__":
    main()
