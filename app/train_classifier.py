"""
TensorFlow画像分類モデル トレーニングスクリプト
文字（漢字）画像の分類モデルを作成
"""

import os
import json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path

# 設定
TRAIN_DATA_DIR = "training/t02/classification"
MODEL_SAVE_PATH = "models/character_classifier"
IMG_SIZE = (30, 30)
BATCH_SIZE = 32
EPOCHS = 50
VALIDATION_SPLIT = 0.2


def create_model(num_classes: int) -> keras.Model:
    """CNNモデルを作成"""
    model = keras.Sequential([
        # 入力層と正規化
        layers.Input(shape=(*IMG_SIZE, 3)),
        layers.Rescaling(1.0 / 255),

        # データ拡張（学習時のみ適用）
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomTranslation(0.1, 0.1),

        # 畳み込み層 1
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # 畳み込み層 2
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # 畳み込み層 3
        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # 全結合層
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ])

    return model


def load_dataset():
    """データセットを読み込み"""
    train_ds = keras.utils.image_dataset_from_directory(
        TRAIN_DATA_DIR,
        validation_split=VALIDATION_SPLIT,
        subset="training",
        seed=42,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
    )

    val_ds = keras.utils.image_dataset_from_directory(
        TRAIN_DATA_DIR,
        validation_split=VALIDATION_SPLIT,
        subset="validation",
        seed=42,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
    )

    return train_ds, val_ds


def save_class_names(class_names: list[str], save_dir: str):
    """クラス名をJSONファイルに保存"""
    class_mapping = {
        "index_to_class": {i: name for i, name in enumerate(class_names)},
        "class_to_index": {name: i for i, name in enumerate(class_names)},
    }

    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "class_names.json"), "w", encoding="utf-8") as f:
        json.dump(class_mapping, f, ensure_ascii=False, indent=2)

    print(f"クラス名マッピングを保存: {save_dir}/class_names.json")


def main():
    print("=" * 60)
    print("TensorFlow 画像分類モデル トレーニング")
    print("=" * 60)

    # GPUの確認
    gpus = tf.config.list_physical_devices("GPU")
    print(f"利用可能なGPU: {len(gpus)}台")
    if gpus:
        for gpu in gpus:
            print(f"  - {gpu}")

    # データセット読み込み
    print("\nデータセットを読み込み中...")
    train_ds, val_ds = load_dataset()

    class_names = train_ds.class_names
    num_classes = len(class_names)
    print(f"クラス数: {num_classes}")
    print(f"サンプルクラス: {class_names[:5]}...")

    # パフォーマンス最適化
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # モデル作成
    print("\nモデルを作成中...")
    model = create_model(num_classes)
    model.summary()

    # コンパイル
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # コールバック設定
    callbacks = [
        # 早期停止
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        # 学習率調整
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
        # モデルチェックポイント
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(MODEL_SAVE_PATH, "checkpoint.keras"),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
    ]

    # モデル保存ディレクトリ作成
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

    # トレーニング
    print("\n" + "=" * 60)
    print("トレーニング開始")
    print("=" * 60)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    # 最終モデル保存
    model.save(os.path.join(MODEL_SAVE_PATH, "final_model.keras"))
    print(f"\n最終モデルを保存: {MODEL_SAVE_PATH}/final_model.keras")

    # クラス名マッピング保存
    save_class_names(class_names, MODEL_SAVE_PATH)

    # 結果表示
    print("\n" + "=" * 60)
    print("トレーニング結果")
    print("=" * 60)

    final_train_acc = history.history["accuracy"][-1]
    final_val_acc = history.history["val_accuracy"][-1]
    best_val_acc = max(history.history["val_accuracy"])

    print(f"最終トレーニング精度: {final_train_acc:.4f}")
    print(f"最終検証精度: {final_val_acc:.4f}")
    print(f"最高検証精度: {best_val_acc:.4f}")

    # 学習履歴を保存
    with open(os.path.join(MODEL_SAVE_PATH, "history.json"), "w") as f:
        json.dump(history.history, f, indent=2)
    print(f"学習履歴を保存: {MODEL_SAVE_PATH}/history.json")

    print("\n完了!")


if __name__ == "__main__":
    main()
