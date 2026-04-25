import os
import zipfile
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from src.config import IMG_SIZE, CHANNELS, BATCH_SIZE, SEED

AUTOTUNE = tf.data.AUTOTUNE


def extract_dataset(zip_path, extract_path):
    os.makedirs(extract_path, exist_ok=True)

    if len(os.listdir(extract_path)) == 0:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_path)

    return extract_path


def collect_image_paths_and_labels(root_dir):
    class_names = sorted([
        folder for folder in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, folder))
    ])

    image_paths = []
    labels = []

    valid_extensions = (".png", ".jpg", ".jpeg", ".bmp")

    for label, class_name in enumerate(class_names):
        class_folder = os.path.join(root_dir, class_name)

        for filename in os.listdir(class_folder):
            if filename.lower().endswith(valid_extensions):
                image_paths.append(os.path.join(class_folder, filename))
                labels.append(label)

    return np.array(image_paths), np.array(labels), class_names


def split_dataset(image_paths, labels, test_size=0.2):
    return train_test_split(
        image_paths,
        labels,
        test_size=test_size,
        random_state=SEED,
        stratify=labels
    )


def load_image(path, label):
    img = tf.io.read_file(path)

    img = tf.image.decode_image(
        img,
        channels=CHANNELS,
        expand_animations=False
    )

    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    img = tf.cast(img, tf.float32) / 255.0

    label = tf.cast(label, tf.int32)

    return img, label


def make_reconstruction_dataset(paths, labels, training=True):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(load_image, num_parallel_calls=AUTOTUNE)

    if training:
        ds = ds.shuffle(buffer_size=len(paths), seed=SEED)

    ds = ds.map(lambda img, label: (img, img), num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

    return ds


def make_labeled_dataset(paths, labels, training=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(load_image, num_parallel_calls=AUTOTUNE)

    if training:
        ds = ds.shuffle(buffer_size=len(paths), seed=SEED)

    ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

    return ds


def add_noise(image, noise_factor=0.35):
    noise = noise_factor * tf.random.normal(shape=tf.shape(image))
    noisy_image = image + noise
    return tf.clip_by_value(noisy_image, 0.0, 1.0)


def make_denoising_dataset(paths, labels, training=True):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(load_image, num_parallel_calls=AUTOTUNE)

    if training:
        ds = ds.shuffle(buffer_size=len(paths), seed=SEED)

    ds = ds.map(
        lambda img, label: (add_noise(img), img),
        num_parallel_calls=AUTOTUNE
    )

    ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

    return ds
