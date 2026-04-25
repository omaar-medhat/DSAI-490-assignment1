import os
import random
import numpy as np
import tensorflow as tf

from src.config import SEED


def set_seed():
    tf.random.set_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)


def create_output_dirs(model_dir, figure_dir):
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(figure_dir, exist_ok=True)


def save_models(
    model_dir,
    ae,
    ae_encoder,
    ae_decoder,
    vae_encoder,
    vae_decoder,
    denoising_ae
):
    ae.save(os.path.join(model_dir, "autoencoder.keras"))
    ae_encoder.save(os.path.join(model_dir, "ae_encoder.keras"))
    ae_decoder.save(os.path.join(model_dir, "ae_decoder.keras"))

    vae_encoder.save(os.path.join(model_dir, "vae_encoder.keras"))
    vae_decoder.save(os.path.join(model_dir, "vae_decoder.keras"))

    denoising_ae.save(os.path.join(model_dir, "denoising_autoencoder.keras"))


def evaluate_reconstruction_quality(ae_model, vae_model, dataset):
    ae_mse_values = []
    vae_mse_values = []

    ae_ssim_values = []
    vae_ssim_values = []

    for x_batch, _ in dataset:
        ae_recon = ae_model.predict(x_batch, verbose=0)
        vae_recon = vae_model.predict(x_batch, verbose=0)

        ae_mse = tf.reduce_mean(tf.square(x_batch - ae_recon), axis=[1, 2, 3])
        vae_mse = tf.reduce_mean(tf.square(x_batch - vae_recon), axis=[1, 2, 3])

        ae_ssim = tf.image.ssim(x_batch, ae_recon, max_val=1.0)
        vae_ssim = tf.image.ssim(x_batch, vae_recon, max_val=1.0)

        ae_mse_values.extend(ae_mse.numpy())
        vae_mse_values.extend(vae_mse.numpy())

        ae_ssim_values.extend(ae_ssim.numpy())
        vae_ssim_values.extend(vae_ssim.numpy())

    return {
        "AE MSE": float(np.mean(ae_mse_values)),
        "VAE MSE": float(np.mean(vae_mse_values)),
        "AE SSIM": float(np.mean(ae_ssim_values)),
        "VAE SSIM": float(np.mean(vae_ssim_values))
    }


def evaluate_denoising(model, denoise_dataset):
    noisy_mse_values = []
    denoised_mse_values = []

    noisy_ssim_values = []
    denoised_ssim_values = []

    for noisy_images, clean_images in denoise_dataset:
        denoised_images = model.predict(noisy_images, verbose=0)

        noisy_mse = tf.reduce_mean(tf.square(clean_images - noisy_images), axis=[1, 2, 3])
        denoised_mse = tf.reduce_mean(tf.square(clean_images - denoised_images), axis=[1, 2, 3])

        noisy_ssim = tf.image.ssim(clean_images, noisy_images, max_val=1.0)
        denoised_ssim = tf.image.ssim(clean_images, denoised_images, max_val=1.0)

        noisy_mse_values.extend(noisy_mse.numpy())
        denoised_mse_values.extend(denoised_mse.numpy())

        noisy_ssim_values.extend(noisy_ssim.numpy())
        denoised_ssim_values.extend(denoised_ssim.numpy())

    return {
        "Noisy Input MSE": float(np.mean(noisy_mse_values)),
        "Denoised Output MSE": float(np.mean(denoised_mse_values)),
        "Noisy Input SSIM": float(np.mean(noisy_ssim_values)),
        "Denoised Output SSIM": float(np.mean(denoised_ssim_values))
    }
