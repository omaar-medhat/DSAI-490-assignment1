import tensorflow as tf

from src.config import (
    AE_LATENT_DIM,
    VAE_LATENT_DIM,
    EPOCHS_AE,
    EPOCHS_VAE,
    EPOCHS_DENOISING
)
from src.models import create_autoencoder, create_vae_networks, VAE


def train_autoencoder(train_ds, val_ds):
    ae, ae_encoder, ae_decoder = create_autoencoder(AE_LATENT_DIM)

    history = ae.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_AE
    )

    return ae, ae_encoder, ae_decoder, history


def train_vae(train_ds, val_ds):
    vae_encoder, vae_decoder = create_vae_networks(VAE_LATENT_DIM)
    vae = VAE(vae_encoder, vae_decoder, name="Variational_Autoencoder")

    vae.compile(optimizer=tf.keras.optimizers.Adam())

    history = vae.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_VAE
    )

    return vae, vae_encoder, vae_decoder, history


def train_denoising_autoencoder(train_denoise_ds, val_denoise_ds):
    denoising_ae, denoising_encoder, denoising_decoder = create_autoencoder(AE_LATENT_DIM)

    history = denoising_ae.fit(
        train_denoise_ds,
        validation_data=val_denoise_ds,
        epochs=EPOCHS_DENOISING
    )

    return denoising_ae, denoising_encoder, denoising_decoder, history
