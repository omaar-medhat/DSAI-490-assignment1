import tensorflow as tf
from tensorflow.keras import layers, Model

from src.config import IMG_SIZE, CHANNELS


def create_autoencoder(latent_dim=32):
    encoder_input = layers.Input(
        shape=(IMG_SIZE, IMG_SIZE, CHANNELS),
        name="ae_encoder_input"
    )

    x = layers.Conv2D(32, 3, strides=2, padding="same", activation="relu")(encoder_input)
    x = layers.Conv2D(64, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2D(128, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.Flatten()(x)

    latent = layers.Dense(latent_dim, name="ae_latent_vector")(x)

    encoder = Model(encoder_input, latent, name="AE_Encoder")

    decoder_input = layers.Input(shape=(latent_dim,), name="ae_decoder_input")

    x = layers.Dense(8 * 8 * 128, activation="relu")(decoder_input)
    x = layers.Reshape((8, 8, 128))(x)
    x = layers.Conv2DTranspose(128, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2DTranspose(64, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2DTranspose(32, 3, strides=2, padding="same", activation="relu")(x)

    decoder_output = layers.Conv2D(
        CHANNELS,
        3,
        padding="same",
        activation="sigmoid",
        name="ae_reconstruction"
    )(x)

    decoder = Model(decoder_input, decoder_output, name="AE_Decoder")

    autoencoder = Model(
        encoder_input,
        decoder(encoder(encoder_input)),
        name="Autoencoder"
    )

    autoencoder.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="mse",
        metrics=["mse"]
    )

    return autoencoder, encoder, decoder


class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def create_vae_networks(latent_dim=3):
    encoder_input = layers.Input(
        shape=(IMG_SIZE, IMG_SIZE, CHANNELS),
        name="vae_encoder_input"
    )

    x = layers.Conv2D(32, 3, strides=2, padding="same", activation="relu")(encoder_input)
    x = layers.Conv2D(64, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2D(128, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.Flatten()(x)

    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling(name="z_sampling")([z_mean, z_log_var])

    encoder = Model(
        encoder_input,
        [z_mean, z_log_var, z],
        name="VAE_Encoder"
    )

    decoder_input = layers.Input(shape=(latent_dim,), name="vae_decoder_input")

    x = layers.Dense(8 * 8 * 128, activation="relu")(decoder_input)
    x = layers.Reshape((8, 8, 128))(x)
    x = layers.Conv2DTranspose(128, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2DTranspose(64, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2DTranspose(32, 3, strides=2, padding="same", activation="relu")(x)

    decoder_output = layers.Conv2D(
        CHANNELS,
        3,
        padding="same",
        activation="sigmoid",
        name="vae_reconstruction"
    )(x)

    decoder = Model(decoder_input, decoder_output, name="VAE_Decoder")

    return encoder, decoder


class VAE(Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)

        self.encoder = encoder
        self.decoder = decoder

        self.total_loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker
        ]

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstruction = self.decoder(z)
        return reconstruction

    def compute_vae_loss(self, x):
        z_mean, z_log_var, z = self.encoder(x)
        reconstruction = self.decoder(z)

        reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(tf.square(x - reconstruction), axis=[1, 2, 3])
        )

        kl_loss = -0.5 * tf.reduce_mean(
            tf.reduce_sum(
                1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var),
                axis=1
            )
        )

        total_loss = reconstruction_loss + kl_loss

        return total_loss, reconstruction_loss, kl_loss

    def train_step(self, data):
        x = data[0] if isinstance(data, tuple) else data

        with tf.GradientTape() as tape:
            total_loss, reconstruction_loss, kl_loss = self.compute_vae_loss(x)

        gradients = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result()
        }

    def test_step(self, data):
        x = data[0] if isinstance(data, tuple) else data

        total_loss, reconstruction_loss, kl_loss = self.compute_vae_loss(x)

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result()
        }
