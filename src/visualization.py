import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def save_or_show(save_path=None):
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()


def show_dataset_samples(dataset, class_names, n=10, save_path=None):
    images, labels_batch = next(iter(dataset))

    plt.figure(figsize=(15, 3))

    for i in range(min(n, len(images))):
        plt.subplot(1, n, i + 1)
        plt.imshow(images[i].numpy().squeeze(), cmap="gray")
        plt.title(class_names[int(labels_batch[i])], fontsize=8)
        plt.axis("off")

    plt.suptitle("Dataset Samples")
    save_or_show(save_path)


def plot_ae_loss(history, save_path=None):
    plt.figure(figsize=(7, 5))
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")

    plt.title("AE Reconstruction Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(True)

    save_or_show(save_path)


def plot_vae_losses(history, figure_dir=None):
    plt.figure(figsize=(7, 5))
    plt.plot(history.history["loss"], label="Train Total Loss")
    plt.plot(history.history["val_loss"], label="Validation Total Loss")
    plt.title("VAE Total Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    save_or_show(os.path.join(figure_dir, "vae_total_loss.png") if figure_dir else None)

    plt.figure(figsize=(7, 5))
    plt.plot(history.history["reconstruction_loss"], label="Train Reconstruction Loss")
    plt.plot(history.history["val_reconstruction_loss"], label="Validation Reconstruction Loss")
    plt.title("VAE Reconstruction Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    save_or_show(os.path.join(figure_dir, "vae_reconstruction_loss.png") if figure_dir else None)

    plt.figure(figsize=(7, 5))
    plt.plot(history.history["kl_loss"], label="Train KL Loss")
    plt.plot(history.history["val_kl_loss"], label="Validation KL Loss")
    plt.title("VAE KL Divergence Loss")
    plt.xlabel("Epoch")
    plt.ylabel("KL Loss")
    plt.legend()
    plt.grid(True)
    save_or_show(os.path.join(figure_dir, "vae_kl_loss.png") if figure_dir else None)


def show_reconstruction(model, dataset, title, n=8, save_path=None):
    x_batch, _ = next(iter(dataset))
    recon = model.predict(x_batch, verbose=0)

    plt.figure(figsize=(2 * n, 4))

    for i in range(n):
        plt.subplot(2, n, i + 1)
        plt.imshow(x_batch[i].numpy().squeeze(), cmap="gray")
        plt.title("Original")
        plt.axis("off")

        plt.subplot(2, n, i + 1 + n)
        plt.imshow(recon[i].squeeze(), cmap="gray")
        plt.title("Reconstructed")
        plt.axis("off")

    plt.suptitle(title)
    save_or_show(save_path)


def compare_ae_vae(ae_model, vae_model, dataset, n=8, save_path=None):
    x_batch, _ = next(iter(dataset))

    ae_recon = ae_model.predict(x_batch, verbose=0)
    vae_recon = vae_model.predict(x_batch, verbose=0)

    plt.figure(figsize=(2 * n, 6))

    for i in range(n):
        plt.subplot(3, n, i + 1)
        plt.imshow(x_batch[i].numpy().squeeze(), cmap="gray")
        plt.title("Original")
        plt.axis("off")

        plt.subplot(3, n, i + 1 + n)
        plt.imshow(ae_recon[i].squeeze(), cmap="gray")
        plt.title("AE")
        plt.axis("off")

        plt.subplot(3, n, i + 1 + 2 * n)
        plt.imshow(vae_recon[i].squeeze(), cmap="gray")
        plt.title("VAE")
        plt.axis("off")

    plt.suptitle("Original vs AE Reconstruction vs VAE Reconstruction")
    save_or_show(save_path)


def get_ae_latents(encoder, labeled_dataset):
    latents = []
    labels_all = []

    for images, labels_batch in labeled_dataset:
        z = encoder.predict(images, verbose=0)
        latents.append(z)
        labels_all.append(labels_batch.numpy())

    return np.concatenate(latents, axis=0), np.concatenate(labels_all, axis=0)


def get_vae_latents(encoder, labeled_dataset):
    latents = []
    labels_all = []

    for images, labels_batch in labeled_dataset:
        z_mean, z_log_var, z = encoder.predict(images, verbose=0)
        latents.append(z_mean)
        labels_all.append(labels_batch.numpy())

    return np.concatenate(latents, axis=0), np.concatenate(labels_all, axis=0)


def plot_latent_2d(latents, labels, class_names, title, save_path=None):
    if latents.shape[1] > 2:
        reducer = PCA(n_components=2, random_state=42)
        latents_2d = reducer.fit_transform(latents)
        x_label = "PCA Dimension 1"
        y_label = "PCA Dimension 2"
    else:
        latents_2d = latents
        x_label = "Latent Dimension 1"
        y_label = "Latent Dimension 2"

    plt.figure(figsize=(8, 6))

    scatter = plt.scatter(
        latents_2d[:, 0],
        latents_2d[:, 1],
        c=labels,
        cmap="tab10",
        alpha=0.7,
        s=12
    )

    cbar = plt.colorbar(scatter)
    cbar.set_ticks(range(len(class_names)))
    cbar.set_ticklabels(class_names)

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)

    save_or_show(save_path)


def plot_latent_3d(latents, labels, class_names, title, save_path=None):
    if latents.shape[1] > 3:
        reducer = PCA(n_components=3, random_state=42)
        latents_3d = reducer.fit_transform(latents)
        x_label = "PCA Dimension 1"
        y_label = "PCA Dimension 2"
        z_label = "PCA Dimension 3"
    elif latents.shape[1] == 3:
        latents_3d = latents
        x_label = "Latent Dimension 1"
        y_label = "Latent Dimension 2"
        z_label = "Latent Dimension 3"
    else:
        print("3D plot needs at least 3 latent dimensions.")
        return

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    scatter = ax.scatter(
        latents_3d[:, 0],
        latents_3d[:, 1],
        latents_3d[:, 2],
        c=labels,
        cmap="tab10",
        alpha=0.7,
        s=12
    )

    cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
    cbar.set_ticks(range(len(class_names)))
    cbar.set_ticklabels(class_names)

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)

    save_or_show(save_path)


def generate_vae_samples(decoder, latent_dim=3, n=10, save_path=None):
    z_samples = np.random.normal(size=(n, latent_dim)).astype("float32")
    generated_images = decoder.predict(z_samples, verbose=0)

    plt.figure(figsize=(2 * n, 2))

    for i in range(n):
        plt.subplot(1, n, i + 1)
        plt.imshow(generated_images[i].squeeze(), cmap="gray")
        plt.title(f"S{i + 1}")
        plt.axis("off")

    plt.suptitle("Generated Samples from VAE Latent Space")
    save_or_show(save_path)


def show_denoising_results(model, denoise_dataset, n=8, save_path=None):
    noisy_images, clean_images = next(iter(denoise_dataset))
    denoised_images = model.predict(noisy_images, verbose=0)

    plt.figure(figsize=(2 * n, 6))

    for i in range(n):
        plt.subplot(3, n, i + 1)
        plt.imshow(clean_images[i].numpy().squeeze(), cmap="gray")
        plt.title("Clean")
        plt.axis("off")

        plt.subplot(3, n, i + 1 + n)
        plt.imshow(noisy_images[i].numpy().squeeze(), cmap="gray")
        plt.title("Noisy")
        plt.axis("off")

        plt.subplot(3, n, i + 1 + 2 * n)
        plt.imshow(denoised_images[i].squeeze(), cmap="gray")
        plt.title("Denoised")
        plt.axis("off")

    plt.suptitle("Denoising Autoencoder Results")
    save_or_show(save_path)
