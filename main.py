try:
    from google.colab import drive
    drive.mount("/content/drive")
except Exception:
    print("Google Colab drive mount skipped. Running locally or Drive is already mounted.")

from src.config import (
    ZIP_PATH,
    EXTRACT_PATH,
    MODEL_SAVE_DIR,
    FIGURE_SAVE_DIR,
    VAE_LATENT_DIM
)
from src.data import (
    extract_dataset,
    collect_image_paths_and_labels,
    split_dataset,
    make_reconstruction_dataset,
    make_labeled_dataset,
    make_denoising_dataset
)
from src.train import (
    train_autoencoder,
    train_vae,
    train_denoising_autoencoder
)
from src.visualization import (
    show_dataset_samples,
    plot_ae_loss,
    plot_vae_losses,
    show_reconstruction,
    compare_ae_vae,
    get_ae_latents,
    get_vae_latents,
    plot_latent_2d,
    plot_latent_3d,
    generate_vae_samples,
    show_denoising_results
)
from src.utils import (
    set_seed,
    create_output_dirs,
    evaluate_reconstruction_quality,
    evaluate_denoising,
    save_models
)


def main():
    set_seed()
    create_output_dirs(MODEL_SAVE_DIR, FIGURE_SAVE_DIR)

    extract_dataset(ZIP_PATH, EXTRACT_PATH)

    image_paths, labels, class_names = collect_image_paths_and_labels(EXTRACT_PATH)

    print("Total images:", len(image_paths))
    print("Classes:", class_names)

    train_paths, val_paths, train_labels, val_labels = split_dataset(image_paths, labels)

    train_ds = make_reconstruction_dataset(train_paths, train_labels, training=True)
    val_ds = make_reconstruction_dataset(val_paths, val_labels, training=False)

    train_labeled_ds = make_labeled_dataset(train_paths, train_labels, training=False)
    val_labeled_ds = make_labeled_dataset(val_paths, val_labels, training=False)

    train_denoise_ds = make_denoising_dataset(train_paths, train_labels, training=True)
    val_denoise_ds = make_denoising_dataset(val_paths, val_labels, training=False)

    show_dataset_samples(
        train_labeled_ds,
        class_names,
        save_path=f"{FIGURE_SAVE_DIR}/dataset_samples.png"
    )

    ae, ae_encoder, ae_decoder, ae_history = train_autoencoder(train_ds, val_ds)

    vae, vae_encoder, vae_decoder, vae_history = train_vae(train_ds, val_ds)

    denoising_ae, denoising_encoder, denoising_decoder, denoising_history = train_denoising_autoencoder(
        train_denoise_ds,
        val_denoise_ds
    )

    plot_ae_loss(ae_history, save_path=f"{FIGURE_SAVE_DIR}/ae_loss.png")
    plot_vae_losses(vae_history, figure_dir=FIGURE_SAVE_DIR)

    show_reconstruction(
        ae,
        val_ds,
        "Autoencoder Reconstruction",
        save_path=f"{FIGURE_SAVE_DIR}/ae_reconstruction.png"
    )

    show_reconstruction(
        vae,
        val_ds,
        "Variational Autoencoder Reconstruction",
        save_path=f"{FIGURE_SAVE_DIR}/vae_reconstruction.png"
    )

    compare_ae_vae(
        ae,
        vae,
        val_ds,
        save_path=f"{FIGURE_SAVE_DIR}/ae_vs_vae.png"
    )

    reconstruction_results = evaluate_reconstruction_quality(ae, vae, val_ds)

    print("\nReconstruction Quality")
    print("----------------------")
    for metric, value in reconstruction_results.items():
        print(f"{metric}: {value:.6f}")

    ae_latents, ae_latent_labels = get_ae_latents(ae_encoder, val_labeled_ds)
    vae_latents, vae_latent_labels = get_vae_latents(vae_encoder, val_labeled_ds)

    plot_latent_2d(
        ae_latents,
        ae_latent_labels,
        class_names,
        "AE Latent Space 2D",
        save_path=f"{FIGURE_SAVE_DIR}/ae_latent_2d.png"
    )

    plot_latent_2d(
        vae_latents,
        vae_latent_labels,
        class_names,
        "VAE Latent Space 2D",
        save_path=f"{FIGURE_SAVE_DIR}/vae_latent_2d.png"
    )

    plot_latent_3d(
        ae_latents,
        ae_latent_labels,
        class_names,
        "AE Latent Space 3D",
        save_path=f"{FIGURE_SAVE_DIR}/ae_latent_3d.png"
    )

    plot_latent_3d(
        vae_latents,
        vae_latent_labels,
        class_names,
        "VAE Latent Space 3D",
        save_path=f"{FIGURE_SAVE_DIR}/vae_latent_3d.png"
    )

    generate_vae_samples(
        vae_decoder,
        latent_dim=VAE_LATENT_DIM,
        save_path=f"{FIGURE_SAVE_DIR}/vae_generated_samples.png"
    )

    show_denoising_results(
        denoising_ae,
        val_denoise_ds,
        save_path=f"{FIGURE_SAVE_DIR}/denoising_results.png"
    )

    denoising_results = evaluate_denoising(denoising_ae, val_denoise_ds)

    print("\nDenoising Quality")
    print("-----------------")
    for metric, value in denoising_results.items():
        print(f"{metric}: {value:.6f}")

    save_models(
        MODEL_SAVE_DIR,
        ae,
        ae_encoder,
        ae_decoder,
        vae_encoder,
        vae_decoder,
        denoising_ae
    )

    print("\nExperiment completed successfully.")


if __name__ == "__main__":
    main()
