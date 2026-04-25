# Autoencoder and Variational Autoencoder for Medical MNIST

## Project Overview

This project implements and compares an Autoencoder (AE) and a Variational Autoencoder (VAE) for medical image reconstruction, latent space visualization, sample generation, and denoising.

The project uses the given Medical MNIST dataset from a `.zip` file. The dataset should be uploaded to Google Drive and loaded after mounting Google Drive in Google Colab, or placed locally and loaded from a local path.

## Objectives

- Develop and train an Autoencoder for image reconstruction.
- Implement a Variational Autoencoder with a probabilistic latent space.
- Apply the reparameterization trick in the VAE.
- Visualize learned latent representations in 2D and 3D.
- Compare AE and VAE reconstruction performance.
- Generate new samples using the VAE decoder.
- Explore denoising using a denoising Autoencoder.
- Visualize AE loss, VAE reconstruction loss, and VAE KL divergence loss.

## Repository Structure

```text
ae_vae_medical_mnist/
│
├── src/
│   ├── config.py
│   ├── data.py
│   ├── models.py
│   ├── train.py
│   ├── visualization.py
│   └── utils.py
│
├── notebooks/
│   └── experiment_notebook.md
│
├── reports/
│   └── technical_report.md
│
├── outputs/
│   ├── figures/
│   └── models/
│
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Dataset Setup

### Google Colab

Upload the dataset zip file to Google Drive, for example:

```text
/content/drive/MyDrive/DSAI490/medical-mnist/archive.zip
```

Then set the dataset path in `src/config.py`:

```python
ZIP_PATH = "/content/drive/MyDrive/DSAI490/medical-mnist/archive.zip"
EXTRACT_PATH = "/content/medical_mnist"
```

### Local Computer

If running locally, place the dataset zip file in a local folder and update:

```python
ZIP_PATH = "data/archive.zip"
EXTRACT_PATH = "data/medical_mnist"
```

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the full experiment:

```bash
python main.py
```

## Main Features

### Autoencoder

The AE uses a convolutional encoder and convolutional transpose decoder. It learns a deterministic latent representation and reconstructs the input image using MSE loss.

### Variational Autoencoder

The VAE uses a probabilistic latent space with:

- `z_mean`
- `z_log_var`
- Reparameterization trick
- Reconstruction loss
- KL divergence loss

### Latent Space Visualization

The latent vectors are visualized in:

- 2D
- 3D

Labels are used to color the latent space points, making it easier to analyze class separation.

### Sample Generation

The VAE decoder generates new samples from random latent vectors sampled from a normal distribution.

### Denoising

A denoising Autoencoder is trained using noisy images as input and clean images as targets.

## Evaluation Metrics

The project compares AE and VAE using:

- Mean Squared Error, MSE
- Structural Similarity Index Measure, SSIM
- Visual reconstruction comparison
- Latent space plots
- Generated samples
- Denoising outputs

## Report

The technical report is available in:

```text
reports/technical_report.md
```

## Video Demonstration

The video demonstration should show:

1. Dataset loading.
2. AE and VAE training.
3. Loss curves.
4. Reconstruction results.
5. Latent space visualization.
6. VAE generated samples.
7. Denoising results.
8. Main observations and conclusion.
