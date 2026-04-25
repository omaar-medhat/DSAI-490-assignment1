"""
Configuration file for the AE and VAE Medical MNIST project.
Update ZIP_PATH and EXTRACT_PATH depending on whether you run the project
on Google Colab or locally.
"""

IMG_SIZE = 64
CHANNELS = 1
BATCH_SIZE = 32

AE_LATENT_DIM = 32
VAE_LATENT_DIM = 3

EPOCHS_AE = 10
EPOCHS_VAE = 10
EPOCHS_DENOISING = 5

SEED = 42

# Google Colab path
ZIP_PATH = "/content/drive/MyDrive/DSAI490/medical-mnist/archive.zip"
EXTRACT_PATH = "/content/medical_mnist"

# Local example:
# ZIP_PATH = "data/archive.zip"
# EXTRACT_PATH = "data/medical_mnist"

MODEL_SAVE_DIR = "outputs/models"
FIGURE_SAVE_DIR = "outputs/figures"
