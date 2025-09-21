# Convolutional Autoencoder for Anomaly Detection on MNIST

This repository contains a Jupyter Notebook (`cnn_autoencoders.ipynb`) that demonstrates how to build and train a Convolutional Neural Network (CNN) based autoencoder for anomaly detection using the MNIST dataset.

## Overview

Anomaly detection involves identifying data samples that significantly differ from the majority of a dataset. Here, a CNN autoencoder is trained in an unsupervised manner (ignoring MNIST labels) to reconstruct handwritten digit images. Images that the autoencoder fails to reconstruct well (i.e., with high reconstruction loss) are considered anomalies.

## Features

- **Unsupervised Learning:** Uses the MNIST dataset without labels as an unsupervised anomaly detection task.
- **CNN Autoencoder Architecture:** The encoder consists of convolutional, ReLU, batch normalization, and max-pooling layers; the decoder uses upsampling, convolution, and sigmoid activations.
- **Custom Loss Function:** Mean Squared Error (MSE) is used to measure reconstruction quality.
- **Training Loop:** Includes both training and validation phases, with support for GPUs.
- **Anomaly Identification:** Computes reconstruction loss per image to rank anomalies and visualize them.

## File Structure

- `cnn_autoencoders.ipynb`: Main notebook containing code, comments, and visualizations.
- `helpers.py`: Utility functions for data loading, reproducibility, and visualization (expected to be present).
- `requirements.txt`: List of required Python packages (expected to be present).

## Usage

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   The notebook includes code to install dependencies and restart the kernel automatically.

2. **Run the Notebook:**
   Open `cnn_autoencoders.ipynb` in Jupyter Notebook or JupyterLab and follow the cells sequentially.

## Main Steps in Notebook

1. **Setup & Imports:** Installs dependencies, imports libraries, and sets random seeds for reproducibility.
2. **Data Preparation:** Loads MNIST dataset using helper functions.
3. **Visualization:** Displays sample MNIST images.
4. **Model Definition:** Implements the CNN autoencoder using PyTorch.
5. **Training:** Trains the autoencoder and plots loss curves.
6. **Anomaly Detection:** Computes per-image reconstruction loss and visualizes anomalies.
7. **Visualization:** Plots the loss distribution and displays the most anomalous images.

## Dependencies

- Python 3.x
- PyTorch
- torchvision
- numpy
- pandas
- matplotlib
- tqdm


## Results

The notebook will display:

- Training and validation loss curves
- Distribution of reconstruction losses
- Images ranked by anomaly score (high loss)

## Notes

- The notebook is designed for educational purposes. MNIST is not inherently an anomaly dataset.
- Labels are ignored to simulate unsupervised anomaly detection.
- The autoencoder architecture and training loop can be adapted for other datasets or tasks.

## License

This project is released under the MIT License.

---

**For questions or suggestions, please open an issue or pull request.**
