# Image Keypoint Prediction Model

## Overview

This project involves building and training a Convolutional Neural Network (CNN) model to predict keypoints on images. The process includes the following steps:

1. **Data Preparation**: 
   - Load image data from a CSV file.
   - Convert image data from string format to numpy arrays.
   - Perform image visualization and augmentation, including flipping and brightness adjustment.
   - Prepare training and testing datasets.

2. **Model Architecture**:
   - Define a CNN model with residual blocks.
   - The network consists of convolutional layers, batch normalization, activation functions, max pooling, and residual connections.
   - Includes fully connected layers for final keypoint prediction.

3. **Training**:
   - Compile the model with Adam optimizer and mean squared error loss function.
   - Train the model using a dataset split into training and validation subsets.
   - Save the best model based on validation loss.

4. **Evaluation and Prediction**:
   - Evaluate the trained model on the test dataset.
   - Calculate and display the Root Mean Squared Error (RMSE) for predictions.
   - Visualize test images and their predicted keypoints.

## Files

- `data.csv`: Contains the image data and keypoints.
- `ModelArchitecture.json`: JSON file with the model architecture.
- `best_weights.keras`: HDF5 file with the best model weights.

## Requirements

- Python
- TensorFlow
- Keras
- NumPy
- pandas
- scikit-learn
- Matplotlib

## Usage

1. Ensure all dependencies are installed.
2. Place the `data.csv` file in the project directory.
3. Run the script to train the model and evaluate its performance.
4. The trained model's architecture and weights will be saved for future use.