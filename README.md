# Shift and Flip Invariant Convolutional Neural Networks for Thermo-Fluid Flow Properties Prediction of Laminar Channel Flows

This repository contains the implementation of a shift and flip invariant convolutional neural network (CNN) designed for predicting thermo-fluid flow properties in laminar channel flows. The code enables robust and precise predictions by leveraging periodic boundary conditions.

---

## Conda Environment Setup

To set up the environment for this project using Conda, follow these steps:

1. Create the Conda environment from the provided `.yml` file:

    ```bash
    conda env create -f bwuni.yml
    ```

2. Activate the new environment:

    ```bash
    conda activate invariant_cnn
    ```

This environment contains all the required dependencies to run the project code, ensuring compatibility and reproducible experiments.

---

## Getting Started

### Training and Testing the Model

To train and test the periodic invariant CNN, you can execute the following command:

```bash
python3 iv_cnn_periodic_file.py --config train_config.yaml --testing False
```

To run the model for testing purposes only, use:

```bash
python3 iv_cnn_periodic_file.py --config train_config.yaml --testing True
```

### Configuration File

The model's settings (such as hyperparameters and data paths) are defined in the YAML configuration file `train_config.yaml`. Adjust this file as necessary to fit your specific dataset and experiment settings.

---


