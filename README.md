# SHIELD: Secure Homomorphic Inference for Encrypted Learning on Data

SHIELD is a library for evaluating pre-trained convolutional neural networks on homomorphically encrypted images. It includes code for training models that are suitable for homomorphic evaluation. Implemented neural network operations include convolution, average pooling, GELU, and linear layers.

This code was used to run the experiments supporting the following paper: [High-Resolution Convolutional Neural Networks on Homomorphically Encrypted Data via Sharding Ciphertexts
](https://arxiv.org/abs/2306.09189). However, operators defined in this project are generic enough to build arbitrary convolutional neural networks as specified in the paper.

## Requirements
This project's dependencies are managed by Poetry, so installing [Poetry](https://python-poetry.org/) is a requirement. [OpenFHE Python bindings](https://github.com/JHUAPL/openfhe-python-bindings) are used to interface with OpenFHE, so the wheel file for these bindings will also need to be built. See the OpenFHE Python bindings repository for further instructions.

Once the bindings are builts, ensure that the `pyproject.toml` file contains a correct path to the bindings. Then to install the Python environment for this project, run `poetry install`. For running unit tests and the small neural network as described below, 32GB of RAM is recommended. For hardware requirements needed to reproduce results for the larger ResNet architecures, see the paper for details.

Code was developed and tested on Ubuntu 20.04. While it should run on Windows platforms as well, this has not been explicitly tested.

## Features

SHIELD implements the following neural network operators:

- Convolution
- Average pooling
- Batch normalization (which are fused with convolution operators for performance)
- Linear
- GELU (Gaussian Error Linear Unit, a smooth alternative to ReLU)
- Upsample

For performance reasons, the core of these algorithms are mostly implemented in the companion OpenFHE Python bindings project (in C++), with this project providing a minimal but more user-friendly Python inference for using them.

The following neural network architectures are implemented using homomorphic implementations of the above operators: a neural network consisting of three convolution blocks (mainly for integration testing), and variations on ResNet including ResNet9 and ResNet50. In addition, code for training models suitable for homomorphic evaluation, using these architectures is included. Training code includes kurtosis regularization required for homomorphic inference. See the referenced paper for more details on the algorithms implemented, as well as performance metrics for homomorphic inference using these neural networks.

## Running the code

### Units tests

Tests are run with `pytest`:

```
poetry run python palisade_he_cnn/test.py
```

### A small neural network

`small_model.py` includes code defining a 3-layer convolutional neural network, as well as code to train a model, on MNIST, instantiated from this network. The training code can be run with:

```
poetry run python palisade_he_cnn/src/small_model.py
```

This will save model weights to `small_model.pt`. To run homomorphic inference with these weights, move the weights to `palisade_he_cnn/src/weights/` and then run:

```
poetry run python palisade_he_cnn/src/small_model_inference.py
```

This script builds an equivalent homomorphic architecture, extracting weights from the plaintext model, and runs inference on MNIST. It prints out inference times to the terminal. For convenience, example weights are already included in `palisade_he_cnn/src/weights`.

### Larger neural networks

Scripts to train larger models are included in `palisade_he_cnn/training`. Scripts that run inference with these models are in `palisade_he_cnn/inference`. Due to significant resources required to train and run homomorphic inference with these larger models, weights used in the paper will be added to this repository in the future.

## Citation and Acknowledgements 

Please cite this work as follows:

```
@misc{maloney2024highresolutionconvolutionalneuralnetworks,
      title={High-Resolution Convolutional Neural Networks on Homomorphically Encrypted Data via Sharding Ciphertexts}, 
      author={Vivian Maloney and Richard F. Obrecht and Vikram Saraph and Prathibha Rama and Kate Tallaksen},
      year={2024},
      eprint={2306.09189},
      archivePrefix={arXiv},
      primaryClass={cs.CR},
      url={https://arxiv.org/abs/2306.09189}, 
}
```

In addition to the authors on the supporting manuscript (Vivian Maloney, Freddy Obrect, Vikram Saraph, Prathibha Rama, and Kate Tallaksen), Lindsay Spriggs and Court Climer also contributed to this work by testing the software and integrating it with internal infrastructure.
