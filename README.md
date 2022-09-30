Our code is implemented based on  "Deep Fusion Clustering Network"<br>



Our code on all datasets will be released for further study after the paper is accepted.<br>

## Installation

* Windows 10 or Linux 18.04

* Python 3.7.5

* [Pytorch (1.2.0+)](https://pytorch.org/)

* Numpy 1.18.0

* Sklearn 0.21.3

* Torchvision 0.3.0

* Matplotlib 3.2.1

* Info-nce-pytorch 0.1.4

* scipy 1.6.2

  

## Code Structure & Usage

Here we provide an implementation of DCL-MGI_DFCN in PyTorch, along with an execution example on the Dblp、Cite and Acm datasets.  The repository is organised as follows:

- `load_data.py`: processes the dataset before passing to the network.
- `DFCN.py`: defines the architecture of the whole network.
- `IGAE.py`: defines the improved graph autoencoder.
- `AE.py`: defines the autoencoder.
- `opt.py`: defines some hyper-parameters.
- utils.py`: defines the lr-policy, metrics, and others.
- train.py`: the entry point for training and testing.

Finally, `main.py` puts all of the above together and may be used to execute a full training run on Dblp、Cite and Acm datasets.

<span id="jump2"></span>
