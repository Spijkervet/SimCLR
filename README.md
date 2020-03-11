# SimCLR
PyTorch implementation of SimCLR: A Simple Framework for Contrastive Learning of Visual Representations by T. Chen et al.


## What is SimCLR?
SimCLR is a "simple framework for contrastive learning of visual representations". The contrastive prediction task is defined on pairs of augmented examples, resulting in 2N examples per minibatch. Two augmented versions of an image are considered as a correlated, "positive" pair (x_i and x_j). The remaining 2(N - 1) augmented examples are considered negative examples. The contrastive prediction task aims to identify x_j in the set of negative examples for a given x_i.


![GitHub Logo](/media/architecture.png)

## Usage
Run the following command to setup a conda environment:
```
sh setup.sh
conda activate simclr
```

Or alternatively with pip:
```
pip install -r requirements.txt
```

Then, simply run:
```
python main.py
```

### Dependencies
```
torch
torchvision
tensorboard
sacred
pyyaml
```