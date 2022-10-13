## Efficient Architecture Search for Diverse Tasks

[[Paper Link](https://arxiv.org/abs/2204.07554)]

Original [PyTorch](https://pytorch.org/) implementation of DASH (**D**iverse-task **A**rchitecture **S**earc**H**). This repo is built on top
of the [XD-operations](https://github.com/mkhodak/relax) from the paper [Rethinking Neural Operations for Diverse Tasks](https://arxiv.org/abs/2103.15798) and can be used to 
replicate experiments on [NAS-Bench-360](https://nb360.ml.cmu.edu), a suite of ten tasks designed for benchmarking NAS in diverse domains. 

DASH is developed for efficiently solving diverse ML problems beyond well-researched domains such as computer vision and natural language processing. 
Being fast, simple, and broadly applicable, DASH fixes a standard convolutional network (CNN) topology and searches for 
the right kernel sizes and dilation rates that its operations should take on. 
It expands the network capacity to extract features at multiple resolutions for different types of data while only requiring searching over the operation space. 
To speed up the search process, DASH computes the mixture-of-operations needed by weight-sharing using the Fourier diagonalization of convolution, which achieves efficiency improvements over standard baselines both asymptotically and empirically.

## Requirements

To run the code, install the dependencies: 
```bash
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=10.1 -c pytorch
pip install scipy sklearn tqdm ml-collections h5py requests
git clone https://github.com/mkhodak/relax relax
cd relax && pip install -e .
```
A Dockerfile is also provided.

Experiments in the paper are done on [Google Cloud Platform](https://cloud.google.com/) using a single NVIDIA Tesla V100 with the following image: c2-deeplearning-pytorch-1-10-cu110-v20220227-debian-10.


## Experiment with NAS-Bench-360

1. In `./src/data`, run `sh download.sh` to download required datasets by specifying the task name(s) in the `download.sh` file.
2. In the `./src` directory, run `sh run.sh` to reproduce the results of the task name(s) specified in the `run.sh` file.

The commands for replicating the speed test for DASH in also in `./src/run.sh`.

## Experiment with Your Own Architectures, Optimizers, and Tasks

### Preparation

#### For new architectures:
Place the corresponding network implementations under the `./src/networks` folder and complete the `get_model` function in `./src/task_configs.py`.

#### For new optimizers:
Add the optimizers to `./src/optimizers.py`.

#### For new tasks:
1. Add the data loaders to `./src/data_loaders.py` and complete the `get_data` function in `./src/task_configs.py`.
2. Add your customized loss functions and evaluation metrics to `./src/task_utils.py` and complete the `get_metric` function in `./src/task_configs.py`.

#### Then:
Modify the `get_config` function in `./src/task_configs.py` to test the new architectures or the new tasks.

### Run DASH
Under the `./src` directory, run the following command:
```
python3 main.py --dataset your_new_task
```

## Citation
If you find this project helpful, please consider citing our paper:
```bibtex
@inproceedings{shen2022efficient,
  title={Efficient Architecture Search for Diverse Tasks},
  author={Shen, Junhong and Khodak, Mikhail and Talwalkar, Ameet},
    booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
    year={2022}
}
```
Thanks!
