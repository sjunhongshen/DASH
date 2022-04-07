FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# Setup Ubuntu
RUN apt-get update --yes
RUN apt-get install -y make cmake build-essential autoconf libtool rsync ca-certificates git grep sed dpkg curl wget bzip2 unzip llvm libssl-dev libreadline-dev libncurses5-dev libncursesw5-dev libbz2-dev libsqlite3-dev zlib1g-dev mpich htop vim

# Get Miniconda and make it the main Python interpreter
RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
RUN bash ~/miniconda.sh -b -p /opt/conda
RUN rm ~/miniconda.sh
ENV PATH /opt/conda/bin:$PATH
RUN conda create -n pytorch_env python=3.6
RUN echo "source activate pytorch_env" > ~/.bashrc
ENV PATH /opt/conda/envs/pytorch_env/bin:$PATH
ENV CONDA_DEFAULT_ENV pytorch_env
RUN conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=10.1 -c pytorch
RUN pip install scipy sklearn tqdm ml-collections h5py requests

# install XD
RUN git clone https://github.com/mkhodak/relax /code/relax
RUN cd /code/relax && pip install -e .
