FROM secretflow/ubuntu-base-ci:latest

LABEL maintainer="secretflow-contact@service.alipay.com"

ARG TARGETPLATFORM

# change dash to bash as default shell
RUN ln -sf /bin/bash /bin/sh

RUN apt update \
    && apt upgrade -y \
    && apt install -y gcc-11 g++-11 libasan6 \
    git wget curl unzip autoconf make lld-15 \
    cmake ninja-build vim-common libgl1 libglib2.0-0 

# RUN DEBIAN_FRONTEND=noninteractive apt install -y tzdata
RUN apt clean \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 100 \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 100 \
    && update-alternatives --install /usr/bin/ld.lld ld.lld /usr/bin/ld.lld-15 100 

# clang is required on arm64 platform
RUN if [ "$TARGETPLATFORM" = "linux/arm64" ] ; then apt install -y clang-15 \
    && apt clean \
    && update-alternatives --install /usr/bin/clang clang /usr/bin/clang-15 100 \
    && update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-15 100 \
; fi


# amd64 is only reqiured on amd64 platform
RUN if [ "$TARGETPLATFORM" = "linux/amd64" ] ; then apt install -y nasm ; fi

# install conda
# RUN if [ "$TARGETPLATFORM" = "linux/arm64" ] ; then CONDA_ARCH=aarch64 ; else CONDA_ARCH=x86_64 ; fi \
#     && wget https://repo.anaconda.com/miniconda/Miniconda3-py310_24.3.0-0-Linux-$CONDA_ARCH.sh \
#     && bash Miniconda3-py310_24.3.0-0-Linux-$CONDA_ARCH.sh -b \
#     && rm -f Miniconda3-py310_24.3.0-0-Linux-$CONDA_ARCH.sh \
#     && /root/miniconda3/bin/conda init


ENV http_proxy=http://192.168.109.37:7890
ENV https_proxy=http://192.168.109.37:7890


# Add conda to path
# ENV PATH="/root/miniconda3/bin:${PATH}" 


# install bazel 
RUN if [ "$TARGETPLATFORM" = "linux/arm64" ] ; then BAZEL_ARCH=arm64 ; else BAZEL_ARCH=amd64 ; fi \
    && wget https://github.com/bazelbuild/bazelisk/releases/download/v1.20.0/bazelisk-linux-$BAZEL_ARCH \
    && mv bazelisk-linux-$BAZEL_ARCH /usr/bin/bazel \
    && chmod +x /usr/bin/bazel 


# Add env
# ENV NAME="cipherdm"
# RUN conda create -n $ENV_NAME python=3.8 -y \
#     && source activate $ENV_NAME

RUN pip install --upgrade pip\
    && pip install grpcio>=1.42.0!=1.48.0 \
    && RUN pip install numpy>=1.22.0 \
    && RUN pip install cloudpickle>=2.0.0 \
    && RUN pip install multiprocess>=0.70.12.2 \
    && RUN pip install cachetools>=5.0.0 \
    && RUN pip install protobuf>=4 \
    && RUN pip install jax[cpu]>=0.4.16 \
    && RUN pip install termcolor>=2.0.0 \
    && RUN pip install pandas>=1.4.2 \
    && RUN pip install flax \
    && RUN pip install scikit-learn \
    && RUN pip install absl-py>=1.1.0 \
    && RUN pip install imageio \
    && RUN pip install tqdm \
    && RUN pip install tqdm \
    && RUN pip install opencv-python \
    && RUN pip install spu

RUN if [ "$(uname -m)" = "x86_64" ] && [ "$(uname -s)" = "Linux" ]; then \
        pip install tensorflow-cpu>=2.12.0; \
    else \
        pip install tensorflow>=2.12.0; \
    fi

RUN if [ "$(uname -m)" = "aarch64" ]; then \
        pip install h5py!=3.11.0; \
    fi


# run as root for now

COPY . /home/admin/CipherDM/

WORKDIR /home/admin/CipherDM/

ENTRYPOINT  ["sh", "/home/admin/CipherDM//eval.sh"]



