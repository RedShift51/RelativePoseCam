#FROM pytorch/pytorch:1.7.1-cuda10.1-cudnn7-devel
FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel

RUN apt-get update && yes | apt-get upgrade
RUN mkdir -p /examples

RUN apt-get install -y git python3-pip vim
RUN pip3 install --upgrade pip

RUN pip install numpy scipy matplotlib imageio
RUN pip install numpy easydict Cython progressbar2 tensorboardX
RUN pip install cython

RUN apt-get install libgl1-mesa-glx -y
RUN pip install opencv-python

RUN apt-get update && apt-get install -y \
    git \
    cmake \
    vim \
    build-essential \
    libboost-program-options-dev \
    libboost-filesystem-dev \
    libboost-graph-dev \
    libboost-system-dev \
    libboost-test-dev \
    libeigen3-dev \
    libsuitesparse-dev \
    libfreeimage-dev \
    libgoogle-glog-dev \
    libgflags-dev \
    libglew-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    libcgal-dev \
    libcgal-qt5-dev

RUN apt-get -y install \
    libatlas-base-dev \
    libsuitesparse-dev
RUN git clone https://github.com/ceres-solver/ceres-solver.git --branch 1.14.0
RUN cd ceres-solver && \
    mkdir build && \
    cd build && \
    cmake .. -DCERES_THREADING_MODEL=OPENMP -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF && \
    make -j4 && \
    make install

# Build and install COLMAP
RUN git clone https://github.com/colmap/colmap.git #--branch 3.6

RUN cd colmap && \
    git checkout dev && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make -j4 && \
    make install



