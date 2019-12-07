FROM ubuntu:18.04
MAINTAINER Hung Pham <hungpham2511@gmail.com>

ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN rm -rf /var/lib/apt/lists/* && apt-get update  \
    && apt-get install -y --no-install-recommends apt-utils lsb-release sudo unzip wget

# Install everything as root
RUN mkdir -p ~/git && cd ~/git    \
    && wget --no-check-certificate https://github.com/crigroup/openrave-installation/archive/0.9.0.zip -O openrave-installation.zip  \
    && unzip openrave-installation.zip -d ~/git    \
    && cd openrave-installation-0.9.0 && sudo ./install-dependencies.sh

# OpenSceneGraph
ENV OSG_COMMIT 1f89e6eb1087add6cd9c743ab07a5bce53b2f480
RUN mkdir -p /usr/src && cd /usr/src \
    && wget -q https://github.com/openscenegraph/OpenSceneGraph/archive/${OSG_COMMIT}.zip -O OpenSceneGraph.zip \
    && unzip -q OpenSceneGraph.zip -d /usr/src \
    && cd /usr/src/OpenSceneGraph-${OSG_COMMIT} && mkdir build && cd build \
    && cmake .. -DDESIRED_QT_VERSION=4 && make -j `nproc` && make install   \
    && rm -rf /usr/src/OpenSceneGraph-${OSG_COMMIT}/build

# FCL
RUN cd /usr/src \
    && wget -q https://github.com/flexible-collision-library/fcl/archive/0.5.0.zip -O fcl.zip \
    && unzip -q fcl.zip -d /usr/src \
    && cd /usr/src/fcl-0.5.0 && mkdir build && cd build \
    && cmake .. && make -j `nproc` && make install  \
    && rm -rf /usr/src/fcl-0.5.0/build

# OpenRAVE
ENV RAVE_COMMIT 7c5f5e27eec2b2ef10aa63fbc519a998c276f908
RUN pip install --upgrade sympy==0.7.1

RUN cd /usr/src \
    && wget -q https://github.com/rdiankov/openrave/archive/${RAVE_COMMIT}.zip -O openrave.zip \
    && unzip -q openrave.zip -d /usr/src \
    && cd /usr/src/openrave-${RAVE_COMMIT} && mkdir build && cd build \
    && cmake -DODE_USE_MULTITHREAD=ON -DOSG_DIR=/usr/local/lib64/ .. \
    && make -j `nproc` && make install  \
    && rm -rf /usr/src/openrave-${RAVE_COMMIT}/build

# Other deps
RUN sudo DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends python-dev python3-dev python3-venv python-tk
# RUN sudo apt-get install -y --no-install-recommends python-dev python3-dev python3-venv

# User
ARG user=hung
ARG group=toppra
ARG uid=1000
ARG gid=1000
ARG home=/home/${user}
RUN mkdir -p /etc/sudoers.d \
	&& groupadd -g ${gid} ${group} \
	&& useradd -d ${home} -u ${uid} -g ${gid} -m -s /bin/bash ${user} \
	&& echo "${user} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/sudoers_${user}
USER ${user}
WORKDIR ${home}

