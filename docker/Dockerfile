# Basic image for my training jobs. If I want to add some global stuff, do it here.
FROM nvcr.io/nvidia/pytorch:24.10-py3

RUN apt-get update 
RUN apt-get upgrade -y
RUN apt-get install -y apt-utils
RUN apt-get install -y sudo
RUN apt-get install -y wget
RUN apt-get install -y curl
RUN apt-get install -y screen tmux
RUN apt-get install -y htop
RUN apt-get install -y libgl1-mesa-glx
# Add other stuff as it comes to mind
RUN useradd -u 396376 -m marktas
RUN groupadd -g 10776 csft
RUN usermod -aG sudo marktas
RUN echo "marktas:janenba" | chpasswd

WORKDIR /home/marktas
RUN chown -R marktas:csft /home/marktas

USER marktas

RUN python -m pip install --upgrade pip

RUN git config --global --add safe.directory /home/marktas

USER root
# Add here as it goes, the python libraries I use in all my projects


