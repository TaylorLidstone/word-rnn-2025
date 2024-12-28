# word-rnn-2025
A self-hosted word-rnn that actually works in 2025! 

Back in the day I was a massive fan of  [char-rnn-tensorflow](https://github.com/sherjilozair/char-rnn-tensorflow) and [word-rnn-tensorflow](https://github.com/hunkim/word-rnn-tensorflow). Yeah, I know ChatGPT exists now, but the random and weird outputs that these early programs produced were infinitely more entertaining.

So... here's my version, heavily co-written with Claude and ChatGPT to recreate something that operates in a similar manner. I have only tested this on my specific system, so it may or may not work for you - future or past versions of the required installs might work, I don't know.

# Required Installs
If you're running this on Windows 11 you probably need to install the following: 

Python 3.10.12

WSL 2

Updated NVIDIA graphics driver

Cuda Toolkit | cuda-repo-wsl-ubuntu-12-6-local_12.6.3-1_amd64.deb 

psutil

# Installing Cuda Toolkit 
This can be a bit annoying to install, so here's a walkthrough.

Sign up for a developer account. Then...

1. ```wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin```
2. ```sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600```
3. ```wget https://developer.download.nvidia.com/compute/cuda/12.6.3/local_installers/cuda-repo-wsl-ubuntu-12-6-local_12.6.3-1_amd64.deb```
4. ```sudo dpkg -i cuda-repo-wsl-ubuntu-12-6-local_12.6.3-1_amd64.deb```
5. ```sudo cp /var/cuda-repo-wsl-ubuntu-12-6-local/cuda-56752864-keyring.gpg /usr/share/keyrings/```
6. ```sudo apt-get update```
7. ```sudo apt-get -y install cuda-toolkit-12-6```
8. Add Cuda to PATH by running ```nano ~/.bashrc``` and adding the following to the bottom. 
```
export PATH=/usr/local/cuda-12.6/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-12.6
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PATH=$CUDA_HOME/bin:$PATH
export CPATH=$CUDA_HOME/include:$CPATH
export LIBRARY_PATH=$CUDA_HOME/lib64/stubs:$LIBRARY_PATH
```
9. ```source ~/.bashrc```
9. Check that it works with ```nvcc --version```
