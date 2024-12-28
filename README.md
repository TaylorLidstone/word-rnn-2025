# Word-RNN-2025
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

-----

> [!TIP]
> Installing Cuda Toolkit can be a bit annoying, so here's a walkthrough.

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
10. Check that it works with ```nvcc --version```

# Basic Usage
1. Git clone this repo.
2. Place a .txt file in the text_file folder (the more text the better).
3. Open Terminal in the main folder.
4. `python3 train.py --input_file text_file/[yourfilename].txt --model_dir checkpoints`  
This will start to train checkpoints with the file that you have put in the text_file folder, and be aware it takes a **LONG TIME**. You can cancel the training at any point, and if you run the program again, it will resume training from the last checkpoint.
5. Once the training has finished or you have any checkpoint you can begin generating - (There is a difference, Epoch 1 will be noticibly less trained than Epoch 50, for example).
6. `python3 generate.py --model_dir checkpoints --num_words 2000 --temperature 0.7`
> [!CAUTION]
> When you first run the training, or run it after deleting your checkpoints. You will see the following message.  
> *INFO:root:No checkpoints found, starting from scratch.*  
> Please be aware that this step can take up to 20 minutes to run, depending on the size of the .txt file, so be patient!

# Advanced Arguments
## Training
**If you change any of the values in bold - you MUST specify the same argument AND values when generating with that checkpoint!**
| Argument | Description | Default Value |
|----------|-------------|---------------|
| --input_file	| Path to the input text file for training.	| *Required* |
| --model_dir |	Directory to save model checkpoints and processor. | `checkpoints` |
| --batch_size | Number of samples per training batch. | `64` |
| --epochs | Total number of training epochs. | `50` |
| **--embedding_dim** | Size of the embedding vector. | **256** |
| **--rnn_units** | Number of LSTM units in each layer. | **512** |
| **--num_layers** | Number of LSTM layers. | **2** |
| **--seq_length** | Length of input sequences for training. | **50** |

Example:
`python3 train.py --input_file data/lordoftherings.txt --model_dir checkpoints --epochs 20`

## Generating
**If you have changed any of the values in bold during training - make sure that you specify the same argument AND values when generating!**
| Argument | Description | Default Value |
|----------|-------------|---------------|
| --model_dir | Directory containing the saved model checkpoints.	| *Required* |
| --epoch	| Specific checkpoint epoch to load. | *Latest Epoch* |
| --num_words	| Number of words to generate. | `100` |
| --temperature	| Controls randomness in text generation. Range: 0.1–10. | `1.0` |
| --seed_text	| Starting text to seed the generation. | *User Input* |
| **--embedding_dim**	| Size of the embedding vector (must match training). | **256** |
| **--rnn_units**	| Number of LSTM units per layer (must match training). | **512** |
| **--num_layers**	| Number of LSTM layers (must match training). | **2** |
| **--seq_length**	| Length of input sequences (must match training). | **50** |

Example:
`python3 generate.py --model_dir checkpoints --epoch 23 --num_words 2000 --temperature 0.8--seed_text "Gandalf jumped into a "`

# Final Notes:
- If no --seed_text argument is included it will prompt you - BUT you can just press enter and it will run without your input.
- A high --temperature is absolute garbage but great for generating... passphrases...? ```somewhere bluebell good! similarly cooling repayment obvious?” centuries ropes generous.```
- I don't know what I'm doing... So, if you have suggestions or improvements, please fork the repository, make changes, and submit a pull request.
