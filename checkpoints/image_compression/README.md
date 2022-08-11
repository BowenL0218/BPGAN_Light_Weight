# Unified Signal Compression Using Generative Adversarial Networks (Light weight version)
Codes for [Unified Signal Compression Using Generative Adversarial Networks](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9053233) (ICASSP 2020), a generative adverdarial networks (GAN) based signal (Image/Speech) compression algorithm.

# Introduction
The proposed unified compression framework (BPGAN) uses a GAN to compress heterogeneous signals. The compressed signal is represented as a latent vector and fed into a generator network that is trained to produce high quality realistic signals that minimize a target objective function. To efficiently quantize the compressed signal, non-uniformly quantized optimal latent vectors are identified by iterative back-propagation with alternating direction method of multipliers (ADMM) optimization performed for each iteration. 

## Framework
![Flow chart](https://github.com/BowenL0218/BPGAN_Light_Weight/blob/main/Images/flowchart.png)
The target signal is first encoded as an initialization to the latent representation by an encoder. Then the latent vector z is optimized according to a reconstruction criterion through interative ADMM back-propagation. The latent variable is also discretized in this iterative back-propagation (BP) process.The quantized latent signal is entropy encoded for further code size reduction. In the proposed framework, the generator parameters are shared between the transmitter and receiver. At the receiver, the generator decodes the latent signal, which is converted back to its original format via post-processing.

## Architecture
The detailed neural network structure of our generator model.
![Generator architecture](https://github.com/BowenL0218/BPGAN_Light_Weight/blob/main/Images/arc.png)

This version supports Quantized Weights, Activations, and Gradients. 
The code for pruning weights by ADMM is also provided.
Configurations of the light-weight audio and image models can be found in the [Slides](https://docs.google.com/presentation/d/1ZCyyqXUCx_55Y0qo-2brLwYs0SR63YFN/edit?usp=sharing&ouid=108057161427597557613&rtpof=true&sd=true).

<!---To run compression, we need run two stage scripts.

The first stage is to get the compressed vector and decompressed spectrums by python code.

The second stage is to transform the spectrums into audios by `Jupyter Lab`.
--->

# 1. Prerequisite

`conda` environment is recommended to install dependencies.

1. Install `Anaconda` or `Miniconda` [[link]](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html#regular-installation)
2. Install required packages using environment.yml:
```shell
conda env create -f environment.yml 
```
This will create an environment named `bpgan_env`, which contains all the dependencies defined in `environment.yml`.

3. Switch to the new environment:
```shell
conda activate bpgan_env
```

If you want to go back to the base conda environment, type 
```shell
conda deactivate
```

# 2. Datasets

### Audio

Audio models are trained and tested on TIMIT dataset. 

The whole process for the Audio data preparation can be done by the following script:

```shell
./scripts/audio_data_prep.sh
```
This shell script will generate a 16-bit mel-spectrogram images with sampling rate of 8K.
The preprocessed spectrogram files are saved in `{$PRJ_ROOT}/datasets/timit/timit_mel_8k`.

### Image

In this version, Image models are trained on [Open Images V6](https://storage.googleapis.com/openimages/web/download.html#download_using_fiftyone) Validation set and tested on [KODAK dataset](http://www.cs.albany.edu/~xypan/research/snr/Kodak.html).

To prepare the training set, please download the `tar.gz` file from [Google Drive](https://drive.google.com/file/d/1G2VjTFswcetZQTQNQrEAAzWT9Q82k0EF/view?usp=sharing), and unzip it into `{$PRJ_ROOT}/datasets/`. 
*Warning: The size of the zip file is about 12GB* 

The KODAK dataset can be prepared by the following script:
```shell
./scripts/image_kodak_prep.sh
```

# 3. Training

Configurations of the light-weight audio and image models can be found in the [Slides](https://docs.google.com/presentation/d/1ZCyyqXUCx_55Y0qo-2brLwYs0SR63YFN/edit?usp=sharing&ouid=108057161427597557613&rtpof=true&sd=true).
The major change is that the activation functions are `relu4`, which is defined as
$$ relu4(x)=min(max(x,0),4).$$
This makes the fixed point conversion easier because the range of the activation function is bounded in both sides.
BPGAN models can be trained by the following script:
### Audio
```shell
./scripts/train_audio.sh
```
The trained model will be saved in `{$PRJ_ROOT}/checkpoints/audio_compression/latest_net_{NETWORK_NAME}.pth`, where `{NETWORK_NAME}` has one of followings:
* `E`: Encoder
* `Decoder`: Decoder
* `Q`: Quantizer
* `D`: Discriminator 

### Image
```shell
./scripts/train_image.sh
```
The trained model will be saved in `{$PRJ_ROOT}/checkpoints/image_compression/latest_net_{NETWORK_NAME}.pth`.

# 4. Pruning and Quantization

In this part, the trained models are going to be downsized by ADMM optimization.

### Audio
```shell
./scripts/reduce_audio_model.sh
```
The output files will be saved as `{$PRJ_ROOT}/checkpoints/audio_compression/ADMM_Q_pruned_net_{NETWORK_NAME}.pth`.

### Image
```shell
./scripts/reduce_image_model.sh
```
The output files will be saved as `{$PRJ_ROOT}/checkpoints/image_compression/ADMM_Q_pruned_net_{NETWORK_NAME}.pth`.

# 5. Testing

## Light Weight Model (8bit Weight/Activation/Gradient, Pruned)
To test light-weight models, run following scripts:
### Audio
```shell
./scripts/test_audio_light.sh
```

### Image
```shell
./scripts/test_image_light.sh
```
The results will be saved in `{$PRJ_ROOT}/ADMM_Results/` with suffix `8bit` 

## Full precision, un-pruned model
### Audio
```shell
./scripts/test_audio.sh
```

### Image
```shell
./scripts/test_image.sh
```
The results will be saved in `{$PRJ_ROOT}/ADMM_Results`.


# 6. (Audio only) Converting Mel-spectrogram to WAV

`{$PRJ_ROOT}/Generate_Audio.ipynb` contains sample codes for converting the generated Melspectrograms (in `png` format) to the audio file (in `wav` format).
You may need to install `Jupyter Notebook` or `Jupyter Lab` to run the notebook.
