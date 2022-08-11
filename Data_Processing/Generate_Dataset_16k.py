import librosa
import numpy as np
import os
#from ourLTFATStft import LTFATStft
#import ltfatpy
import imageio
import math
import glob
import tqdm


fft_hop_size = 128
fft_window_length = 512
clipBelow = -10
#anStftWrapper = LTFATStft()
sampling_fre = 16000
imagesize = 128
trunk_step = 16
data_root = "../../datasets/timit/data/TEST"
bit_num = 16
count = 0
phase = "test"
n_mels=128
outroot = "../../datasets/timit/timit_mel_{}k_{}_{}bit".format(sampling_fre//1000,phase,bit_num)
mel_matrix = librosa.filters.mel(sr=sampling_fre,n_fft=fft_window_length,n_mels=n_mels)
if not os.path.exists(outroot):
    os.mkdir(outroot)
#for root,dirs,files in os.walk(data_root):
for name in tqdm.tqdm(glob.glob(os.path.join(data_root, "*/*/*.wav"))):
    audio,sr = librosa.core.load(name,sr=sampling_fre,mono=False)
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels, n_fft=fft_window_length, hop_length=fft_hop_size)
    mel_db = librosa.power_to_db(mel, ref=1.0)
    mel_db = np.maximum(mel_db, clipBelow)
    C = -1/(clipBelow)
    mel_db = mel_db*C + 1 

    dir1, dir2, name = os.path.split(name)[0].split("/")[-2], os.path.split(name)[0].split("/")[-1], os.path.split(name)[1].split(".")[0]
    if phase == "train":
        for i in range((mel_db.shape[1]-imagesize)//trunk_step):
            sliced = mel_db[:,i*trunk_step:(i+1)*trunk_step+imagesize]
            sliced = np.round(sliced[0:n_mels,:]*(2**(bit_num)-1))
            filename = os.path.join(outroot,dir1+"_"+dir2+"_"+name+str(i)+".png")
            if bit_num == 16:
                imageio.imwrite(filename,sliced.astype(np.uint16))
            elif bit_num == 8:
                imageio.imwrite(filename,sliced.astype(np.uint8))
            else:
                raise NotImplementedError
    elif phase == "test":
        mel_db = np.round(mel_db * (2** (bit_num)-1))
        filename = os.path.join(outroot,dir1+"_"+dir2+"_"+name+".png")
        if bit_num == 16:
            imageio.imwrite(filename, mel_db.astype(np.uint16))
        elif bit_num == 8:
            imageio.imwrite(filename, mel_db.astype(np.uint8))
        else:
            raise NotImplementedError
        '''
        for i in range(math.ceil(mel_db.shape[1]/imagesize)):
            if (i+1)*imagesize<= mel_db.shape[1]:
                sliced = mel_db[:,i*imagesize:(i+1)*imagesize]
            else:
                sliced = mel_db[:,i*imagesize:]
                sliced = np.pad(sliced,(0,imagesize-sliced.shape[1]),'constant')
            sliced = np.round(sliced[0:n_mels,:]*(2**(bit_num)-1))
            if bit_num == 16:
                imageio.imwrite(filename,sliced.astype(np.uint16))
            elif    bit_num == 8:
                imageio.imwrite(filename,sliced.astype(np.uint8))
            else :
                raise NotImplementedError

        '''

