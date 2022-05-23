import torch
import torchaudio
import torch.nn as nn
import torchaudio.transforms as T
import librosa
from torchvision import transforms
from tqdm import tqdm
from enum import Enum
import numpy as np
import random
import os
import pandas as pd
from rnnoise_wrapper import RNNoise
from sklearn.preprocessing import MinMaxScaler

### 1. ToMFCC1에 f0, f0 1차 차분, f0 2차 차분 추가해줌 --> [batch_size, 3, 14, 400] 반환
### 2. 성별, 연령, 방언 모두 사용 가능하도록 변경 --> AudioData(~, mode=Mode.AGE, ~)
###    - 성별: Female = 0 / male = 1
###    - 연령: 효림이 ADL이랑 동일
###    - 방언: 수도권 = 0 / 전라도 = 1 / 경상도 = 2 / 충청도 = 3 / 강원도 = 4 / 제주도 = 5

"""
<<음성 데이터 Data Loader>>

폴더명을 입력하면 폴더 내부에 있는 모든 .wav 데이터들에 대해 labeling과 필요한 전처리를 모두 마치고
원하는 batch size만큼 반환

<Class 목록>
 1) Preprocess: 전처리 방법
 2) Mode: 불러오고 싶은 라벨의 종류 지정 (성별, 연령, 방언 중 선택)
 3) AudioData: 지정하는 batch_size만큼 데이터를 바로 Input Data의 형태로 변환하여 차례로 반환
 4) AudioInput: torchaudio로 .wav 파일 불러오기 (파일 경로를 list로 입력, train data인 경우 shuffle=True)
 5) Spectrogram: torchaudio로 spectrogram 반환
 6) MelSpec1: torchaudio로 Mel Spectrogram을 계산하여 1차 차분값과 stack한 결과 반환 (torch.Size([2, 80, 100000]))
 7) MelSpec2: torchaudio로 Mel Spectrogram을 계산하여 1차 차분값과 cat한 결과 반환 (torch.Size([200000, 160]))
 6) ToMFCC1: librosa로 MFCC와 f0 주파수를 계산하여 [14, ~] tensor를 만들고 1차 차분, 2차 차분값을 쌓은 결과 반환 (torch.Size([3, 14, 400])으로 padding)
 7) ToMFCC2: 위와 동일하지만, 쌓지 않고 연결해서 [40, 400] 형태의 결과 반환
 7) WaveLen: directory 내의 .wav 파일 길이 dictionary 반환
 8) Files: directory 내의 .wav 파일 중 사용 가능한 파일들 필터링('RIFF Header not found' error 때문에 .load()가 안 되는 파일 제외)

<Data Loader 출력 구성요소>
 - label: 분류 라벨
 - input: 모델의 최종 input으로 활용될 tensor

<출력 예시> 

{'input': tensor([[[[-6.5308e+02, -6.5308e+02, -6.5308e+02,  ...,  0.0000e+00,
             0.0000e+00,  0.0000e+00],
           [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  ...,  0.0000e+00,
             0.0000e+00,  0.0000e+00],
           [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  ...,  0.0000e+00,
             0.0000e+00,  0.0000e+00],
           ...,
           [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  ...,  0.0000e+00,
             0.0000e+00,  0.0000e+00],
           [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  ...,  0.0000e+00,
             0.0000e+00,  0.0000e+00],
           [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  ...,  0.0000e+00,
             0.0000e+00,  0.0000e+00]],

          [[ 7.7874e-15,  7.7874e-15,  7.7874e-15,  ...,  0.0000e+00,
             0.0000e+00,  0.0000e+00],
           [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  ...,  0.0000e+00,
             0.0000e+00,  0.0000e+00],
           [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  ...,  0.0000e+00,
             0.0000e+00,  0.0000e+00],
           ...,
           [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  ...,  0.0000e+00,
             0.0000e+00,  0.0000e+00],
           [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  ...,  0.0000e+00,
             0.0000e+00,  0.0000e+00],
show more (open the raw output data in a text editor) ...

             0.0000e+00,  0.0000e+00],
           [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  ...,  0.0000e+00,
             0.0000e+00,  0.0000e+00],
           [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  ...,  0.0000e+00,
             0.0000e+00,  0.0000e+00]]]]),
 'label': tensor([2])}
 """


class Mode(Enum):
    AGE = 0
    GENDER = 1
    DIALECT = 2
    MULTITASK = 3

class Train(Enum):
    TRAIN = 0
    VAL = 1

class AudioData(torch.utils.data.Dataset):
    def __init__(
            self,
            file: str,
            shuffle: bool,
            sample_num: int,
            mode: Mode,
            train: Train,
            balance=False
    ):
        self.shuffle = shuffle
        self.sample_num = sample_num
        self.file = pd.read_pickle(file)
        self.mode = mode
        self.train = train

        self.file = self.file[(self.file['Age'] != 'NotProvided') & (self.file['Gender'] != 'NotProvided') &
                              (self.file['Dialect'] != 'NotProvided') & (self.file['Dialect'] != '기타(외국)')]
        self.file = self.file.iloc[:2010000,:]
        if balance:
            self.file1 = self.file[(self.file['Age'] == '11~19') & (self.file['Gender'] == 'Male') & (self.file['Dialect'] == '제주도')]

            self.file = self.file[~((self.file['Age'] == '11~19') & (self.file['Gender'] == 'Male') & (self.file['Dialect'] == '제주도'))] 

            self.file = self.file.groupby(['Age','Gender','Dialect']).sample(self.sample_num, random_state=42).reset_index()
            ####
            self.file = pd.concat([self.file, self.file1]).reset_index()
            self.sample_num = self.sample_num * 59 + self.file1.shape[0]
        if self.shuffle:
            self.file = self.file.sample(frac=1, random_state=42).reset_index(drop=True)
        else:
            pass
        
        # wav_files = [j for j in os.listdir(self.path) if j.endswith('.wav')]  # 디렉토리 내의 모든 .wav 파일 가져옴
        data = []

        ####

        

        for i in tqdm(range(self.sample_num)):  # self.file.shape[0] : 나중에 40 이걸로 바꾸기 // self.sample_num
            label = [[0] for i in range(3)]
            directory = self.file.iloc[i]['File_Path']
            if directory.split('/')[3] == '명령어 음성(소아, 유아)':
                continue
        # 연령 라벨 가져오기
            #if self.file.loc[i, 'Age'] == "3~10":
            #    label[0] = 0
            #elif self.file.loc[i, 'Age'] == "11~19":
            #    label[0] = 1
            #elif self.file.loc[i, 'Age'] == "20~29":
            #    label[0] = 2
            #elif self.file.loc[i, 'Age'] == "30~39":
            #    label[0] = 2
            #elif self.file.loc[i, 'Age'] == "40~49":
            #    label[0] = 3
            #elif self.file.loc[i, 'Age'] == "50~59":
            #    label[0] = 3
            #elif self.file.loc[i, 'Age'] == "60~69":
            #    label[0] = 4
            #elif self.file.loc[i, 'Age'] == "over70":
            #    label[0] = 4
            #else:
            #    continue
           
            if self.file.loc[i, 'Age'] == "11~19":
                label[0] = 0
            elif self.file.loc[i, 'Age'] == "20~29":
                label[0] = 1
            elif self.file.loc[i, 'Age'] == "30~39":
                label[0] = 1
            elif self.file.loc[i, 'Age'] == "40~49":
                label[0] = 2
            elif self.file.loc[i, 'Age'] == "50~59":
                label[0] = 2
            elif self.file.loc[i, 'Age'] == "60~69":
                label[0] = 3
            elif self.file.loc[i, 'Age'] == "over70":
                label[0] = 3
            else:
                continue
                

            if self.file.loc[i, 'Gender'] == 'Female':
                label[1] = 0
            elif self.file.loc[i, 'Gender'] == 'Male':
                label[1] = 1
            else:
                continue



            if self.file.loc[i, 'Dialect'] == '수도권':
                label[2] = 0
            elif self.file.loc[i, 'Dialect'] == '전라도':
                label[2] = 1
            elif self.file.loc[i, 'Dialect'] == '경상도':
                label[2] = 2
            elif self.file.loc[i, 'Dialect'] == '충청도':
                label[2] = 3
            elif self.file.loc[i, 'Dialect'] == '강원도':
                label[2] = 4
            elif self.file.loc[i, 'Dialect'] == '제주도':
                label[2] = 5
            else:
                continue
            
            if self.mode == Mode.AGE:
                label = label[0]
            elif self.mode == Mode.GENDER:
                label = label[1]
            elif self.mode == Mode.DIALECT:
                label = label[2]
            else:
                pass


            tmp = {'Directory': directory,
                    'Label' : label,
                    'train' : self.train}
            try:
                tmp = mfcc1_pre(tmp)
            except Exception as e:
                print(e)
                continue
            data.append(tmp)
        
        self.data = data


    def __len__(self):  # 데이터셋의 길이
        return len(self.data)

    def __getitem__(self, index):  # 데이터셋에서 한 개의 데이터 가져오기
        

        # Mode에 따라 다른 라벨 반환

        result = {'input' : self.data[index]['input'], 'label' : self.data[index]['Label']}

        return result


class AudioInput(object):
    def __init__(self):
        pass

    def __call__(self, data):
        file_dir = data['Directory']
        train = data['train']
        file_name = file_dir.split('/')[-1][:-4]
        if train==Train.TRAIN:
            file = f'pt/train/{file_name}.pt'
        else:
            file = f'pt/validation/{file_name}.pt'
        _, sr = torchaudio.load(file_dir)
        signal = torch.load(file)

        data['signal'] = signal
        data['sample_rate'] = sr

        return data




class ToMFCC1(object):
    def __init__(self):
        pass

    def __call__(self, data):
        
        signal = data['signal']
        sr = data['sample_rate']
        '''
        audio_mfcc = signal[0, :-1, :]
        delta1 = signal[1,:-1,:]
        delta2 = signal[2,:-1,:]
        f0_delta1 = signal[1,-1,:].view(1, -1)
        f0_delta2 = signal[2,-1,:].view(1, -1)
        
        ##scaling
        mfcc_scaler = MinMaxScaler()
        delta1_scaler = MinMaxScaler()
        delta2_scaler = MinMaxScaler()
        f0_scaler = MinMaxScaler()
        f0_delta1_scaler = MinMaxScaler()
        f0_delta2_scaler = MinMaxScaler()

        audio_mfcc = mfcc_scaler.fit_transform(audio_mfcc)
        delta1 = delta1_scaler.fit_transform(delta1)
        delta2 = delta2_scaler.fit_transform(delta2)
        f0_delta1 = f0_delta1_scaler.fit_transform(f0_delta1)
        f0_delta2 = f0_delta2_scaler.fit_transform(f0_delta2)
        
        f0 = signal[0,-1,:]
        f0 = f0.reshape(-1,1)
        f0_scaler.fit(f0)
        f0 = f0_scaler.transform(f0)
        f0 = f0.squeeze()
        
        delta1 = torch.FloatTensor(delta1)
        delta2 = torch.FloatTensor(delta2)
        f0_delta1 = torch.FloatTensor(f0_delta1)
        f0_delta2 = torch.FloatTensor(f0_delta2)
        audio_mfcc = torch.FloatTensor(audio_mfcc)
        f0 = torch.FloatTensor(f0)


        audio_mfcc = np.concatenate((audio_mfcc, f0.reshape((1, -1))))
        delta1 = torch.cat((delta1, f0_delta1))
        delta2 = torch.cat((delta2, f0_delta2))

        audio_mfcc = torch.from_numpy(audio_mfcc)

        mfcc_result = torch.stack([audio_mfcc, delta1, delta2])
        '''
        signal[0,-1,:][signal[0,-1,:]>500]=0
        f0 = signal[0,-1,:]
        signal[1,-1,:] = torch.FloatTensor(librosa.feature.delta(f0)).view(1, -1)
        signal[2,-1,:] = torch.FloatTensor(librosa.feature.delta(f0, order=2)).view(1, -1)
        mfcc_result = mfcc_pad1(signal)

        data['input'] = mfcc_result

        return data


def mfcc_pad1(data, length=400):  # 126000을 데이터 상황에 맞추어 400으로 바꾸어줌
    if data.shape[2] > length:
        #padded = torch.cat([data, torch.ones(data.shape[0], data.shape[1], 1) * -1], dim=2)
        padded = data[:, :, 0:length]
    else:
        #padded = torch.cat([data, torch.ones(data.shape[0], data.shape[1], 1) * -1], dim=2)
        padded = torch.cat([data, torch.zeros(data.shape[0], data.shape[1], length - data.shape[2])], dim=2)
        

    return padded

mfcc1_pre = transforms.Compose([AudioInput(), ToMFCC1()])

