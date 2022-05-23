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


class Preprocess(Enum):  # 공유 드라이브 '프로젝트 진행 방향' 문서에 있는 2가지 전처리 방법 모두 사용 가능
    MFCC1 = 0  # [3, 13, 126000] 반환
    MFCC2 = 1  # [39, 126000] 반환
    SPECTROGRAM = 2
    MEL1 = 3
    MEL2 = 4
    RAW = 5

class Mode(Enum):
    AGE = 0
    GENDER = 1
    DIALECT = 2

class AudioData(torch.utils.data.Dataset):
    def __init__(
            self,
            file: str,
            shuffle: bool,
            sample_num: int,
            preprocess: Preprocess,
            mode: Mode,
            balance=True
    ):

        self.preprocess = preprocess
        self.shuffle = shuffle
        self.sample_num = sample_num
        self.file = pd.read_csv(file)
        self.mode = mode

        self.file = self.file[(self.file['Age'] != 'NotProvided') & (self.file['Gender'] != 'NotProvided') &
                              (self.file['Dialect'] != 'NotProvided') & (self.file['Dialect'] != '기타(외국)')]
        
        self.file1 = self.file[(self.file['Age'] == '11~19') & (self.file['Gender'] == 'Male') & (self.file['Dialect'] == '제주도')]

        
        if balance:
            self.file = self.file[~((self.file['Age'] == '11~19') & (self.file['Gender'] == 'Male') & (self.file['Dialect'] == '제주도'))] 

            self.file = self.file.groupby(['Age','Gender','Dialect']).sample(self.sample_num).reset_index()
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

        label = None
        
        if self.mode == Mode.AGE:
            for i in tqdm(range(self.sample_num)):  # self.file.shape[0] : 나중에 40 이걸로 바꾸기 // self.sample_num

            # 연령 라벨 가져오기
                if self.file.loc[i, 'Age'] == "3~10":
                    label = 0
                elif self.file.loc[i, 'Age'] == "11~19":
                    label = 1
                elif self.file.loc[i, 'Age'] == "20~29":
                    label = 2
                elif self.file.loc[i, 'Age'] == "30~39":
                    label = 2
                elif self.file.loc[i, 'Age'] == "40~49":
                    label = 3
                elif self.file.loc[i, 'Age'] == "50~59":
                    label = 3
                elif self.file.loc[i, 'Age'] == "60~69":
                    label = 4
                elif self.file.loc[i, 'Age'] == "over70":
                    label = 4
                else:
                    continue
                
                directory = self.file.iloc[i]['File_Path']
                tmp = {'Directory': directory,
                        'Label' : label}
                if self.preprocess == Preprocess.SPECTROGRAM:
                    tmp = spec_pre(tmp)
                elif self.preprocess == Preprocess.MEL1:
                    tmp = mel1_pre(tmp)
                elif self.preprocess == Preprocess.MEL2:
                    tmp = mel2_pre(tmp)
                elif self.preprocess == Preprocess.MFCC1:
                    tmp = mfcc1_pre(tmp)
                elif self.preprocess == Preprocess.MFCC2:
                    tmp = mfcc2_pre(tmp)
                elif self.preprocess == Preprocess.RAW:
                    tmp = raw_pre(tmp)

                
                data.append(tmp)
        
        elif self.mode == Mode.GENDER:
            for i in tqdm(range(self.sample_num)):
                if self.file.loc[i, 'Gender'] == 'Female':
                    label = 0
                elif self.file.loc[i, 'Gender'] == 'Male':
                    label = 1
                else:
                    continue
                    
                directory = self.file.iloc[i]['File_Path']
                tmp = {'Directory': directory,
                        'Label' : label}
                if self.preprocess == Preprocess.SPECTROGRAM:
                    tmp = spec_pre(tmp)
                elif self.preprocess == Preprocess.MEL1:
                    tmp = mel1_pre(tmp)
                elif self.preprocess == Preprocess.MEL2:
                    tmp = mel2_pre(tmp)
                elif self.preprocess == Preprocess.MFCC1:
                    tmp = mfcc1_pre(tmp)
                elif self.preprocess == Preprocess.MFCC2:
                    tmp = mfcc2_pre(tmp)
                elif self.preprocess == Preprocess.RAW:
                    tmp = raw_pre(tmp)

                data.append(tmp)
        
        elif self.mode == Mode.DIALECT:
            for i in tqdm(range(self.sample_num)):
                if self.file.loc[i, 'Dialect'] == '수도권':
                    label = 0
                elif self.file.loc[i, 'Dialect'] == '전라도':
                    label = 1
                elif self.file.loc[i, 'Dialect'] == '경상도':
                    label = 2
                elif self.file.loc[i, 'Dialect'] == '충청도':
                    label = 3
                elif self.file.loc[i, 'Dialect'] == '강원도':
                    label = 4
                elif self.file.loc[i, 'Dialect'] == '제주도':
                    label = 5
                else:
                    continue
                
                directory = self.file.iloc[i]['File_Path']
                tmp = {'Directory': directory,
                        'Label' : label}
                if self.preprocess == Preprocess.SPECTROGRAM:
                    tmp = spec_pre(tmp)
                elif self.preprocess == Preprocess.MEL1:
                    tmp = mel1_pre(tmp)
                elif self.preprocess == Preprocess.MEL2:
                    tmp = mel2_pre(tmp)
                elif self.preprocess == Preprocess.MFCC1:
                    tmp = mfcc1_pre(tmp)
                elif self.preprocess == Preprocess.MFCC2:
                    tmp = mfcc2_pre(tmp)
                elif self.preprocess == Preprocess.RAW:
                    tmp = raw_pre(tmp)

                
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
        denoiser = RNNoise()
        audio = denoiser.read_wav(file_dir)
        sr = audio.frame_rate
        denoised_audio = denoiser.filter(audio)
        signal = torch.tensor(denoised_audio.get_array_of_samples()) / 2**15
        
        #signal, sr = torchaudio.load(file_dir)

        data['signal'] = signal
        data['sample_rate'] = sr

        return data


class Spectrogram(object):
    def __init__(self):
        pass

    def __call__(self, data):
        signal = data["signal"]
        sr = data['sample_rate']

        self.n_fft = int(np.ceil(0.025 * sr))
        self.win_length = int(np.ceil(0.025 * sr))
        self.hop_length = int(np.ceil(0.01 * sr))

        spec = nn.Sequential(
            T.Spectrogram(
                n_fft=self.n_fft,
                win_length=self.win_length,
                hop_length=self.hop_length
            ),
            T.AmplitudeToDB()
        )

        data['Spectrogram'] = spec(signal)
        data['input'] = spec(signal)

        return data


class MelSpec1(object):
    def __init__(self):
        pass

    def __call__(self, data):
        signal = data["signal"]
        sr = data['sample_rate']

        mel_input = signal[0].numpy()

        mel_spec = librosa.power_to_db(
            librosa.feature.melspectrogram(y=signal[0].numpy(),
                                           sr=sr, n_mels=128, fmax=8000, fmin=100),
            ref=np.max)  # [128, 5000]
        mel_delta = librosa.feature.delta(mel_spec)

        mel_spec = torch.Tensor(mel_spec)
        mel_delta = torch.Tensor(mel_delta)

        output = torch.stack([mel_spec, mel_delta])  # torch.Size((2, 128, 400])

        data['MelSpectrogram'] = output
        data['input'] = output

        return data


class MelSpec2(object):
    def __init__(self):
        pass

    def __call__(self, data):
        signal = data["signal"]
        sr = data['sample_rate']
        if signal.ndim != 1:
            signal = signal[0]
        
        mel_input = signal.numpy()
        

        mel_spec = librosa.power_to_db(
            librosa.feature.melspectrogram(y=signal.numpy(),
                                           sr=sr, n_mels=128, fmax=8000, fmin=100), #### 명령어음성 상황에 맞추어 0:150으로 조정
            ref=np.max)  # [128, 5000]

        mel_delta = librosa.feature.delta(mel_spec)

        output = torch.Tensor(np.concatenate([mel_spec, mel_delta], axis=0)).transpose(0, 1)  # torch.Size([t, 256])
        output = mel_pad2(output)
        data['MelSpectrogram'] = output
        data['input'] = output

        return data



class ToMFCC1(object):
    def __init__(self):
        pass

    def __call__(self, data):
        signal = data['signal']
        sr = data['sample_rate']
        self.n_fft = int(np.ceil(0.025 * sr))
        self.win_length = int(np.ceil(0.025 * sr))
        self.hop_length = int(np.ceil(0.01 * sr))
        audio_mfcc = torch.FloatTensor(librosa.feature.mfcc(y=signal.numpy().reshape(-1),
                                                            sr=sr,
                                                            n_mfcc=13,
                                                            n_fft=self.n_fft,
                                                            hop_length=self.hop_length))
        #f0, voiced_flag, voiced_probs = librosa.pyin(y=signal.numpy().reshape(-1),
        #                                             sr=sr,
        #                                             frame_length=n_fft,
        #                                             hop_length=hop_length,
        #                                             fmin=librosa.note_to_hz('C2'),
        #                                             fmax=librosa.note_to_hz('C7'))
        f0 = torch.FloatTensor(librosa.yin(y=signal.numpy().reshape(-1),
                     sr=sr,
                     frame_length = self.n_fft,
                     hop_length = self.hop_length,
                     fmin=librosa.note_to_hz('C2'),
                     fmax=librosa.note_to_hz('C7')
            ))
        f0[f0 > 500] = 0
        
        f0[torch.isnan(f0)] = 0
        delta1 = torch.FloatTensor(librosa.feature.delta(audio_mfcc))
        delta2 = torch.FloatTensor(librosa.feature.delta(audio_mfcc, order=2))
        # mfcc_result = np.concatenate((audio_mfcc, delta1, delta2), axis=0)
        f0_delta1 = torch.FloatTensor(librosa.feature.delta(f0)).view(1, -1)
        f0_delta2 = torch.FloatTensor(librosa.feature.delta(f0, order=2)).view(1, -1)

        audio_mfcc = np.concatenate((audio_mfcc, f0.reshape((1, -1))))
        delta1 = torch.cat((delta1, f0_delta1))
        delta2 = torch.cat((delta2, f0_delta2))

        audio_mfcc = torch.from_numpy(audio_mfcc)

        mfcc_result = torch.stack([audio_mfcc, delta1, delta2])
        mfcc_result = mfcc_pad1(mfcc_result)

        data['MFCC'] = mfcc_result
        data['input'] = mfcc_result

        return data


class ToMFCC2(object):
    def __init__(self):
        pass

    def __call__(self, data):
        signal = data['signal']
        sr = data['sample_rate']

        self.n_fft = int(np.ceil(0.025 * sr))
        self.win_length = int(np.ceil(0.025 * sr))
        self.hop_length = int(np.ceil(0.01 * sr))

        audio_mfcc = librosa.feature.mfcc(y=signal.numpy().reshape(-1),
                                          sr=sr,
                                          n_mfcc=13,
                                          n_fft=self.n_fft,
                                          hop_length=self.hop_length)
        
        #f0, voiced_flag, voiced_probs = librosa.pyin(y=signal.numpy().reshape(-1),
        #                                     sr=sr,
        #                                     frame_length = self.n_fft,
        #                                     hop_length = self.hop_length,
        #                                     fmin=librosa.note_to_hz('C2'),
        #                                     fmax=librosa.note_to_hz('C7'))
        
        f0 = librosa.yin(y=signal.numpy().reshape(-1),
                     sr=sr,
                     frame_length = self.n_fft,
                     hop_length = self.hop_length,
                     fmin=librosa.note_to_hz('C2'),
                     fmax=librosa.note_to_hz('C7')
            )
        f0[f0 > 500] = 0
        
        f0[np.isnan(f0)] = 0
        
        delta1 = librosa.feature.delta(audio_mfcc)
        #delta1 = delta1 / np.linalg.norm(delta1)
        delta2 = librosa.feature.delta(audio_mfcc, order=2)
        #delta2 = delta2 / np.linalg.norm(delta2)
        mfcc_result = np.concatenate((audio_mfcc.transpose(), delta1.transpose(),
                                      delta2.transpose(),  f0.transpose().reshape((-1,1))), axis=1)

        mfcc_result = torch.from_numpy(mfcc_result)
        mfcc_result = mfcc_pad2(mfcc_result)
        #f0 = f0 / np.linalg.norm(f0)

        data['MFCC'] = mfcc_result
        data['input'] = mfcc_result

        return data

class ToRaw(object):
    def __init__(self):
        pass
    
    def __call__(self, data):
        signal = data['signal']
        raw_result = raw_pad(signal)
        data['input'] = raw_result
        return data


class WaveLen(object):
    def __init__(
            self,
            directory: str
    ):
        self.path = directory

        wavs = [j for j in os.listdir(self.path) if j.endswith('.wav')]
        audio_len = []

        for j in wavs:
            wav_path = os.path.join(self.path, j)
            check = open(wav_path, 'rb')
            l1 = check.readline()

            if l1[0:4] != b'RIFF':
                continue

            sg, sr = torchaudio.load(wav_path)
            target_len = sg.shape[1] / sr
            audio_len.append((j, target_len))

        self.audio_len = audio_len

    def __call__(self):
        return self.audio_len

    def __len__(self):
        return len(self.audio_len)

    def __getitem__(self, index):
        file_name, wav_len = self.audio_len[index]
        data = {"File_Name": file_name,
                "Length": wav_len}

        self.data = data

        return data


class Files(object):
    def __init__(
            self,
            directory: str
    ):
        self.path = directory

        wavs = [j for j in os.listdir(self.path) if j.endswith('.wav')]
        audio_len = []

        for j in wavs:
            wav_path = os.path.join(self.path, j)
            check = open(wav_path, 'rb')
            l1 = check.readline()

            if l1[0:4] != b'RIFF':
                continue

            audio_len.append(wav_path)

        self.audio_len = audio_len

    def __call__(self):
        return self.audio_len

    def __len__(self):
        return len(self.audio_len)

    def __getitem__(self, index):
        file_name = self.audio_len[index]
        data = {"File_Name": file_name}

        self.data = data

        return data


def mel_pad1(data, length=400):
    if data.shape[2] > length:
        padded = data[:, :, 0:length]
    else:
        padded = torch.cat([data, torch.zeros(data.shape[0], data.shape[1], length - data.shape[2])], dim=2)

    return padded


def mel_pad2(data, length=400):
    if data.shape[0] > length:
        padded = data[0:length, :]
    else:
        padded = torch.cat([data, torch.zeros(length - data.shape[0], data.shape[1])], dim=0)

    return padded


def mfcc_pad1(data, length=400):  # 126000을 데이터 상황에 맞추어 400으로 바꾸어줌
    if data.shape[2] > length:
        padded = data[:, :, 0:length]
    else:
        padded = torch.cat([data, torch.zeros(data.shape[0], data.shape[1], length - data.shape[2])], dim=2)

    return padded


def mfcc_pad2(data, length=400):  #########에러 
    if data.shape[0] > length:
        padded = data[0:length, :]
    else:
        padded = torch.cat([data, torch.zeros(length - data.shape[0], data.shape[1])], dim=0)

    return padded


def raw_pad(data, length=240000):
    if data.shape[0] > 1:
        data = data.view(1,-1)
    
    if data.shape[1] > length:
        padded = data[:, 0:length]
    else:
        padded = torch.cat([data, torch.zeros(data.shape[0],length - data.shape[1])], dim=1)

    return padded


# 전처리 함수 2가지

spec_pre = transforms.Compose([AudioInput(), Spectrogram()])
mfcc1_pre = transforms.Compose([AudioInput(), ToMFCC1()])
mfcc2_pre = transforms.Compose([AudioInput(), ToMFCC2()])
mel1_pre = transforms.Compose([AudioInput(), MelSpec1()])
mel2_pre = transforms.Compose([AudioInput(), MelSpec2()])
raw_pre = transforms.Compose([AudioInput(), ToRaw()])

