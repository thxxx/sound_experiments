import pandas as pd
from audiotools import AudioSignal
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
import torch
import re

class SeperationDataset(Dataset):
    def __init__(self, cfg, train=True):
        self.train = train
        
        self.target_sample_rate = cfg.sample_rate
        self.duration = cfg.duration
        self.device = cfg.device

        if self.train:
            self.audio_paths = cfg.train_data_path
        else:
            self.audio_paths = cfg.eval_data_path

        self.df = pd.read_csv(self.audio_paths)

    def __len__(self):
        lens=0
        if self.train:
            lens = len(self.df[:41200])
        else:
            lens = len(self.df[:4560])
            
        return lens

    def process_data(self, wav):
        # Encode audio signal as one long file
        wav.to_mono()
        wav.resample(self.target_sample_rate)

        if wav.duration < self.duration:
            pad_len = int(self.duration * self.target_sample_rate) - wav.signal_length

            random_integer = random.randint(0, pad_len-1)
            zero_wavs = np.zeros((1,1,48000))
            zero_wavs[:,:,:wav.signal_length] = wav.numpy().squeeze()
            wav = AudioSignal(zero_wavs, sample_rate=16000)
            # wav.zero_pad(0, pad_len)
        elif wav.duration > self.duration:
            wav.truncate_samples(self.duration * self.target_sample_rate)
        
        return wav

    def __getitem__(self, idx):
        data = self.df.iloc[idx] #self.audio_files_list[idx]

        audio_path, description, synthesized_index = data['audio_path'], data['caption'], data['synthesized_index']

        # Load audio signal file
        wav = AudioSignal(audio_path)

        # 두번째 데이터
        # Load second audio signal file for combining
        data = self.df.iloc[synthesized_index-1] # self.audio_files_list[idx]
        audio_path, description2 = data['audio_path'], data['caption']

        wav2 = AudioSignal(audio_path)
        length = wav.signal_length
        length2 = wav2.signal_length
        wav = self.process_data(wav)
        wav2 = self.process_data(wav2)

        # combine
        synthesized_audio = AudioSignal(wav.numpy() + wav2.numpy(), sample_rate=16000)

        description2 = re.sub("The sound of ", "", description2)
        prompt = f"Remove The sound of '{description2}'"
        ground_truth = wav

        # # or 
        # prompt = "Remain " + description
        # ground_truth = wav

        return torch.tensor(synthesized_audio.numpy()).squeeze(dim=0), prompt, torch.tensor(ground_truth.numpy()).squeeze(dim=0), length
        # return wav.audio_data.squeeze(1), description, length


class AudioDataset(Dataset):
    def __init__(self, cfg, train=True):
        self.train = train
        
        self.target_sample_rate = cfg.sample_rate
        self.duration = cfg.duration
        self.device = cfg.device

        if self.train:
            self.audio_paths = cfg.train_data_path
        else:
            self.audio_paths = cfg.eval_data_path

        self.df = pd.read_csv(self.audio_paths)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        
        data = self.df.iloc[idx] #self.audio_files_list[idx]
        audio_path = data['audio_path']
        description = data['caption']

        # Load audio signal file
        
        wav = AudioSignal(audio_path)
        length = wav.signal_length

        # Encode audio signal as one long file
        wav.to_mono()
        wav.resample(self.target_sample_rate)

        if wav.duration < self.duration:
          pad_len = int(self.duration * self.target_sample_rate) - wav.signal_length
          wav.zero_pad(0, pad_len)
        elif wav.duration > self.duration:
          wav.truncate_samples(self.duration * self.target_sample_rate)


        return wav.audio_data.squeeze(1), description, length

class TestDataset(Dataset):
    def __init__(self, cfg):

        if cfg.prompts is None:
            test_df = pd.read_csv(cfg.eval_data_path)
            self.prompts = [test_df.iloc[0]['caption'], test_df.iloc[1]['caption'], test_df.iloc[2]['caption'], test_df.iloc[3]['caption'], test_df.iloc[4]['caption'], test_df.iloc[5]['caption'], test_df.iloc[6]['caption'], test_df.iloc[7]['caption'] ]
        else:
            self.prompts = cfg.prompts

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.prompts[idx]