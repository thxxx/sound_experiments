import pandas as pd
from audiotools import AudioSignal
from torch.utils.data import Dataset, DataLoader

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
        duration = data['duration']
        description = description + self.duration_captioning(duration, 1)

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
        
    def duration_captioning(self, duration, decimal=1):
        return " ," + str(round(duration, decimal)) + "seconds" 

class TestDataset(Dataset):
    def __init__(self, cfg):
        self.df = pd.read_csv(cfg.eval_data_path)[:8]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        data = self.df.iloc[idx]
        prompts, duration = data['caption'], data['duration']
        return prompts + self.duration_captioning(duration, 1)

    def duration_captioning(self, duration, decimal=1):
        return " ," + str(round(duration, decimal)) + "seconds" 