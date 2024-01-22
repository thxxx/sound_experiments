import os
import json
import math
import sys
import copy
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm

import librosa
import soundfile as sf
from audiotools import AudioSignal

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

import wandb

from accelerate import Accelerator
from transformers import get_scheduler

from beats.BEATs import BEATsConfig, BEATs

from config import Config
from audiomodel_inpainting import AudioProcessing
from audiocraft.modules.conditioners import JointEmbedCondition, SegmentWithAttributes, WavCondition, ConditioningAttributes

def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def wandb_init(cfg):
    wandb.init(
            # set the wandb project where this run will be logged
            project=cfg.wandb_project_name,
            
            # track hyperparameters and run metadata
            config={
            "learning_rate": cfg.learning_rate,
            "epochs": cfg.num_train_epochs,
            "batch_size": cfg.batch_size,
            }
    )
    
def save_checkpoint(cfg, model, result, best_loss, epoch=0):
    save_checkpoint = False
    with open("{}/summary.jsonl".format(cfg.output_dir), "a") as f:
        f.write(json.dumps(result) + "\n\n")
        
    if result["train_loss"] < best_loss:
      best_loss = result["train_loss"]
      save_checkpoint = True
      
    # 모델 상태 저장
    if save_checkpoint and cfg.checkpointing_steps == "best":
        torch.save(model.state_dict(), os.path.join(cfg.output_dir, "best.pth"))

    torch.save(model.state_dict(), os.path.join(cfg.output_dir, "last.pth"))
    torch.save(model.state_dict(), os.path.join(cfg.output_dir, f"epoch_{epoch}.pth"))

    return best_loss

def build_model(cfg):
        from audiocraft.models.loaders import load_compression_model, load_lm_model
        """Instantiate models and optimizer."""     
        compression_model = load_compression_model('facebook/audiogen-medium', device=cfg.device)
        lm = load_lm_model('facebook/audiogen-medium', custom_cfg=cfg)
        return compression_model, lm

def process_audio_tokenizer(wav, compression_model):
        with torch.no_grad():
            audio_tokens, scale = compression_model.encode(wav)
        return audio_tokens

def post_process_audio_tokenizer(audio_tokens, audio_lengths=None, compression_model=None, lm=None, cfg=None):
    padding_mask = torch.ones_like(audio_tokens, dtype=torch.bool, device=audio_tokens.device)
    audio_tokens = audio_tokens.clone()
    padding_mask = padding_mask.clone()
    token_sample_rate = compression_model.frame_rate
    B, K, T_s = audio_tokens.shape
    
    for i in range(B):
        valid_tokens = math.floor(audio_lengths[i] / cfg.sample_rate * token_sample_rate)
        audio_tokens[i, :, valid_tokens:] = lm.special_token_id
        padding_mask[i, :, valid_tokens:] = 0

    return audio_tokens, padding_mask

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
        
        import random
        audio_path = data['audio_path']
        total_duration = data['duration']
        description = "<RANDOM>"
        
        # Set duration
        duration = self.duration if total_duration >= 3 else total_duration  # Duration is 3 seconds or total_duration if less than 3
        
        # Set offset based on conditions
        if total_duration < self.duration or self.train == False:
            offset = 0.0 
        else:
            max_offset = total_duration - duration  # Calculate the maximum possible offset
            offset = random.uniform(0, max_offset)  # Choose a random offset within the possible range
        
        # Load audio signal file
        wav = AudioSignal(audio_path, offset=offset, duration=duration)
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
        
        self.target_sample_rate = cfg.sample_rate
        self.duration = cfg.duration
        self.device = cfg.device
        self.audio_paths = cfg.eval_data_path

        self.df = pd.read_csv(self.audio_paths)[:20]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        data = self.df.iloc[idx] #self.audio_files_list[idx]
        
        import random
        audio_path = data['audio_path']
        total_duration = data['duration']
        description = "<RANDOM>"
        
        # Set duration
        duration = self.duration if total_duration >= 3 else total_duration  # Duration is 3 seconds or total_duration if less than 3
        
        offset = 0.0   
        # Load audio signal file
        wav = AudioSignal(audio_path, offset=offset, duration=duration)
        length = wav.signal_length

        # Encode audio signal as one long file
        wav.to_mono()
        wav.resample(self.target_sample_rate)

        if wav.duration < self.duration:
          pad_len = int(self.duration * self.target_sample_rate) - wav.signal_length
          wav.zero_pad(0, pad_len)
        elif wav.duration > self.duration:
          wav.truncate_samples(self.duration * self.target_sample_rate)


        return wav.audio_data.squeeze(1), description

# SoundConditioner 클래스 정의
class SoundConditioner(nn.Module):
    def __init__(self, cfg):
        super(SoundConditioner, self).__init__()

        # 비트 모델 로드
        beats_ckpt = "beats/weights.pt"
        self.device = cfg.device
        self.beats_model = self.load_beats(beats_ckpt)
        self.beats_model.eval()

    def forward(self, wav):
        # 오디오 토큰 길이 설정
        audio_token_length = 180
        # 오디오 임베딩 처리
        audio_embeds = self.process_audio_embedding(wav.squeeze(1).to(self.device), self.beats_model, audio_token_length, self.device)
        return audio_embeds, None

    def load_beats(self, beats_ckpt):
        beats_checkpoint = torch.load(beats_ckpt, map_location='cpu')
        beats_cfg = BEATsConfig(beats_checkpoint['cfg'])
        beats = BEATs(beats_cfg)
        beats.load_state_dict(beats_checkpoint['model'])
        for name, param in beats.named_parameters():
            param.requires_grad = False
        return beats

    def process_audio_embedding(self, wav, beats, audio_token_length, device):
        # 오디오 패딩 마스크 생성
        audio_padding_mask = torch.zeros(wav.shape, device=wav.device).bool()
    
        # 오디오 특징 추출
        audio_embeds, _ = beats.extract_features(wav, padding_mask=audio_padding_mask, feature_only=True)
    
        # 현재 길이 확인
        current_length = audio_embeds.size(1)
    
        if current_length > audio_token_length:
            # 오디오 임베딩 자르기
            audio_embeds = audio_embeds.narrow(1, 0, audio_token_length)
        elif current_length < audio_token_length:
            # 필요한 패딩 길이 계산 및 적용
            padding_length = audio_token_length - current_length
            audio_embeds = F.pad(audio_embeds, (0, 0, 0, padding_length))
    
        return audio_embeds

# SoundConditioner 클래스 정의
class SoundConditionerEncodec(nn.Module):
    def __init__(self, compression_model):
        super(SoundConditionerEncodec, self).__init__()

        # 비트 모델 로드
        self.compression_model = compression_model

    def forward(self, wav):
        # 오디오 토큰 길이 설정
        emb = self.compression_model.encoder(wav)

        return emb.permute(0,2,1), None


def main():
    cfg = Config()
    accelerator = Accelerator(gradient_accumulation_steps=cfg.gradient_accumulation_steps)
    device = accelerator.device
    cfg.update(device=accelerator.device)
    make_dir(cfg.output_dir)
    make_dir(cfg.generated_dir)
    
    base_path = "./csv_files/"
    train_data_path = f"{base_path}/train_dataset.csv"
    eval_data_path = f"{base_path}/eval_dataset.csv"
    cfg.update(train_data_path=train_data_path, eval_data_path=eval_data_path)
    
    # 'sound'를 'cross' 키의 리스트에 추가
    cfg.fuser['cross'].append('sound')
    if accelerator.is_main_process: 
        wandb_init(cfg)
    
    with accelerator.main_process_first():  
        compression_model, lm = build_model(cfg)
        model = AudioProcessing(cfg, lm)  
        t5conditioner = copy.deepcopy(lm.condition_provider.conditioners.description)
        soundconditioner = SoundConditionerEncodec(compression_model)
        audio_dataset = AudioDataset(cfg, train=True) 
        eval_dataset = AudioDataset(cfg, train=False)
    test_dataset = TestDataset(cfg)
    
    audio_dataloader = DataLoader(audio_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=8)
    eval_dataloader = DataLoader(eval_dataset, batch_size=cfg.eval_batch_size, shuffle=False, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=1)
        
    optimizer_parameters = [param for param in model.parameters() if param.requires_grad]
        
    optimizer = torch.optim.AdamW(
        optimizer_parameters, lr=cfg.learning_rate,
        betas=(cfg.adam_beta1, cfg.adam_beta2),
        weight_decay=cfg.adam_weight_decay,
        eps=cfg.adam_epsilon,
    )
    
    
    num_update_steps_per_epoch = math.ceil(len(audio_dataloader) / cfg.gradient_accumulation_steps)
    if cfg.max_train_steps is None:
      cfg.max_train_steps = cfg.num_train_epochs * num_update_steps_per_epoch
    
    lr_scheduler = get_scheduler(
          name=cfg.lr_scheduler_type,
          optimizer=optimizer,
          num_warmup_steps=cfg.num_warmup_steps * cfg.gradient_accumulation_steps,
          num_training_steps=cfg.max_train_steps * cfg.gradient_accumulation_steps,
      )

    with accelerator.main_process_first():
        if cfg.resume_from_checkpoint is not None:
            accelerator.print(f"Resumed from local checkpoint: {cfg.resume_from_checkpoint}")
            model.load_state_dict(torch.load(cfg.resume_from_checkpoint, map_location=accelerator.device))
            #accelerator.load_state(cfg.resume_from_checkpoint)

    audio_dataloader, eval_dataloader, model, compression_model, t5conditioner, soundconditioner, optimizer, lr_scheduler = accelerator.prepare(
audio_dataloader, eval_dataloader, model, compression_model, t5conditioner, soundconditioner, optimizer, lr_scheduler
)

    starting_epoch, completed_steps, best_loss, save_epoch = 0, 0, np.inf, 0
    progress_bar = tqdm(range(cfg.max_train_steps), disable=not accelerator.is_local_main_process)

    
    for epoch in range(starting_epoch, cfg.num_train_epochs):
        accelerator.print(f"-------------------EPOCH{epoch}-------------------------" )
        total_loss, total_val_loss = 0, 0
        model.train()
        for batch_idx, (wav, descriptions, lengths) in enumerate(audio_dataloader):
            with accelerator.accumulate(model):
                with torch.no_grad():
                    unwrapped_textconditioner = accelerator.unwrap_model(t5conditioner)
                    unwrapped_soundconditioner = accelerator.unwrap_model(soundconditioner)
                    
                    tokenized = {}
                    tokenized["description"] =  unwrapped_textconditioner.tokenize(descriptions)
                    tokenized["sound"] = wav
                    
                    # conditioning
                    output = {}
                    for attribute, inputs in tokenized.items():
                        if attribute == "description":   
                            condition, mask = unwrapped_textconditioner(inputs)
                        elif attribute == "sound":
                            condition, mask = unwrapped_soundconditioner(inputs)
                        output[attribute] = (condition, mask)

                    unwrapped_vae = accelerator.unwrap_model(compression_model)
                    audio_tokens = process_audio_tokenizer(wav, unwrapped_vae)
                    audio_tokens, padding_mask = post_process_audio_tokenizer(audio_tokens, lengths, unwrapped_vae, lm, cfg) 

                loss = model.module(audio_tokens, padding_mask, attributes=None, condition_tensors=output)
                ppl =  torch.exp(loss)
                total_loss += loss.detach().float()
                accelerator.backward(loss)     
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    completed_steps += 1
        
        model.eval()
        for batch_idx, (wav, descriptions, lengths) in enumerate(eval_dataloader):
            with accelerator.accumulate(model):
                with torch.no_grad():
                    unwrapped_textconditioner = accelerator.unwrap_model(t5conditioner)
                    unwrapped_soundconditioner = accelerator.unwrap_model(soundconditioner)
                    
                    tokenized = {}
                    tokenized["description"] =  unwrapped_textconditioner.tokenize(descriptions)
                    tokenized["sound"] = wav
                    
                    # conditioning
                    output = {}
                    for attribute, inputs in tokenized.items():
                        if attribute == "description":   
                            condition, mask = unwrapped_textconditioner(inputs)
                        elif attribute == "sound":
                            condition, mask = unwrapped_soundconditioner(inputs)
                        output[attribute] = (condition, mask)

                    unwrapped_vae = accelerator.unwrap_model(compression_model)
                    audio_tokens = process_audio_tokenizer(wav, unwrapped_vae)
                    audio_tokens, padding_mask = post_process_audio_tokenizer(audio_tokens, lengths, unwrapped_vae, lm, cfg) 

                    loss = model.module(audio_tokens, padding_mask, attributes=None, condition_tensors=output)
                    total_val_loss += loss  
    
        if accelerator.is_main_process:         
            result = {}
            result["epoch"] = save_epoch + 1,
            result["step"] = completed_steps
            result["train_loss"] = round(total_loss.item()/cfg.save_steps, 4)
            result["valid_loss"] = round(total_val_loss.item()/len(eval_dataloader), 4)
            
            wandb.log(result)
            result_string = "Epoch: {}, Loss Train: {}, Valid: {}\n".format(save_epoch + 1, result["train_loss"], result["valid_loss"])    
            accelerator.print(result_string) 
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_vae = accelerator.unwrap_model(compression_model)
            unwrapped_soundconditioner = accelerator.unwrap_model(soundconditioner)
            best_loss = save_checkpoint(cfg, unwrapped_model, result, best_loss, save_epoch)
            for test_step, (wav, descriptions) in enumerate(test_dataloader):
                audio_conditions = unwrapped_soundconditioner(wav.to(device))
                gen_token, gen_audio = unwrapped_model.inference(descriptions, audio_conditions, unwrapped_vae)
                audio_filename = f"epoch_{save_epoch}_{test_step}.wav"
                unwrapped_model.save_audio(gen_audio, audio_filename, cfg)
            save_epoch += 1 

if __name__== "__main__":
    parser = argparse.ArgumentParser(description='Training script for different categories.')
    args = parser.parse_args()
    main()