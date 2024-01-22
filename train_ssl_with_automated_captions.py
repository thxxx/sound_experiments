import os
import json
import math
import sys
import random
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm

import soundfile as sf
import librosa

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler

import wandb

from accelerate import Accelerator
from transformers import get_scheduler

from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
from beats.BEATs import BEATsConfig, BEATs

from config import Config
from captioning_config import CaptionConfig
from audiomodel import AudioProcessing
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

# beats
def load_beats(beats_ckpt, device):
    beats_checkpoint = torch.load(beats_ckpt, map_location='cpu')
    beats_cfg = BEATsConfig(beats_checkpoint['cfg'])
    beats = BEATs(beats_cfg)
    beats.load_state_dict(beats_checkpoint['model'])
    for name, param in beats.named_parameters():
        param.requires_grad = False
    return beats

class CaptionModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.model = GPT2LMHeadModel.from_pretrained(cfg.lm_model_name)
        self.ln_audio = nn.LayerNorm(cfg.audio_embedding_size)
        self.lm_proj_audio = nn.Linear(cfg.audio_embedding_size, cfg.text_embedding_size)

    def forward(self, audio_embeds, input_ids, labels, tokenizer, device):
        # 오디오 임베딩 레이어 정규화 및 크기 조정
        audio_embeds = self.ln_audio(audio_embeds)
        audio_embeds = self.lm_proj_audio(audio_embeds)

        # 텍스트 임베딩
        embed_tokens = self.model.transformer.wte
        inputs_embeds = embed_tokens(input_ids)

        # BOS 토큰 임베딩 생성 및 반복
        bsz = input_ids.size(0)
        bos_embeds = embed_tokens(torch.ones([1], dtype=torch.long, device=device) * tokenizer.bos_token_id)
        bos_embeds = bos_embeds.repeat(bsz, 1, 1)

        # 오디오, BOS 및 텍스트 임베딩 결합
        inputs_embeds = torch.cat([bos_embeds, audio_embeds, inputs_embeds], dim=1)

        # 모델 실행 및 손실 계산
        output = self.model(inputs_embeds=inputs_embeds, labels=labels)
        loss = output.loss

        return loss

    def generate(self, audio_embeds, input_ids, tokenizer, max_length=50, num_return_sequences=1):

        # 텍스트 생성
        audio_embeds = self.ln_audio(audio_embeds)
        audio_embeds = self.lm_proj_audio(audio_embeds)

        # 텍스트 임베딩
        embed_tokens = self.model.transformer.wte
        inputs_embeds = embed_tokens(input_ids)

        # BOS 토큰 임베딩 생성 및 반복
        bsz = input_ids.size(0)
        bos_embeds = embed_tokens(torch.ones([1], dtype=torch.long, device=inputs_embeds.device) * tokenizer.bos_token_id)
        bos_embeds = bos_embeds.repeat(bsz, 1, 1)

        # 오디오, BOS 및 텍스트 임베딩 결합
        inputs_embeds = torch.cat([bos_embeds, audio_embeds, inputs_embeds], dim=1)
        atts = torch.ones(inputs_embeds.size()[:-1], dtype=torch.long).to(inputs_embeds.device)
        
        output = self.model.generate(inputs_embeds=inputs_embeds, attention_mask=atts, max_length=20, num_return_sequences=1,pad_token_id=tokenizer.eos_token_id)
        
        # 생성된 텍스트 디코딩
        generated_texts = []
        for i in range(len(output)):
            generated_text = tokenizer.decode(output[i], skip_special_tokens=True)
            generated_texts.append(generated_text)
        
        return generated_texts, output

def process_audio_embedding(wav, beats, audio_token_length, device):
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

class CaptionDataset(Dataset):
    def __init__(self, cfg, tokenizer: GPT2Tokenizer, train=True):
        if train:
            self.data_path = cfg.train_data_path
        else:
            self.data_path = cfg.eval_data_path
        self.dataframe = pd.read_csv(self.data_path)
        self.tokenizer = tokenizer
        self.tokenizer.pad_token = tokenizer.eos_token
        self.audio_token_length = cfg.audio_token_length
        self.text_max_length = cfg.text_max_length
        self.max_length = self.audio_token_length + self.text_max_length + 1   # 전체 길이 설정
        self.sample_rate = cfg.sample_rate
        self.duration = cfg.duration
        
    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # DataFrame에서 데이터 로드
        data = self.dataframe.iloc[idx]
        wav_path = data['audio_path']
        
        caption = "<TEXT>"

        # 오디오 파일 정보 읽기
        info = sf.info(wav_path)
        lengths = info.duration * info.samplerate

        # 오디오 파일이 3초 이상인 경우
        if info.duration > 3:
            # 무작위 시작 지점 선택
            start = random.randint(0, int(info.duration * info.samplerate) - 3 * info.samplerate)
            # 해당 위치에서부터 3초 동안의 오디오 읽기
            wav, sr = sf.read(wav_path, start=start, frames=3 * info.samplerate)
        else:
            # 전체 파일 읽기
            wav, sr = sf.read(wav_path)
            
        if len(wav.shape) == 2:
            wav = wav[:, 0]

        # 샘플링 레이트 조정
        if sr != self.sample_rate:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=self.sample_rate, res_type="fft")

        #from IPython.display import Audio
        #display(Audio(wav, rate=16000))
        
        # 오디오 길이 조정 (3초로)
        target_length = self.duration * self.sample_rate  # 3초에 해당하는 샘플 수
        if len(wav) > target_length:
            wav = wav[:target_length]  # 3초를 초과하는 경우 자르기
        elif len(wav) < target_length:
            padding = target_length - len(wav)  # 필요한 패딩 계산
            wav = np.pad(wav, (0, padding), 'constant')  # 패딩 적용

        # 토큰화
        batch_encoding = self.tokenizer(caption, return_tensors='pt')
        input_ids = batch_encoding['input_ids'].squeeze(0)

        return wav, input_ids, lengths

def build_model(cfg):
        from audiocraft.models.loaders import load_compression_model, load_lm_model
        """Instantiate models and optimizer."""     
        compression_model = load_compression_model('facebook/audiogen-medium', device=cfg.device)
        lm = load_lm_model('facebook/audiogen-medium', device=cfg.device)
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

class TestDataset(Dataset):
    def __init__(self, cfg):

        if cfg.prompts is None:
            test_df = pd.read_csv(cfg.test_data_path)
            self.prompts = [test_df.iloc[0]['caption'], test_df.iloc[1]['caption'], test_df.iloc[2]['caption'], test_df.iloc[3]['caption'], test_df.iloc[4]['caption'], test_df.iloc[5]['caption'], test_df.iloc[6]['caption'], test_df.iloc[7]['caption'] ]
        else:
            self.prompts = cfg.prompts

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):

        return self.prompts[idx]

def main():
    cfg = Config()
    caption_cfg = CaptionConfig()

    accelerator = Accelerator(gradient_accumulation_steps=cfg.gradient_accumulation_steps)
    device = accelerator.device
    cfg.update(device=accelerator.device)
    make_dir(cfg.output_dir)
    make_dir(cfg.generated_dir)
    if accelerator.is_main_process: 
        wandb_init(cfg)

    with accelerator.main_process_first():  
        compression_model, lm = build_model(cfg)
        model = AudioProcessing(cfg, lm)
    
        beats_model = load_beats(caption_cfg.beats_ckpt, device).to(device)
        tokenizer = GPT2Tokenizer.from_pretrained(caption_cfg.lm_model_name)
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
        caption_model = CaptionModel(caption_cfg).to(device)
        caption_model.load_state_dict(torch.load("caption_weight/6.pth"))
    
        audio_dataset = CaptionDataset(caption_cfg, tokenizer, train=True) 
        eval_dataset = CaptionDataset(caption_cfg, tokenizer, train=False)
        test_dataset = TestDataset(caption_cfg)

    # RandomSampler 설정
    train_sampler = RandomSampler(audio_dataset, num_samples=cfg.train_sample_num, replacement=True)
    
    audio_dataloader = DataLoader(audio_dataset, batch_size=caption_cfg.batch_size, shuffle=False, sampler=train_sampler, num_workers=12)
    eval_dataloader = DataLoader(eval_dataset, batch_size=caption_cfg.eval_batch_size, shuffle=False, num_workers=8)
    test_dataloader = DataLoader(test_dataset, batch_size=1)
    
    optimizer_parameters = [param for param in model.lm.parameters() if param.requires_grad]

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
      if cfg.resume_from_checkpoint:
            if cfg.resume_from_checkpoint is not None or cfg.resume_from_checkpoint != "":
                accelerator.print(f"Resumed from local checkpoint: {cfg.resume_from_checkpoint}")
                model.load_state_dict(torch.load(cfg.resume_from_checkpoint))
                #accelerator.load_state(cfg.resume_from_checkpoint)
                # path = os.path.basename(args.resume_from_checkpoint)
                accelerator.print(f"Resumed from local checkpoint: {cfg.resume_from_checkpoint}")



    audio_dataloader, eval_dataloader, model, compression_model, caption_model, beats_model, optimizer, lr_scheduler = accelerator.prepare(
    audio_dataloader, eval_dataloader, model, compression_model, caption_model, beats_model, optimizer, lr_scheduler
)
    compression_model.eval()
    beats_model.eval()
    caption_model.eval()
    torch.cuda.empty_cache()

    starting_epoch, completed_steps, best_loss, save_epoch = 0, 0, np.inf, 96
    progress_bar = tqdm(range(cfg.max_train_steps), disable=not accelerator.is_local_main_process)
    
    for epoch in range(starting_epoch, cfg.num_train_epochs):
        accelerator.print(f"-------------------EPOCH{epoch}-------------------------" )
        total_loss, total_val_loss = 0, 0
        model.train()
        for batch_idx, (wav, input_ids, lengths) in enumerate(audio_dataloader):
             with accelerator.accumulate(model):
                with torch.no_grad():
                    unwrapped_vae = accelerator.unwrap_model(compression_model)
                    audio_tokens = process_audio_tokenizer(wav.unsqueeze(1).to(torch.float32), unwrapped_vae)
                    audio_tokens, padding_mask = post_process_audio_tokenizer(audio_tokens, lengths, unwrapped_vae, lm, cfg) 
                
                    unwrapped_beats = accelerator.unwrap_model(beats_model)
                    unwrapped_gpt = accelerator.unwrap_model(caption_model)
                    audio_embeds = process_audio_embedding(wav, unwrapped_beats, caption_cfg.audio_token_length, device)
    
                    descriptions, output = unwrapped_gpt.generate(audio_embeds, input_ids, tokenizer)
                    torch.cuda.empty_cache()
                    #for d in descriptions:
                    #    print(d)
                    attributes = [
                        ConditioningAttributes(text={'description': str(description)})
                        for description in descriptions]
              
                loss = model(audio_tokens, padding_mask, attributes)
                #print(loss)
                ppl =  torch.exp(loss)
                total_loss += loss.detach().float()
                accelerator.backward(loss)     
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                torch.cuda.empty_cache()

                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    completed_steps += 1
        model.eval()
        for batch_idx, (wav, input_ids, lengths) in enumerate(eval_dataloader):
            with accelerator.accumulate(model):
                with torch.no_grad():
                  
                    unwrapped_vae = accelerator.unwrap_model(compression_model)
                    audio_tokens = process_audio_tokenizer(wav.unsqueeze(1).to(torch.float32), unwrapped_vae)
                    audio_tokens, padding_mask = post_process_audio_tokenizer(audio_tokens, lengths, unwrapped_vae, lm, cfg) 
                
                    unwrapped_beats = accelerator.unwrap_model(beats_model)
                    unwrapped_gpt = accelerator.unwrap_model(caption_model)
                    audio_embeds = process_audio_embedding(wav.to(device), unwrapped_beats, caption_cfg.audio_token_length, device)
                    descriptions, output = unwrapped_gpt.generate(audio_embeds, input_ids, tokenizer)
                    #for d in descriptions:
                    #    print(d)
                    attributes = [
                        ConditioningAttributes(text={'description': str(description)})
                        for description in descriptions]
                    loss = model(audio_tokens, padding_mask, attributes)
                    total_val_loss += loss 
                    torch.cuda.empty_cache()
                    
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
            best_loss = save_checkpoint(cfg, unwrapped_model, result, best_loss, save_epoch)
            for test_step, batch in enumerate(test_dataloader):
                gen_token, gen_audio = unwrapped_model.inference(batch, unwrapped_vae)
                audio_filename = f"epoch_{save_epoch}_{test_step}.wav"
                unwrapped_model.save_audio(gen_audio, audio_filename, cfg)
            save_epoch += 1 

if __name__== "__main__":
    parser = argparse.ArgumentParser(description='Training script for different categories.')
    args = parser.parse_args()
    main()
    