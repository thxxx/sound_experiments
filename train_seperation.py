import os
import json
import math
import sys
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

import wandb

from accelerate import Accelerator
from transformers import get_scheduler

from audiocraft.modules.conditioners import JointEmbedCondition, SegmentWithAttributes, WavCondition, ConditioningAttributes
from config import Config
from audiomodel import AudioProcessing
from audiodataset import SeperationDataset, TestDataset

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
        
    if result["valid_loss"] < best_loss:
      best_loss = result["valid_loss"]
      save_checkpoint = True
      
    # 모델 상태 저장
    if save_checkpoint and cfg.checkpointing_steps == "best":
        torch.save(model.state_dict(), os.path.join(cfg.output_dir, f"best.pth"))

    #torch.save(model.state_dict(), os.path.join(cfg.output_dir, "last.pth"))
    torch.save(model.state_dict(), os.path.join(cfg.output_dir, f"{epoch}.pth"))

    return best_loss

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


def main():
    base_path = "./csv_files/"
    train_data_path = f"{base_path}/train_dataset_combined_captions.csv"
    eval_data_path = f"{base_path}/eval_dataset_combined_captions.csv"
    
    cfg = Config()
    
    cfg.update(train_data_path=train_data_path, eval_data_path=eval_data_path)
    
    accelerator = Accelerator(gradient_accumulation_steps=cfg.gradient_accumulation_steps)
    cfg.update(device=accelerator.device)
    make_dir(cfg.output_dir)
    make_dir(cfg.generated_dir)
    if accelerator.is_main_process: 
        wandb_init(cfg)

    with accelerator.main_process_first():  
        compression_model, lm = build_model(cfg)
        audio_dataset = SeperationDataset(cfg, train=True)
        eval_dataset = SeperationDataset(cfg, train=False)
    compression_model.eval()
    
    model = AudioProcessing(cfg, lm)
    test_dataset = TestDataset(cfg)
    
    audio_dataloader = DataLoader(audio_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=8)
    eval_dataloader = DataLoader(eval_dataset, batch_size=cfg.eval_batch_size, shuffle=False, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=1)

    optimizer_parameters = [param for param in model.lm.parameters() if param.requires_grad]
    
    optimizer = torch.optim.AdamW(
        optimizer_parameters, 
        lr=cfg.learning_rate,
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


    audio_dataloader, eval_dataloader, model, compression_model, optimizer, lr_scheduler = accelerator.prepare(
        audio_dataloader, eval_dataloader, model, compression_model, optimizer, lr_scheduler
    )

    starting_epoch, completed_steps, best_loss, save_epoch = 0, 0, np.inf, 0
    progress_bar = tqdm(range(cfg.max_train_steps), disable=not accelerator.is_local_main_process)
    
    for epoch in range(starting_epoch, cfg.num_train_epochs):
        accelerator.print(f"-------------------EPOCH{epoch}-------------------------" )
        total_loss, total_val_loss = 0, 0
        model.train()
        
        for batch_idx, (synthesized_wav, prompts, ground_truth, lengths) in enumerate(audio_dataloader):
            # Consider batch
            with accelerator.accumulate(model):
                with torch.no_grad():
                    unwrapped_vae = accelerator.unwrap_model(compression_model)
                    synthesized_audio_tokens = process_audio_tokenizer(synthesized_wav, unwrapped_vae)
                    audio_tokens = process_audio_tokenizer(wav, unwrapped_vae)
                    audio_tokens, padding_mask = post_process_audio_tokenizer(audio_tokens, lengths, unwrapped_vae, lm, cfg)
                    attributes = [
                        ConditioningAttributes(text={'description': str(description)})
                        for description in prompts
                    ]
                
                loss = model(audio_tokens, padding_mask, attributes)
                ppl =  torch.exp(loss)
                total_loss += loss.detach().float()
                accelerator.backward(loss)     
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    completed_steps += 1
                    
            #if batch_idx == cfg.save_steps:
            #    break
    

            #if completed_steps & cfg.save_steps == 0:
        
        # model.eval()
        # for batch_idx, (wav, descriptions, lengths) in enumerate(eval_dataloader):
        #     with accelerator.accumulate(model):
        #         with torch.no_grad():
        #             unwrapped_vae = accelerator.unwrap_model(compression_model)
        #             audio_tokens = process_audio_tokenizer(wav, unwrapped_vae)
        #             audio_tokens, padding_mask = post_process_audio_tokenizer(audio_tokens, lengths, unwrapped_vae, lm, cfg) 
        #             attributes = [
        #                 ConditioningAttributes(text={'description': str(description)})
        #                 for description in descriptions]
        #             loss = model(audio_tokens, padding_mask, attributes)
        #             total_val_loss += loss  
    
        # if accelerator.is_main_process:         
        #     result = {}
        #     result["epoch"] = save_epoch + 1,
        #     result["step"] = completed_steps
        #     result["train_loss"] = round(total_loss.item()/cfg.save_steps, 4)
        #     result["valid_loss"] = round(total_val_loss.item()/len(eval_dataloader), 4)
            
        #     wandb.log(result)
        #     result_string = "Epoch: {}, Loss Train: {}, Valid: {}\n".format(save_epoch + 1, result["train_loss"], result["valid_loss"])    
        #     accelerator.print(result_string) 
        #     unwrapped_model = accelerator.unwrap_model(model)
        #     unwrapped_vae = accelerator.unwrap_model(compression_model)
        #     best_loss = save_checkpoint(cfg, unwrapped_model, result, best_loss, save_epoch)
        #     for test_step, batch in enumerate(test_dataloader):
        #         gen_token, gen_audio = unwrapped_model.inference(batch, unwrapped_vae)
        #         audio_filename = f"epoch_{save_epoch}_{test_step}.wav"
        #         unwrapped_model.save_audio_and_tokens(gen_audio, gen_token, audio_filename, cfg)
        #     save_epoch += 1 

if __name__== "__main__":
    parser = argparse.ArgumentParser(description='Training script for different categories.')
    args = parser.parse_args()
    main()
    