import os
import sys
import math
import typing as tp
import torch
import torch.nn as nn
from torch.nn import functional as F
import torchaudio
from audiotools import AudioSignal
from audiocraft.modules.conditioners import JointEmbedCondition, SegmentWithAttributes, WavCondition, ConditioningAttributes


class AudioProcessing(nn.Module):
    
    def __init__(self, cfg, lm):
        super().__init__()  # 부모 클래스 초기화 호출
        self.cfg = cfg
        self.lm  = lm
        self.audio_embeds_linear = nn.Linear(cfg.audio_embeds_dim, cfg.condition_dim)
        self.to_float32()
        self.freeze_layers()

    def forward(self, audio_tokens, padding_mask, attributes=None, condition_tensors=None):

        for attribute, inputs in condition_tensors.items():
            if attribute == "sound":
                audio_embeds, mask = condition_tensors[attribute]
                audio_embeds = self.audio_embeds_linear(audio_embeds)
                condition_tensors[attribute] = (audio_embeds, mask)
        
        model_output = self.lm.compute_predictions(audio_tokens, conditions=None, condition_tensors=condition_tensors)  # type: ignore 
        logits = model_output.logits
    
        mask = padding_mask & model_output.mask
        ce, ce_per_codebook = self.compute_cross_entropy(logits, audio_tokens, mask)
        
        return ce
    
    def compute_cross_entropy(self, logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> tp.Tuple[torch.Tensor, tp.List[torch.Tensor]]:

        B, K, T = targets.shape
        assert logits.shape[:-1] == targets.shape
        assert mask.shape == targets.shape
        ce = torch.zeros([], device=targets.device)
        ce_per_codebook: tp.List[torch.Tensor] = []
        for k in range(K):
            logits_k = logits[:, k, ...].contiguous().view(-1, logits.size(-1))  # [B x T, card]
            targets_k = targets[:, k, ...].contiguous().view(-1)  # [B x T]
            mask_k = mask[:, k, ...].contiguous().view(-1)  # [B x T]
            ce_targets = targets_k[mask_k]
            ce_logits = logits_k[mask_k]
            q_ce = F.cross_entropy(ce_logits, ce_targets)
            ce += q_ce
            ce_per_codebook.append(q_ce.detach())
        # average cross entropy across codebooks
        ce = ce / K
        return ce, ce_per_codebook

    def audio_generate(self, condition_tensors, audio_conditions, gen_duration=5, compression_model=None):
        with torch.no_grad():
            total_gen_len = math.ceil(gen_duration * compression_model.frame_rate)
            gen_tokens = self.lm.generate_audio_inpainting(
                None, condition_tensors, audio_conditions=audio_conditions, max_gen_len=total_gen_len,
                num_samples=1)
            gen_audio = compression_model.decode(gen_tokens, None)

        return gen_tokens, gen_audio

    def inference(self, descriptions, audio_conditions, compression_model):
        with torch.no_grad():
            attributes = [
            ConditioningAttributes(text={'description': description})
            for description in descriptions]
            audio_embeds, mask = audio_conditions
            audio_embeds = self.audio_embeds_linear(audio_embeds)
            audio_conditions = (audio_embeds, mask)
            gen_tokens, gen_audio = self.audio_generate(attributes, audio_conditions=audio_conditions, gen_duration=self.cfg.duration, compression_model=compression_model)
            
        return gen_tokens, gen_audio

    def save_audio(self, gen_audio, audio_filename, cfg):
        # audio_path 정의
        prompt = audio_filename.split("_")[0]
        save_path = cfg.save_path + f'/{prompt}/'

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        audio_path = os.path.join(save_path, audio_filename)

        # 오디오 파일 저장
        #accelerator.wait_for_everyone()
        wav = gen_audio[0].cpu().detach().numpy()
        generated_audio = AudioSignal(wav, cfg.sample_rate)
        generated_audio.cpu().detach().write(audio_path)
        
    def save_audio_and_tokens(self, gen_audio, gen_tokens, audio_filename, cfg):
        # audio_path 정의
        prompt = audio_filename.split("_")[0]
        save_path = cfg.save_path + f'/{prompt}/'
    
        # 디렉토리가 존재하지 않으면 생성
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        audio_path = os.path.join(save_path, audio_filename)
        text_path = os.path.join(save_path, audio_filename.rsplit('.', 1)[0] + ".txt")  # 오디오 파일명에서 확장자를 제거하고 .txt 추가
    
        # 오디오 파일 저장
        wav = gen_audio[0].cpu().detach().numpy()
        generated_audio = AudioSignal(wav, cfg.sample_rate)
        generated_audio.cpu().detach().write(audio_path)
    
        # gen_tokens를 텍스트로 변환하고 파일로 저장
        with open(text_path, 'w') as text_file:
            # gen_tokens는 숫자 리스트이므로 이를 문자열로 변환
            token_strings = [str(token) for token in gen_tokens.cpu().numpy()]
            text_file.write(' '.join(token_strings))


    def to_float32(self):
        # 모든 가중치를 FP32로 변환
        for param in self.lm.parameters():
            param.data = param.data.to(dtype=torch.float32)

    def freeze_layers(self, train_layers=48):
        for param in self.lm.parameters():
            param.requires_grad = True
            
        '''if train_layers > 0 :
            num_layers = len(self.lm.transformer.layers)
            
            for i in range(num_layers - train_layers, num_layers):
                for param in self.lm.transformer.layers[i].parameters():
                    param.requires_grad = True
                    
            for name, param in self.lm.named_parameters():
                if 'out_norm' in name or 'linears' in name:
                    param.requires_grad = True'''
                
            
        
        
            