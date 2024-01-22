import omegaconf

class Config:
    def __init__(self, config_paths=None):
        
        self.config_paths = config_paths
        self.update_dataset_config()
        self.update_conditioner_config()
        self.update_hyperparameter_config()
        self.update_train_config()
        self.update_audiocraft_config()
        if self.config_paths is None:
            self.set_config_paths()
        self.update_from_config_paths(self.config_paths)
        
    def update_from_config_paths(self, config_paths):
        # 모든 설정 파일을 로드하고 병합
        merged_cfg = omegaconf.OmegaConf.merge(*[omegaconf.OmegaConf.load(path) for path in config_paths])
        merged_cfg_dict = omegaconf.OmegaConf.to_container(merged_cfg, resolve=True)
        
        # 설정 값 할당
        for key, value in merged_cfg_dict.items():
            setattr(self, key, value)
        

    def update(self, **kwargs):
        for key, value in kwargs.items():
            # 기존 속성에 값 할당하거나 새 속성 생성
            if not hasattr(self, key):
                print(key)
            setattr(self, key, value)

    def update_dataset_config(self):
        self.sample_rate = 16000
        self.duration = 3
        self.train_data_path = "./train_dataset.csv"
        self.eval_data_path = "./eval_dataset.csv"
        self.prompts = None

    def update_conditioner_config(self):
        self.audio_embeds_dim = 128
        self.text_embeds_dim = 1536
        self.condition_dim = 1536

    def update_hyperparameter_config(self):
        self.batch_size = 2
        self.eval_batch_size = 2
        self.learning_rate = 3e-6
        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.999
        self.adam_weight_decay = 1e-2
        self.adam_epsilon = 1e-08
        self.lr_scheduler_type = "linear"
        self.snr_gamma = 5.0
        self.gradient_accumulation_steps = 1

    def update_train_config(self):
        self.num_train_epochs = 1000
        self.num_warmup_steps = 0
        self.max_train_steps = None
        self.device = 'cuda' 
        self.output_dir = "./output_dir_finetune"
        self.generated_dir = './generated_audios_finetune'
        self.save_path = './generated_audios_finetune'
        self.checkpointing_steps = "best"
        self.save_steps = 20
        self.resume_from_checkpoint = None
        self.resume_epoch = 0 
        self.wandb_project_name = "audiogen-finetune-init-test1"
        self.wandb_id = None 
            
    def update_audiocraft_config(self):
        self.solver = None
        self.fsdp = None
        self.profiler = None
        self.deadlock = None
        self.dataset = None
        self.checkpoint = None
        self.generate = None
        self.evaluate = None
        self.optim = None
        self.schedule = None
        self.default = None
        self.defaults = None
        self.autocast = None
        self.autocast_dtype = None

        self.compression_model_checkpoint = None
        self.channels = None
        self.logging = None
        self.lm_model = None
        self.codebooks_pattern = None
        self.transformer_lm = None
        self.classifier_free_guidance = None
        self.attribute_dropout = None
        self.fuser = None
        self.conditioners = None
        self.datasource = None

    def set_config_paths(self):
        config_paths = [
            "audiocraft/config/solver/default.yaml",
            "audiocraft/config/solver/audiogen/audiogen_base_16khz.yaml",
            "audiocraft/config/model/lm/audiogen_lm.yaml",
            "audiocraft/config/conditioner/text2sound.yaml",
            "audiocraft/config/model/lm/model_scale/small.yaml",
            "audiocraft/config/dset/audio/audiocaps_16khz.yaml"
        ]
        self.config_paths = config_paths

    