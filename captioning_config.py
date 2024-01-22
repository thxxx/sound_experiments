import omegaconf

class CaptionConfig:
    def __init__(self, config_paths=None):
        self.update_default_config()
        self.update_dataset_config()
        self.update_hyperparameter_config()
        self.update_train_config()

    def update(self, **kwargs):
        for key, value in kwargs.items():
            # 기존 속성에 값 할당하거나 새 속성 생성
            if not hasattr(self, key):
                print(key)
            setattr(self, key, value)

    def update_default_config(self):
        self.beats_ckpt = "beats/weights.pt"
        self.lm_model_name = 'gpt2-xl'
        self.audio_embedding_size = 768
        self.text_embedding_size = 1600
        self.audio_token_length = 180
        self.text_max_length = 50
        

    def update_dataset_config(self):
        self.sample_rate = 16000
        self.duration = 3
        self.train_data_path =  "./csv_files/train_dataset.csv" #"./csv_files/train_epidemic_dataset.csv"
        self.eval_data_path = "./csv_files/eval_dataset.csv" #"./csv_files/eval_epidemic_dataset.csv"
        self.test_data_path = "./csv_files/eval_dataset_combined_captions.csv"
        self.prompts = None

    def update_hyperparameter_config(self):
        self.batch_size = 16
        self.eval_batch_size = 16
        self.learning_rate = 3e-6
        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.999
        self.adam_weight_decay = 1e-2
        self.adam_epsilon = 1e-08
        self.lr_scheduler_type = "linear"
        self.gradient_accumulation_steps = 1

    def update_train_config(self):
        self.num_train_epochs = 1000
        self.num_warmup_steps = 0
        self.max_train_steps = None
        self.device = 'cuda' 
        self.output_dir = "./output_dir_xl"
        self.generated_dir = './generated_audios_xl'
        self.save_path = './generated_audios_xl'
        self.checkpointing_steps = "best"
        self.save_steps = 20
        self.resume_from_checkpoint = None
        self.resume_epoch = 0 
        self.wandb_project_name = "captioning-gpt-xl-init-test2"
        self.wandb_id = None 
    