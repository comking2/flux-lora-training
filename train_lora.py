import os
import argparse
import torch

# Load .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ .env file loaded")
except ImportError:
    print("⚠️  python-dotenv not installed, loading .env manually...")
    # Manual .env loading
    env_file = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(env_file):
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key] = value
        print("✅ .env file loaded manually")
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from diffusers import FluxPipeline, StableDiffusionPipeline, DDPMScheduler, AutoPipelineForText2Image
from peft import LoraConfig, get_peft_model, TaskType
from transformers import CLIPTextModel, CLIPTokenizer
from dataset_processor import create_dataloader, validate_dataset
import json
import time
from tqdm import tqdm
import wandb

class FluxLoRATrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Using device: {self.device}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name()}")
    
    def setup_model(self):
        print("Loading FLUX pipeline...")
        
        # Load .env file if exists
        env_file = os.path.join(os.path.dirname(__file__), '.env')
        if os.path.exists(env_file):
            with open(env_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        os.environ[key] = value
        
        # Get Hugging Face token from environment
        hf_token = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")
        if hf_token and hf_token != "your_token_here":
            print("🔑 Using Hugging Face token from environment")
        else:
            print("⚠️  No valid Hugging Face token found!")
            print("💡 Please:")
            print("   1. Get token from: https://huggingface.co/settings/tokens")
            print("   2. Update .env file: HUGGINGFACE_TOKEN=hf_your_actual_token")
            print("   3. Request access: https://huggingface.co/black-forest-labs/FLUX.1-schnell")
        
        # Check if model exists locally first
        try:
            print(f"Checking for local model: {self.config.model_name}")
            self.pipe = FluxPipeline.from_pretrained(
                self.config.model_name,
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                local_files_only=True,  # Try local first
                token=hf_token
            )
            print("✅ Found local model, using cached version")
        except Exception as e:
            print(f"❌ Local model not found: {e}")
            print("🔄 Downloading model from Hugging Face...")
            print("⚠️  This may take a while for first download (~17GB)")
            try:
                # RTX 2070 8GB 최적화 설정
                try:
                    self.pipe = FluxPipeline.from_pretrained(
                        self.config.model_name,
                        torch_dtype=torch.float8_e4m3fn if self.device.type == "cuda" else torch.float32,  # FP8 시도
                        local_files_only=False,
                        token=hf_token,
                        low_cpu_mem_usage=True,
                        device_map="balanced"
                    )
                except:
                    # FP8 실패 시 FP16으로 폴백
                    self.pipe = FluxPipeline.from_pretrained(
                        self.config.model_name,
                        torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                        local_files_only=False,
                        token=hf_token,
                        low_cpu_mem_usage=True,
                        device_map="balanced"
                    )
                print("✅ Model downloaded successfully")
            except Exception as download_error:
                print(f"❌ Download failed: {str(download_error)}")
                print(f"Error type: {type(download_error).__name__}")
                
                # Check specific error types
                error_str = str(download_error).lower()
                if "disk" in error_str or "space" in error_str:
                    print("💡 Disk space issue - check available storage")
                elif "network" in error_str or "connection" in error_str:
                    print("💡 Network issue - check internet connection")
                elif "token" in error_str or "401" in error_str:
                    print("💡 Authentication issue - verify Hugging Face token")
                elif "403" in error_str:
                    print("💡 Access denied - request access to FLUX.1-schnell")
                else:
                    print("💡 General solutions:")
                    print("   - Check internet connection")
                    print("   - Verify Hugging Face token is valid") 
                    print("   - Make sure you have access to FLUX.1-schnell")
                    print("   - Check disk space (~30GB needed)")
                    print("   - Try running as administrator")
                
                raise download_error
        
        # Setup FLUX model components
        self.transformer = self.pipe.transformer.to(self.device)
        self.text_encoder = self.pipe.text_encoder.to(self.device)
        self.text_encoder_2 = self.pipe.text_encoder_2.to(self.device)
        self.tokenizer = self.pipe.tokenizer
        self.tokenizer_2 = self.pipe.tokenizer_2
        self.vae = self.pipe.vae.to(self.device)
        self.scheduler = self.pipe.scheduler
        
        lora_config = LoraConfig(
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type=TaskType.DIFFUSION,
        )
        
        self.transformer = get_peft_model(self.transformer, lora_config)
        
        print(f"LoRA parameters: {self.transformer.num_parameters()}")
        print(f"Trainable parameters: {sum(p.numel() for p in self.transformer.parameters() if p.requires_grad)}")
        
        self.transformer.train()
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        for param in self.text_encoder_2.parameters():
            param.requires_grad = False
        for param in self.vae.parameters():
            param.requires_grad = False
    
    def encode_text(self, prompts):
        with torch.no_grad():
            text_inputs = self.tokenizer(
                prompts,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)
            
            text_inputs_2 = self.tokenizer_2(
                prompts,
                padding="max_length",
                max_length=self.tokenizer_2.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)
            
            prompt_embeds = self.text_encoder(**text_inputs)[0]
            pooled_prompt_embeds = self.text_encoder_2(**text_inputs_2)[0]
            
            return prompt_embeds, pooled_prompt_embeds
    
    def encode_images(self, images):
        with torch.no_grad():
            latents = self.vae.encode(images).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor
            return latents
    
    def train_step(self, batch):
        images = batch['image'].to(self.device, dtype=torch.float16 if self.device.type == "cuda" else torch.float32)
        captions = batch['caption']
        
        with torch.no_grad():
            latents = self.encode_images(images)
            prompt_embeds, pooled_prompt_embeds = self.encode_text(captions)
        
        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0, self.scheduler.config.num_train_timesteps, (latents.shape[0],), 
            device=self.device
        ).long()
        
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)
        
        model_pred = self.transformer(
            hidden_states=noisy_latents,
            timestep=timesteps,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_prompt_embeds,
            return_dict=False,
        )[0]
        
        if self.scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.scheduler.config.prediction_type == "v_prediction":
            target = self.scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.scheduler.config.prediction_type}")
        
        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        return loss
    
    def train(self):
        if not validate_dataset(self.config.data_dir, self.config.caption_file):
            raise ValueError("Dataset validation failed!")
        
        dataloader = create_dataloader(
            self.config.data_dir,
            self.config.caption_file,
            batch_size=self.config.batch_size,
            image_size=self.config.image_size
        )
        
        try:
            from bitsandbytes.optim import AdamW8bit
            optimizer = AdamW8bit(
                self.transformer.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
            print("Using 8-bit AdamW optimizer")
        except ImportError:
            optimizer = AdamW(
                self.transformer.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
            print("Using standard AdamW optimizer")
        
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.config.epochs * len(dataloader),
            eta_min=self.config.learning_rate * 0.1
        )
        
        if self.config.use_wandb:
            wandb.init(project="flux-lora-training", config=vars(self.config))
        
        print(f"Starting training for {self.config.epochs} epochs...")
        print(f"Dataset size: {len(dataloader.dataset)}")
        print(f"Batches per epoch: {len(dataloader)}")
        
        global_step = 0
        for epoch in range(self.config.epochs):
            epoch_loss = 0.0
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.config.epochs}")
            
            for batch_idx, batch in enumerate(progress_bar):
                optimizer.zero_grad()
                
                loss = self.train_step(batch)
                loss.backward()
                
                if self.config.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.transformer.parameters(), self.config.max_grad_norm)
                
                optimizer.step()
                scheduler.step()
                
                epoch_loss += loss.item()
                global_step += 1
                
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'lr': f'{scheduler.get_last_lr()[0]:.2e}'
                })
                
                if self.config.use_wandb and global_step % self.config.log_steps == 0:
                    wandb.log({
                        'loss': loss.item(),
                        'learning_rate': scheduler.get_last_lr()[0],
                        'epoch': epoch,
                        'global_step': global_step
                    })
                
                if global_step % self.config.save_steps == 0:
                    self.save_checkpoint(global_step)
            
            avg_loss = epoch_loss / len(dataloader)
            print(f"Epoch {epoch+1} completed. Average loss: {avg_loss:.4f}")
            
            if (epoch + 1) % self.config.save_epochs == 0:
                self.save_model(f"epoch_{epoch+1}")
        
        self.save_model("final")
        print("Training completed!")
    
    def save_checkpoint(self, step):
        checkpoint_dir = os.path.join(self.config.output_dir, f"checkpoint-{step}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.transformer.save_pretrained(checkpoint_dir)
        
        with open(os.path.join(checkpoint_dir, "training_config.json"), 'w') as f:
            json.dump(vars(self.config), f, indent=2)
    
    def save_model(self, suffix):
        output_dir = os.path.join(self.config.output_dir, f"flux-lora-{suffix}")
        os.makedirs(output_dir, exist_ok=True)
        self.transformer.save_pretrained(output_dir)
        
        with open(os.path.join(output_dir, "training_config.json"), 'w') as f:
            json.dump(vars(self.config), f, indent=2)
        
        print(f"Model saved to {output_dir}")

class TrainingConfig:
    def __init__(self):
        self.model_name = os.getenv("FLUX_MODEL_NAME", "black-forest-labs/FLUX.1-schnell")
        self.data_dir = "./training_data"
        self.caption_file = None
        self.output_dir = "./flux_lora_output"
        self.image_size = 512
        self.batch_size = 1
        self.epochs = 10
        self.learning_rate = 1e-4
        self.weight_decay = 0.01
        self.lora_rank = 16
        self.lora_alpha = 32
        self.lora_dropout = 0.1
        self.target_modules = ["to_k", "to_q", "to_v", "to_out.0"]
        self.max_grad_norm = 1.0
        self.save_steps = 500
        self.save_epochs = 1
        self.log_steps = 50
        self.use_wandb = False

def main():
    parser = argparse.ArgumentParser(description="Train Flux LoRA")
    parser.add_argument("--data_dir", type=str, required=True, help="Training data directory")
    parser.add_argument("--caption_file", type=str, help="Caption file (JSON/CSV)")
    parser.add_argument("--output_dir", type=str, default="./flux_lora_output", help="Output directory")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--lora_rank", type=int, default=16, help="LoRA rank")
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases logging")
    
    args = parser.parse_args()
    
    config = TrainingConfig()
    for key, value in vars(args).items():
        if value is not None:
            setattr(config, key, value)
    
    os.makedirs(config.output_dir, exist_ok=True)
    
    trainer = FluxLoRATrainer(config)
    trainer.setup_model()
    trainer.train()

if __name__ == "__main__":
    main()