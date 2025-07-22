import os
import argparse
import torch
from diffusers import FluxPipeline
from peft import PeftModel
from PIL import Image
import json

class FluxLoRAInference:
    def __init__(self, base_model_path=None, lora_path=None, device=None):
        if base_model_path is None:
            base_model_path = os.getenv("FLUX_MODEL_NAME", "black-forest-labs/FLUX.1-schnell")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Loading FLUX model from {base_model_path}...")
        
        # Get Hugging Face token from environment
        hf_token = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")
        if hf_token:
            print("🔑 Using Hugging Face token from environment")
        else:
            print("⚠️  No Hugging Face token found. Make sure to login: huggingface-cli login")
        
        # Check if model exists locally first
        try:
            print(f"Checking for local model: {base_model_path}")
            self.pipe = FluxPipeline.from_pretrained(
                base_model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                local_files_only=True,  # Try local first
                token=hf_token
            )
            print("✅ Found local model, using cached version")
        except Exception as e:
            print(f"❌ Local model not found: {e}")
            print("🔄 Downloading model from Hugging Face...")
            print("⚠️  This may take a while for first download (~17GB)")
            self.pipe = FluxPipeline.from_pretrained(
                base_model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                local_files_only=False,  # Download if needed
                token=hf_token
            )
            print("✅ Model downloaded successfully")
        
        if lora_path and os.path.exists(lora_path):
            print(f"Loading LoRA weights from {lora_path}...")
            self.pipe.transformer = PeftModel.from_pretrained(
                self.pipe.transformer,
                lora_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
        
        self.pipe = self.pipe.to(self.device)
        print(f"Pipeline loaded on {self.device}")
    
    def generate_image(
        self, 
        prompt, 
        negative_prompt="", 
        height=1024, 
        width=1024, 
        num_inference_steps=50, 
        guidance_scale=7.5, 
        seed=None,
        output_path=None
    ):
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None
        
        print(f"Generating image with prompt: '{prompt}'")
        
        with torch.autocast(self.device):
            image = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt if negative_prompt else None,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator
            ).images[0]
        
        if output_path:
            image.save(output_path)
            print(f"Image saved to: {output_path}")
        
        return image
    
    def batch_generate(self, prompts, output_dir="./generated_images", **kwargs):
        os.makedirs(output_dir, exist_ok=True)
        
        images = []
        for i, prompt in enumerate(prompts):
            output_path = os.path.join(output_dir, f"generated_{i+1:03d}.png")
            image = self.generate_image(prompt, output_path=output_path, **kwargs)
            images.append(image)
        
        return images
    
    def test_lora(self, test_prompts=None, output_dir="./test_results"):
        if test_prompts is None:
            test_prompts = [
                "a beautiful landscape with mountains and lake",
                "a portrait of a person in artistic style",
                "a futuristic cityscape at night",
                "a still life with flowers and fruits"
            ]
        
        print(f"Testing LoRA with {len(test_prompts)} prompts...")
        images = self.batch_generate(test_prompts, output_dir)
        
        results = []
        for i, (prompt, image) in enumerate(zip(test_prompts, images)):
            result = {
                "prompt": prompt,
                "image_path": os.path.join(output_dir, f"generated_{i+1:03d}.png"),
                "image_size": image.size
            }
            results.append(result)
        
        results_path = os.path.join(output_dir, "test_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Test results saved to: {results_path}")
        return results

def compare_models(base_model_path, lora_path, prompts, output_dir="./comparison"):
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading base model...")
    base_inference = FluxLoRAInference(base_model_path)
    
    print("Loading LoRA model...")
    lora_inference = FluxLoRAInference(base_model_path, lora_path)
    
    comparison_results = []
    
    for i, prompt in enumerate(prompts):
        print(f"\nGenerating comparison {i+1}/{len(prompts)}: {prompt}")
        
        base_output = os.path.join(output_dir, f"base_{i+1:03d}.png")
        lora_output = os.path.join(output_dir, f"lora_{i+1:03d}.png")
        
        base_image = base_inference.generate_image(prompt, output_path=base_output)
        lora_image = lora_inference.generate_image(prompt, output_path=lora_output)
        
        result = {
            "prompt": prompt,
            "base_image": base_output,
            "lora_image": lora_output
        }
        comparison_results.append(result)
    
    results_path = os.path.join(output_dir, "comparison_results.json")
    with open(results_path, 'w') as f:
        json.dump(comparison_results, f, indent=2)
    
    print(f"\nComparison results saved to: {results_path}")
    return comparison_results

def main():
    parser = argparse.ArgumentParser(description="Flux LoRA Inference")
    parser.add_argument("--lora_path", type=str, help="Path to LoRA model")
    parser.add_argument("--prompt", type=str, help="Prompt for generation")
    parser.add_argument("--prompts_file", type=str, help="File containing prompts (one per line)")
    parser.add_argument("--output_dir", type=str, default="./generated_images", help="Output directory")
    parser.add_argument("--base_model", type=str, default=os.getenv("FLUX_MODEL_NAME", "black-forest-labs/FLUX.1-schnell"), help="Base model path")
    parser.add_argument("--height", type=int, default=1024, help="Image height")
    parser.add_argument("--width", type=int, default=1024, help="Image width")
    parser.add_argument("--steps", type=int, default=50, help="Number of inference steps")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Guidance scale")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--test", action="store_true", help="Run test with default prompts")
    parser.add_argument("--compare", action="store_true", help="Compare base model vs LoRA")
    
    args = parser.parse_args()
    
    if args.test:
        inference = FluxLoRAInference(args.base_model, args.lora_path)
        inference.test_lora(output_dir=args.output_dir)
        return
    
    if args.compare:
        if not args.lora_path:
            raise ValueError("--lora_path is required for comparison")
        
        if args.prompts_file:
            with open(args.prompts_file, 'r') as f:
                prompts = [line.strip() for line in f.readlines() if line.strip()]
        else:
            prompts = [
                "a beautiful landscape with mountains and lake",
                "a portrait of a person in artistic style"
            ]
        
        compare_models(args.base_model, args.lora_path, prompts, args.output_dir)
        return
    
    inference = FluxLoRAInference(args.base_model, args.lora_path)
    
    if args.prompts_file:
        with open(args.prompts_file, 'r') as f:
            prompts = [line.strip() for line in f.readlines() if line.strip()]
        inference.batch_generate(
            prompts,
            output_dir=args.output_dir,
            height=args.height,
            width=args.width,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed
        )
    elif args.prompt:
        output_path = os.path.join(args.output_dir, "generated.png")
        os.makedirs(args.output_dir, exist_ok=True)
        inference.generate_image(
            args.prompt,
            height=args.height,
            width=args.width,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
            output_path=output_path
        )
    else:
        print("Please provide either --prompt, --prompts_file, --test, or --compare")

if __name__ == "__main__":
    main()