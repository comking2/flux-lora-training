from diffusers import FLUXPipeline
from peft import LoraConfig, get_peft_model

pipe = FLUXPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.float16)
model = pipe.unet