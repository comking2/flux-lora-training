import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd

class FluxLoRADataset(Dataset):
    def __init__(self, data_dir, caption_file=None, image_size=512, tokenizer=None):
        self.data_dir = data_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        
        self.image_files = []
        self.captions = []
        
        if caption_file and os.path.exists(caption_file):
            self._load_captions_from_file(caption_file)
        else:
            self._load_images_and_captions()
            
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
    
    def _load_captions_from_file(self, caption_file):
        if caption_file.endswith('.json'):
            with open(caption_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for item in data:
                if 'image' in item and 'caption' in item:
                    image_path = os.path.join(self.data_dir, item['image'])
                    if os.path.exists(image_path):
                        self.image_files.append(image_path)
                        self.captions.append(item['caption'])
        
        elif caption_file.endswith('.csv'):
            df = pd.read_csv(caption_file)
            for _, row in df.iterrows():
                image_path = os.path.join(self.data_dir, row['image'])
                if os.path.exists(image_path):
                    self.image_files.append(image_path)
                    self.captions.append(row['caption'])
    
    def _load_images_and_captions(self):
        for filename in os.listdir(self.data_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(self.data_dir, filename)
                txt_path = os.path.join(self.data_dir, os.path.splitext(filename)[0] + '.txt')
                
                if os.path.exists(txt_path):
                    with open(txt_path, 'r', encoding='utf-8') as f:
                        caption = f.read().strip()
                    self.image_files.append(image_path)
                    self.captions.append(caption)
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        caption = self.captions[idx]
        
        try:
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            image = torch.zeros(3, self.image_size, self.image_size)
        
        return {
            'image': image,
            'caption': caption,
            'image_path': image_path
        }

def create_dataloader(data_dir, caption_file=None, batch_size=4, image_size=512, num_workers=4):
    dataset = FluxLoRADataset(data_dir, caption_file, image_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

def validate_dataset(data_dir, caption_file=None):
    dataset = FluxLoRADataset(data_dir, caption_file)
    print(f"Dataset validation:")
    print(f"Total samples: {len(dataset)}")
    
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Sample image shape: {sample['image'].shape}")
        print(f"Sample caption: {sample['caption'][:100]}...")
        print(f"Sample image path: {sample['image_path']}")
    else:
        print("No valid samples found!")
    
    return len(dataset) > 0