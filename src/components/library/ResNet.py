from torchvision.models import resnet50, ResNet50_Weights
from src.entity import LibraryConfig
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from pathlib import Path
import cv2
import os
import torch 
import numpy as np
import torch.nn as nn
from PIL import Image

class CraterDataset(Dataset):
    def __init__(self,path:Path):
        self.path=path
        self.crops = os.listdir(path)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.crops)

    def __getitem__(self, idx):
        filename=self.crops[idx]
        full_path=os.path.join(self.path,filename)
        img = cv2.imread(full_path)

        if img is None:
            raise ValueError(f"Failed to read image: {full_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
        img = Image.fromarray(img)                 
        return self.transform(img)

class Embedding:
    def __init__(self,config:LibraryConfig):
        self.config=config
        self.device= "cuda" if torch.cuda.is_available() else "cpu"

        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.model = nn.Sequential(*list(backbone.children())[:-1])

        self.model.to(self.device)

    
    def run(self):
        
        dataset = CraterDataset(Path(self.config.data_path))
        loader = DataLoader(dataset, batch_size=16, shuffle=False)

        self.model.eval()
        all_embeddings = []

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                embeddings = self.model(batch)        
                embeddings = embeddings.view(embeddings.size(0), -1)
                all_embeddings.append(embeddings.cpu().numpy())
    
        all_embeddings = np.concatenate(all_embeddings, axis=0)
        filenames = dataset.crops

        return all_embeddings,filenames
    
    def embed_single(self,image):
        self.model.eval()
        transform = transforms.ToTensor()
        with torch.no_grad():
            image = Image.fromarray(image)
            image = transform(image)
            image = image.unsqueeze(0) 
            image = image.to(self.device)
            embeddings = self.model(image)         
            embeddings = embeddings.view(embeddings.size(0), -1)
  

            return embeddings.cpu().numpy()