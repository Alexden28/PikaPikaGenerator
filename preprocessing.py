import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
import json
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PokemonDataProcessor:
    
    
    def __init__(self, config: Dict):
        self.config = config
        self.df = None
        self.tokenizer = None
        self.stats = {}
        
    def load_and_validate_data(self) -> pd.DataFrame:
        """Load and validate Pokemon CSV data"""
        logger.info("Loading Pokemon dataset...")
        
        # prova più encodings 
        encodings = ['utf-16', 'utf-8', 'cp1252', 'latin1']
        for encoding in encodings:
            try:
                self.df = pd.read_csv(
                    self.config['data']['raw_data_path'], 
                    encoding=encoding,
                    sep='\t'
                )
                logger.info(f"Successfully loaded with encoding: {encoding}")
                break
            except Exception as e:
                continue
        
        if self.df is None:
            raise ValueError("Could not load CSV with any encoding")
        
        # Convalida le columns
        required_cols = ['description', 'english_name', 'national_number']
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Pulisce i data
        initial_count = len(self.df)
        self.df = self.df.dropna(subset=['description'])
        self.df = self.df[self.df['description'].str.len() > 10]  # Rimuovi descrizioni troppo corte
        final_count = len(self.df)
        
        logger.info(f"Loaded {final_count} valid Pokemon (removed {initial_count - final_count})")
        
        # Store statistics
        self.stats['total_pokemon'] = final_count
        self.stats['description_lengths'] = self.df['description'].str.len().describe().to_dict()
        
        return self.df
    
    def validate_images(self) -> Dict[int, str]:
        """Validate available images and create mapping"""
        logger.info("Validating Pokemon images...")
        
        images_dir = Path(self.config['data']['images_dir'])
        valid_images = {}
        missing_images = []
        
        for idx, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Validating images"):
            national_number = int(row['national_number'])
            
            # Controlla possibili percorsi dell'immagine
            possible_paths = [
                images_dir / f"{national_number:03d}.png",
                images_dir / "small_images" / f"{national_number:03d}.png",
                images_dir / f"{national_number}.png"
            ]
            
            image_found = False
            for img_path in possible_paths:
                if img_path.exists():
                    try:
                        # Valida immagini che siano immagini valide
                        img = Image.open(img_path)
                        img.verify()
                        valid_images[idx] = str(img_path)
                        image_found = True
                        break
                    except:
                        continue
            
            if not image_found:
                missing_images.append(national_number)
        
        logger.info(f"Found {len(valid_images)} valid images")
        if missing_images:
            logger.warning(f"Missing images for {len(missing_images)} Pokemon: {missing_images[:10]}...")
        
        self.stats['valid_images'] = len(valid_images)
        self.stats['missing_images'] = len(missing_images)
        
        return valid_images
    
    def create_splits(self, valid_indices: List[int]) -> Dict[str, List[int]]:
        """Create train/val/test splits"""
        logger.info("Creating data splits...")
        
        # Filter to only valid entries
        valid_df = self.df.loc[valid_indices]
        
        # Shuffle
        indices = valid_df.index.tolist()
        np.random.seed(self.config['project']['seed'])
        np.random.shuffle(indices)
        
        # Calculate split sizes
        n_total = len(indices)
        n_train = int(n_total * self.config['data']['train_split'])
        n_val = int(n_total * self.config['data']['val_split'])
        
        # Create splits
        splits = {
            'train': indices[:n_train],
            'val': indices[n_train:n_train + n_val],
            'test': indices[n_train + n_val:]
        }
        
        logger.info(f"Split sizes - Train: {len(splits['train'])}, "
                   f"Val: {len(splits['val'])}, Test: {len(splits['test'])}")
        
        return splits
    
    def save_processed_data(self, valid_images: Dict, splits: Dict):
        """Salva i dati processati in file JSON e CSV"""
        output_dir = Path(self.config['data']['processed_data_path'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Salva mappatura delle immagini valide
        with open(output_dir / 'valid_images.json', 'w') as f:
            json.dump(valid_images, f)
        
        # Salva gli indici delle suddivisioni
        with open(output_dir / 'splits.json', 'w') as f:
            json.dump(splits, f)
        
        # Salva il dataframe processato
        self.df.to_csv(output_dir / 'processed_pokemon.csv', index=True)
        
        # Salva le statistiche del dataset
        with open(output_dir / 'dataset_stats.json', 'w') as f:
            json.dump(self.stats, f, indent=2)
        
        logger.info(f"Saved processed data to {output_dir}")
    
    def process(self):
        """Avvia il processo di preprocessing"""
        logger.info("Starting data preprocessing...")
        
        # Carica e valida i dati
        self.load_and_validate_data()
        
        # Convalida le immagini
        valid_images = self.validate_images()
        valid_indices = list(valid_images.keys())
        
        # Crea le suddivisioni
        splits = self.create_splits(valid_indices)
        
        # Salva i dati processati
        self.save_processed_data(valid_images, splits)
        
        logger.info("Data preprocessing completed!")
        return self.stats


class PokemonDataset(Dataset):
    """Dataset per Pokemon con immagini e descrizioni testuali"""
    
    def __init__(
        self, 
        data_path: str,
        split: str,
        tokenizer_name: str,
        max_length: int = 128,
        image_size: int = 215,
        augment: bool = False
    ):
        super().__init__()
        self.split = split
        self.max_length = max_length
        self.image_size = image_size
        self.augment = augment
        
        # Carica i dati processati
        processed_dir = Path(data_path)
        
        # Carica il dataframe 
        self.df = pd.read_csv(processed_dir / 'processed_pokemon.csv', index_col=0)
        
        # Carica la mappatura delle immagini valide
        with open(processed_dir / 'valid_images.json', 'r') as f:
            self.valid_images = {int(k): v for k, v in json.load(f).items()}
        
        # Carica le suddivisioni
        with open(processed_dir / 'splits.json', 'r') as f:
            splits = json.load(f)
            self.indices = splits[split]
        
        # Inizializza il tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Setta le trasformazioni delle immagini
        self._setup_transforms()
        
        logger.info(f"Loaded {split} dataset with {len(self.indices)} samples")
    
    def _setup_transforms(self):
        """Setta le trasformazioni delle immagini"""
        import torchvision.transforms as transforms
        
        # Definisci le trasformazioni di base
        basic_transforms = [
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ]
        
        if self.augment and self.split == 'train':
            # Aggiungi trasformazioni di data augmentation
            self.transform = transforms.Compose([
                transforms.Resize((self.image_size + 20, self.image_size + 20)),
                transforms.RandomCrop(self.image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:
            self.transform = transforms.Compose(basic_transforms)
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        # Ottiene l'indice del dataframe
        df_idx = self.indices[idx]
        row = self.df.loc[df_idx]
        
        # Carica e processa l'immagine
        image_path = self.valid_images[df_idx]
        image = Image.open(image_path).convert('RGBA')
        
        # Converti RGBA a RGB se necessario
        if image.mode == 'RGBA':
            background = Image.new('RGB', image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[3])
            image = background
        
        image = self.transform(image)
        
        # Tokenizza il testo
        text = row['description']
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'image': image,
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'text': text,
            'name': row['english_name'],
            'idx': df_idx
        }




def create_dataloaders(config: Dict, tokenizer_name: str) -> Dict[str, DataLoader]:
    """Create dataloaders for all splits"""
    dataloaders = {}
    
    # Leggi image_size dal config con fallback
    image_size = config.get('data', {}).get('image_size', 215)
    
    for split in ['train', 'val', 'test']:
        dataset = PokemonDataset(
            data_path=config['data']['processed_data_path'],
            split=split,
            tokenizer_name=tokenizer_name,
            max_length=config['model']['encoder']['max_length'],
            image_size=image_size,  # ← AGGIUNGI QUESTA RIGA!
            augment=True if split == 'train' else False
        )
        
        dataloaders[split] = DataLoader(
            dataset,
            batch_size=config['training']['batch_size'],
            shuffle=True if split == 'train' else False,
            num_workers=config['training']['num_workers'],
            pin_memory=torch.cuda.is_available(),
            drop_last=True if split == 'train' else False
        )
    
    return dataloaders


if __name__ == "__main__":
    # Test preprocessing
    import yaml
    
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    processor = PokemonDataProcessor(config)
    stats = processor.process()
    
    print("\nDataset Statistics:")
    print(json.dumps(stats, indent=2))