import argparse
import yaml
import logging
from pathlib import Path
import torch
import numpy as np

from src.data.preprocessing import PokemonDataProcessor
from src.training.trainer import train_model
from app import main as launch_app

import warnings
import torch

# Sopprime alcuni warning comuni durante l'addestramento per un output più pulito
def suppress_training_warnings():
    """Suppress common training warnings for cleaner output"""
    
    # Sopprime pin_memory warnings
    warnings.filterwarnings("ignore", message=".*pin_memory.*")
    
    # Sopprime i warning di deprecazione di torchvision
    warnings.filterwarnings("ignore", message=".*pretrained.*deprecated.*")
    warnings.filterwarnings("ignore", message=".*Arguments other than a weight enum.*")
    
    # Sopprime i warning di autocast deprecati
    warnings.filterwarnings("ignore", message=".*autocast.*deprecated.*")
    
    print("🔇 Training warnings suppressed for cleaner output")
    print(f"📱 Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print("=" * 50)

# Chiama la funzione per sopprimere i warning all'avvio
if __name__ == "__main__":
    suppress_training_warnings()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def preprocess_data(config: dict):
    """Avvia la pipeline di data preprocessing"""
    logger.info("Starting data preprocessing...")
    
    processor = PokemonDataProcessor(config)
    stats = processor.process()
    
    logger.info("Data preprocessing completed!")
    logger.info(f"Dataset statistics:")
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")
    
    return stats


def train(config: dict, resume: bool = False):
    """Addestra il modello"""
    logger.info("Starting model training...")
    
    # Setta random seed per riproducibilità
    torch.manual_seed(config['project']['seed'])
    np.random.seed(config['project']['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['project']['seed'])
    
    # Addestra il modello
    trainer = train_model(config)
    
    logger.info("Training completed!")


def evaluate(config: dict):
    """Valuta il modello"""
    logger.info("Starting model evaluation...")
    
    from src.utils.evaluation import evaluate_model
    
    results = evaluate_model(config)
    
    logger.info("Evaluation completed!")
    logger.info(f"Results: {results}")
    
    return results


def demo(config: dict):
    """Avvia la Gradio demo"""
    logger.info("Launching Gradio demo...")
    launch_app()


def main():
    parser = argparse.ArgumentParser(description="PikaPikaGenerator - Pokemon Sprite Generation")
    parser.add_argument(
        'action',
        choices=['preprocess', 'train', 'evaluate', 'demo', 'all'],
        help='Action to perform'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume training from checkpoint'
    )
    
    args = parser.parse_args()
    
    # Carica la configurazione
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Loaded configuration from {args.config}")
    logger.info(f"Action: {args.action}")
    
    # Esegue action
    if args.action == 'preprocess':
        preprocess_data(config)
    
    elif args.action == 'train':
        # Controlla se i dati preprocessati esistono
        processed_path = Path(config['data']['processed_data_path'])
        if not processed_path.exists():
            logger.info("Processed data not found. Running preprocessing first...")
            preprocess_data(config)
        
        train(config, resume=args.resume)
    
    elif args.action == 'evaluate':
        evaluate(config)
    
    elif args.action == 'demo':
        demo(config)
    
    elif args.action == 'all':
        # Avvia la pipeline completa
        logger.info("Running complete pipeline...")
        
        # 1. Preprocessing dei dati
        preprocess_data(config)
        
        # 2. Addestra il modello
        train(config)
        
        # 3. Evaluation del modello
        evaluate(config)
        
        # 4. Avvia la demo
        demo(config)
    
    logger.info("Done!")


if __name__ == "__main__":
    main()