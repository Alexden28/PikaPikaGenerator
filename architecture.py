"""
Architettura completa del modello PikaPikaGenerator
implementa Encoder - Decoder con Attention per la generazione di sprite a partire da descrizioni testuali.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
import numpy as np
from typing import Dict, Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)


class TextEncoder(nn.Module):
    """Encoder testuale basato su BERT con possibilità di fine-tuning"""
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config['model']['encoder']
        
        # In questa fase carico il modello pre-addestrato
        self.bert = AutoModel.from_pretrained(self.config['model_name'])
        self.bert_dim = self.bert.config.hidden_size
        
        # Layer di per adattare la dimensione dell'output alla dimesione desiderata
        self.projection = nn.Sequential(
            nn.Linear(self.bert_dim, self.config['hidden_dim']),
            nn.LayerNorm(self.config['hidden_dim']),
            nn.Dropout(self.config['dropout'])
        )
        
        # Opzione Facoltativa per bloccare i pesi di BERT all'inizio per evitare aggiornamenti
        self.freeze_bert_layers(freeze=True)
        
    def freeze_bert_layers(self, freeze: bool = True):
        """Congela o sgrava i parametri di BERT per il training"""
        for param in self.bert.parameters():
            param.requires_grad = not freeze
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass attraverso l'encoder testuale
        Returns: (sequence_output, pooled_output)
        """
        # Eseguo il forward su BERT
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Prende l'output per ogni token (sequence_output))
        sequence_output = outputs.last_hidden_state  # [batch, seq_len, bert_dim]
        
        # Proietta alla dimensione desiderata
        sequence_output = self.projection(sequence_output)  # [batch, seq_len, hidden_dim]
        
        # ottiene  output (CLS token)
        pooled_output = sequence_output[:, 0, :]  # [batch, hidden_dim]
        
        return sequence_output, pooled_output


class MultiHeadAttention(nn.Module):
    """Meccanismo di multi-head attention per allineamento tra testo e immagini"""
    
    def __init__(self, config: Dict):
        super().__init__()
        self.hidden_dim = config['model']['encoder']['hidden_dim']
        self.num_heads = 8
        self.head_dim = self.hidden_dim // self.num_heads
        
        # Layer di proiezione per Query, Key, Value 
        self.q_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.k_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.v_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.out_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        
        self.dropout = nn.Dropout(config['model']['encoder']['dropout'])
        
    def forward(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Applica multi-head attention
        Returns: (output, attention_weights)
        """
        batch_size, seq_len = key.shape[:2]
        
        # Proietta e ridimensiona per multi-head attention
        Q = self.q_proj(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Calcola attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        
        # Applica maschera se fornita
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)  # Aggiungi dimensione batch e head
            mask = mask.unsqueeze(1).unsqueeze(2)  # Aggiungi dimensione batch e head
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Applica softmax per ottenere i attention weights
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Moltiplica i weights con i valori
        context = torch.matmul(attention_weights, V)
        
        # Rifirmatta e proietta l'output
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_dim)
        output = self.out_proj(context)
        
        # Media degli attention weights tra gli heads, utile per la visualization
        attention_weights = attention_weights.mean(dim=1)
        
        return output, attention_weights


class ResidualBlock(nn.Module):
    """Blocco Residuo per il generatore CNN"""
    
    def __init__(self, in_channels: int, out_channels: int, upsample: bool = False):
        super().__init__()
        self.upsample = upsample
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Connessione Shortcut
        if in_channels != out_channels or upsample:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
        
        if upsample:
            self.upsample_layer = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        if self.upsample:
            out = self.upsample_layer(out)
            residual = self.upsample_layer(residual)
        
        residual = self.shortcut(residual)
        out = F.relu(out + residual)
        
        return out


class SpriteGenerator(nn.Module):
    """Generatore CNN avanzato con connessioni residuali"""
    
    def __init__(self, config: Dict):
        super().__init__()
        self.text_dim = config['model']['encoder']['hidden_dim']
        self.noise_dim = config['model']['generator']['noise_dim']
        self.base_channels = config['model']['generator']['base_channels']
        self.output_size = config['model']['generator']['output_size']
        
        # Dimensione Spaziale iniziale
        self.init_size = 4  # Dimensione iniziale dell'immagine (4x4)
        
        # Proiezione Iniziale
        self.fc = nn.Sequential(
            nn.Linear(self.text_dim + self.noise_dim, self.base_channels * self.init_size * self.init_size),
            nn.BatchNorm1d(self.base_channels * self.init_size * self.init_size),
            nn.ReLU(inplace=True)
        )
        
        # Blocchi residuali con upsampling progressivo fino alla dimensione finale
        self.blocks = nn.ModuleList([
            ResidualBlock(self.base_channels, self.base_channels, upsample=True),      # 4x4 -> 8x8
            ResidualBlock(self.base_channels, self.base_channels // 2, upsample=True), # 8x8 -> 16x16
            ResidualBlock(self.base_channels // 2, self.base_channels // 4, upsample=True), # 16x16 -> 32x32
            ResidualBlock(self.base_channels // 4, self.base_channels // 8, upsample=True), # 32x32 -> 64x64
            ResidualBlock(self.base_channels // 8, self.base_channels // 16, upsample=True), # 64x64 -> 128x128
            ResidualBlock(self.base_channels // 16, 64, upsample=True), # 128x128 -> 256x256
        ])
        
        # Convoluzione finale per generare immagine RGB con valori in [-1,1]
        self.final_conv = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Tanh()
        )
        
        # Pooling adattivo per ottenere esttamente la dimensione dell'output_size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((self.output_size, self.output_size))
        
    def forward(self, text_features: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Generate sprite from text features and noise"""
        batch_size = text_features.shape[0]
        
        # Se il rumore non viene fornito, viene generato casualmente
        if noise is None:
            noise = torch.randn(batch_size, self.noise_dim, device=text_features.device)
        
        # Concatena features testuali e rumore
        combined = torch.cat([text_features, noise], dim=1)
        
        # Proiezione iniziale e reshape in feature map
        x = self.fc(combined)
        x = x.view(batch_size, self.base_channels, self.init_size, self.init_size)
        
        # Unshampling progressivo tramite blocchi residuali
        for block in self.blocks:
            x = block(x)
        
        # Convoluzioni finali per immagine RGB
        x = self.final_conv(x)
        
        # Pooling adattivo per ottenere la dimensione finale desiderata
        x = self.adaptive_pool(x)
        
        return x


class PikaPikaGenerator(nn.Module):
    """Modello completo per text-to sprite generation"""
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        # Componenti iniziali
        self.text_encoder = TextEncoder(config)
        self.attention = MultiHeadAttention(config)
        self.generator = SpriteGenerator(config)
        
        # Carica il Tokenizer per la tokenizzazione del testo
        self.tokenizer = AutoTokenizer.from_pretrained(config['model']['encoder']['model_name'])
        
    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass attraverso il model completo
        Restituisce un dizionario con lle immagini generate e gli attention weights
        """
        # Encoder testuale
        sequence_output, pooled_output = self.text_encoder(input_ids, attention_mask)
        
        # Applica self-attention per rifinire le features testuali
        attended_features, attention_weights = self.attention(
            query=pooled_output.unsqueeze(1),
            key=sequence_output,
            value=sequence_output,
            mask=attention_mask
        )
        
        attended_features = attended_features.squeeze(1)  # Rimuove la dimensione del sequence length
        
        # Generazione sprite
        generated_image = self.generator(attended_features, noise)
        
        return {
            'generated_image': generated_image,
            'attention_weights': attention_weights,
            'text_features': attended_features
        }
    
    def generate(
        self, 
        text: str, 
        noise: Optional[torch.Tensor] = None,
        device: str = 'cpu'
    ) -> np.ndarray:
        """Genera sprite dalle descrizioni testuali"""
        self.eval()
        
        # Tokenizza il testo
        encoding = self.tokenizer(
            text,
            max_length=self.config['model']['encoder']['max_length'],
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Sposta i tensori sul dispositivo specificato
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask, noise)
            generated_image = outputs['generated_image']
        
        # Converte l'immagine generata in un array NumPy
        image = generated_image.squeeze(0).cpu()
        image = (image + 1) / 2  # Denormalize from [-1, 1] to [0, 1]
        image = image.permute(1, 2, 0).numpy()
        image = (image * 255).astype(np.uint8)
        
        return image
    
    def get_attention_visualization(
        self, 
        text: str,
        device: str = 'cpu'
    ) -> Tuple[np.ndarray, List[str], np.ndarray]:
        """Ottiene gli attention weights e i tokens per visualizzazione"""
        self.eval()
        
        # Tokenizza il testo
        encoding = self.tokenizer(
            text,
            max_length=self.config['model']['encoder']['max_length'],
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Ottiene i tokens
        tokens = self.tokenizer.convert_ids_to_tokens(encoding['input_ids'][0])
        
        # Sposta i tensori sul dispositivo specificato
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            attention_weights = outputs['attention_weights']
            generated_image = outputs['generated_image']
        
        # Processa gli attention weights
        attention_weights = attention_weights.squeeze().cpu().numpy()
        
        # Converte l'immagine generata in un array NumPy
        image = generated_image.squeeze(0).cpu()
        image = (image + 1) / 2
        image = image.permute(1, 2, 0).numpy()
        image = (image * 255).astype(np.uint8)
        
        return image, tokens, attention_weights


class Discriminator(nn.Module):
    """Discriminator per la valutazione della qualità delle immagini generate"""
    
    def __init__(self, config: Dict):
        super().__init__()
        self.input_size = config['model']['generator']['output_size']
        
        # Progressive downsampling con blocchi convoluzionali
        self.blocks = nn.Sequential(
            # 215x215 -> 107x107
            self._make_block(3, 64, downsample=True),
            # 107x107 -> 53x53
            self._make_block(64, 128, downsample=True),
            # 53x53 -> 26x26
            self._make_block(128, 256, downsample=True),
            # 26x26 -> 13x13
            self._make_block(256, 512, downsample=True),
            # 13x13 -> 6x6
            self._make_block(512, 512, downsample=True),
        )
        
        # Media pooling per ridurre le dimensioni
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Classificatore finale
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )
    
    def _make_block(self, in_channels: int, out_channels: int, downsample: bool = True):
        """Crea un blocco convoluzionale con opzioni di downsampling"""
        layers = [
            nn.Conv2d(in_channels, out_channels, 4, 2 if downsample else 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass attraverso il discriminatore"""
        features = self.blocks(x)
        features = self.global_pool(features).view(features.size(0), -1)
        output = self.classifier(features)
        return output


def create_model(config: Dict) -> PikaPikaGenerator:
    """Funzione per creare e inizializzare il modello PikaPikaGenerator"""
    model = PikaPikaGenerator(config)
    
    # Inizializza i weights
    def init_weights(m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    
    model.generator.apply(init_weights)
    
    # Logging delle informazioni sul modello
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Created PikaPikaGenerator model")
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    return model


if __name__ == "__main__":
    # Test di creazione del modello e forward pass
    import yaml
    
    # Esempio config per il testing
    test_config = {
        'model': {
            'encoder': {
                'model_name': 'prajjwal1/bert-mini',
                'hidden_dim': 256,
                'max_length': 128,
                'dropout': 0.1
            },
            'generator': {
                'noise_dim': 100,
                'base_channels': 512,
                'output_size': 215
            }
        }
    }
    
    try:
        with open('configs/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print("Config file not found, using test config")
        config = test_config
    
    model = create_model(config)
    
    # Test forward pass
    batch_size = 2
    seq_len = 128
    
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    
    print("Testing forward pass...")
    outputs = model(input_ids, attention_mask)
    
    print(f"Generated image shape: {outputs['generated_image'].shape}")
    print(f"Attention weights shape: {outputs['attention_weights'].shape}")
    
    # Generazione del testo
    print("Testing text generation...")
    test_text = "A small yellow electric mouse Pokemon with red cheeks"
    image = model.generate(test_text)
    print(f"Generated sprite shape: {image.shape}")
    print("Architecture test completed successfully!")