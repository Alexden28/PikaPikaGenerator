import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont
from typing import List, Tuple, Optional, Dict
import io


def create_sample_grid(
    images: List[np.ndarray],
    texts: List[str],
    grid_size: Tuple[int, int] = None,
    image_size: int = 215
) -> Image.Image:
    n_images = len(images)
    
    # Auto-calcola la griglia se non specificata
    if grid_size is None:
        cols = int(np.ceil(np.sqrt(n_images)))
        rows = int(np.ceil(n_images / cols))
    else:
        rows, cols = grid_size
    
    # Crea la griglia
    margin = 10
    text_height = 30
    cell_width = image_size + 2 * margin
    cell_height = image_size + text_height + 2 * margin
    
    grid_width = cols * cell_width
    grid_height = rows * cell_height
    
    # Crea uno sfondo bianco per la griglia
    grid = Image.new('RGB', (grid_width, grid_height), color='white')
    draw = ImageDraw.Draw(grid)
    
    # Prova a caricare un font, altrimenti usa il font di default
    try:
        font = ImageFont.truetype("arial.ttf", 12)
    except:
        font = ImageFont.load_default()
    
    # Posiziona le immagini e i testi nella griglia
    for idx, (img, text) in enumerate(zip(images, texts)):
        row = idx // cols
        col = idx % cols
        
        x = col * cell_width + margin
        y = row * cell_height + margin
        
        # Converti l'immagine in PIL se è un array NumPy
        if isinstance(img, np.ndarray):
            img_pil = Image.fromarray(img)
        else:
            img_pil = img
        
        # Ridimensiona l'immagine se necessario
        if img_pil.size != (image_size, image_size):
            img_pil = img_pil.resize((image_size, image_size), Image.Resampling.LANCZOS)
        
        # Incolla l'immagine nella griglia
        grid.paste(img_pil, (x, y))
        
        # Aggiungi il testo sotto l'immagine
        text_y = y + image_size + 5
        # Tronca il testo se troppo lungo
        if len(text) > 40:
            text = text[:37] + "..."
        
        # Centra il testo
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_x = x + (image_size - text_width) // 2
        
        draw.text((text_x, text_y), text, fill='black', font=font)
    
    return grid


def create_attention_heatmap(
    tokens: List[str],
    attention_weights: np.ndarray,
    max_tokens: int = 20
) -> Image.Image:
    # Limita il numero di token e pesi di attenzione
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
        attention_weights = attention_weights[:max_tokens]
    
    # Crea la figura per la heatmap
    plt.figure(figsize=(10, 8))
    
    # Crea la heatmap
    sns.heatmap(
        attention_weights.reshape(-1, 1),
        xticklabels=['Attention'],
        yticklabels=tokens,
        cmap='YlOrRd',
        cbar_kws={'label': 'Attention Weight'},
        annot=True,
        fmt='.3f'
    )
    
    plt.title('Attention Weights for Text Tokens')
    plt.tight_layout()
    
    # Converti in PIL Image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    plt.close()
    
    return img


def visualize_training_progress(
    losses: Dict[str, List[float]],
    metrics: Dict[str, List[float]],
    save_path: Optional[str] = None
) -> Image.Image:
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    # Grafico  losses
    ax = axes[0]
    for loss_name, loss_values in losses.items():
        ax.plot(loss_values, label=loss_name)
    ax.set_title('Training Losses')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Grafico delle metriche di validazione
    metric_names = ['ssim', 'psnr', 'l1_distance']
    for idx, metric_name in enumerate(metric_names):
        if metric_name in metrics:
            ax = axes[idx + 1]
            ax.plot(metrics[metric_name])
            ax.set_title(f'{metric_name.upper()}')
            ax.set_xlabel('Validation Step')
            ax.set_ylabel(metric_name)
            ax.grid(True, alpha=0.3)
    
    plt.suptitle('Training Progress', fontsize=16)
    plt.tight_layout()
    
    # Salva o converti in PIL Image
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    plt.close()
    
    return img


def plot_metrics(
    metrics_dict: Dict[str, float],
    title: str = "Evaluation Metrics"
) -> Image.Image:
    
    plt.figure(figsize=(10, 6))
    
    # Crea un grafico a barre per le metriche
    metric_names = list(metrics_dict.keys())
    metric_values = list(metrics_dict.values())
    
    bars = plt.bar(metric_names, metric_values, color='skyblue', edgecolor='navy')
    
    # Aggiungi le etichette sopra le barre
    for bar, value in zip(bars, metric_values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom')
    
    plt.title(title, fontsize=16)
    plt.xlabel('Metrics')
    plt.ylabel('Value')
    plt.ylim(0, max(metric_values) * 1.2)
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Converti in PIL Image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    plt.close()
    
    return img


def create_comparison_grid(
    real_images: List[np.ndarray],
    generated_images: List[np.ndarray],
    texts: List[str],
    n_samples: int = 8
) -> Image.Image:
    
    n_samples = min(n_samples, len(real_images))
    
    fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4 * n_samples))
    
    for i in range(n_samples):
        # Immagine Reale
        axes[i, 0].imshow(real_images[i])
        axes[i, 0].set_title('Real' if i == 0 else '')
        axes[i, 0].axis('off')
        
        # Genera immagine
        axes[i, 1].imshow(generated_images[i])
        axes[i, 1].set_title('Generated' if i == 0 else '')
        axes[i, 1].axis('off')
        
        # Descrizione testuale
        axes[i, 2].text(0.5, 0.5, texts[i][:50] + '...', 
                       ha='center', va='center', wrap=True, fontsize=10)
        axes[i, 2].set_title('Description' if i == 0 else '')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    
    # Converti in PIL Image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    plt.close()
    
    return img


if __name__ == "__main__":
    # Testa le funzioni di visualizzazione
    # Crea dati fittizi per il test
    dummy_images = [np.random.randint(0, 255, (215, 215, 3), dtype=np.uint8) for _ in range(8)]
    dummy_texts = [f"Test Pokemon {i}: A description of the Pokemon" for i in range(8)]
    
    # Giglia di esempio
    grid = create_sample_grid(dummy_images, dummy_texts)
    grid.save("test_grid.png")
    print("Created test grid")
    
    # Crea heatmap di attenzione di esempio
    dummy_tokens = ["A", "small", "yellow", "electric", "mouse", "Pokemon", "[PAD]"] * 3
    dummy_attention = np.random.rand(len(dummy_tokens))
    heatmap = create_attention_heatmap(dummy_tokens, dummy_attention)
    heatmap.save("test_attention.png")
    print("Created test attention heatmap")