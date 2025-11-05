import gradio as gr
import torch
import numpy as np
from PIL import Image
import yaml
import json
from pathlib import Path
import logging
from typing import Tuple, List, Optional

from src.models.architecture import create_model
from src.utils.visualization import create_attention_heatmap, plot_metrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PikaPikaGeneratorApp:
    """Classe principale applicazione per l'Interfaccia Gradio di PikaPikaGenerator"""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        # Carica la configurazione
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.model_loaded = False
        
        # Carica il modello
        self.load_model()
    
    def load_model(self):
        """Carica il modello addestrato"""
        try:
            model_path = Path(self.config['paths']['checkpoints_dir']) / 'best_model.pt'
            
            if model_path.exists():
                logger.info(f"Loading model from {model_path}")
                
                # Crea il modello
                self.model = create_model(self.config).to(self.device)
                
                # Carica il checkpoint
                checkpoint = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.eval()
                
                self.model_loaded = True
                logger.info("Model loaded successfully!")
            else:
                logger.warning(f"Model not found at {model_path}")
                self.model_loaded = False
                
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.model_loaded = False
    
    def generate_sprite(
        self,
        description: str,
        noise_scale: float = 1.0,
        show_attention: bool = False
    ) -> Tuple[Image.Image, Optional[Image.Image], str]:
        """Generate a Pokemon sprite from description"""
        
        if not self.model_loaded:
            return None, None, "❌ Model not loaded. Please train the model first."
        
        if not description.strip():
            return None, None, "❌ Please enter a description."
        
        try:
            # Genera sprite
            noise = torch.randn(1, self.config['model']['generator']['noise_dim']) * noise_scale
            noise = noise.to(self.device)
            
            sprite = self.model.generate(description, noise=noise, device=self.device)
            sprite_img = Image.fromarray(sprite)
            
            # Genera attenzione heatmap se richiesto
            attention_img = None
            if show_attention:
                _, tokens, attention_weights = self.model.get_attention_visualization(
                    description, device=self.device
                )
                attention_img = create_attention_heatmap(tokens, attention_weights)
            
            status = f"✅ Generated sprite for: {description[:50]}..."
            
            return sprite_img, attention_img, status
            
        except Exception as e:
            logger.error(f"Error generating sprite: {e}")
            return None, None, f"❌ Error: {str(e)}"
    
    def batch_generate(
        self,
        descriptions: str,
        num_variations: int = 3,
        noise_scale: float = 1.0
    ) -> Tuple[List[Image.Image], str]:
        """Genera Sprite Multiple per Descrizioni Multiple"""
        
        if not self.model_loaded:
            return [], "❌ Model not loaded."
        
        # Parse descriptions (one per line)
        desc_list = [d.strip() for d in descriptions.strip().split('\n') if d.strip()]
        
        if not desc_list:
            return [], "❌ Please enter at least one description."
        
        generated_images = []
        
        try:
            for desc in desc_list:
                for i in range(num_variations):
                    # Rumore differente per ogni variazione
                    noise = torch.randn(1, self.config['model']['generator']['noise_dim']) * noise_scale
                    noise = noise.to(self.device)
                    
                    sprite = self.model.generate(desc, noise=noise, device=self.device)
                    sprite_img = Image.fromarray(sprite)
                    
                    # Aggiungi testo alla sprite
                    from PIL import ImageDraw, ImageFont
                    draw = ImageDraw.Draw(sprite_img)
                    try:
                        font = ImageFont.truetype("arial.ttf", 10)
                    except:
                        font = ImageFont.load_default()
                    
                    text = f"{desc[:30]}... (v{i+1})"
                    draw.text((5, 5), text, fill='white', font=font, stroke_width=1, stroke_fill='black')
                    
                    generated_images.append(sprite_img)
            
            status = f"✅ Generated {len(generated_images)} sprites from {len(desc_list)} descriptions"
            return generated_images, status
            
        except Exception as e:
            logger.error(f"Error in batch generation: {e}")
            return [], f"❌ Error: {str(e)}"
    
    def load_evaluation_metrics(self) -> Tuple[Optional[Image.Image], str]:
        """Load and display evaluation metrics"""
        
        try:
            # Carica metriche dall'ultimo checkpoint
            checkpoint_dir = Path(self.config['paths']['checkpoints_dir'])
            checkpoint_files = list(checkpoint_dir.glob('checkpoint_epoch_*.pt'))
            
            if not checkpoint_files:
                return None, "❌ No checkpoints found."
            
            # Ottieni l'ultimo checkpoint
            latest_checkpoint = max(checkpoint_files, key=lambda p: p.stat().st_mtime)
            checkpoint = torch.load(latest_checkpoint, map_location='cpu')
            
            if 'val_metrics' in checkpoint:
                metrics = checkpoint['val_metrics']
                metrics_img = plot_metrics(metrics, f"Validation Metrics - Epoch {checkpoint['epoch']}")
                
                # Formatta le metriche come testo
                metrics_text = f"**Epoch {checkpoint['epoch']} Metrics:**\n"
                for key, value in metrics.items():
                    metrics_text += f"- {key}: {value:.4f}\n"
                
                return metrics_img, metrics_text
            else:
                return None, "❌ No validation metrics found in checkpoint."
                
        except Exception as e:
            logger.error(f"Error loading metrics: {e}")
            return None, f"❌ Error: {str(e)}"
    
    def get_model_info(self) -> str:
        
        
        if not self.model_loaded:
            return "❌ Model not loaded."
        
        try:
            # Conta i parametri del modello
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            info = f"""
            **Model Architecture:**
            - Text Encoder: {self.config['model']['encoder']['model_name']}
            - Hidden Dimension: {self.config['model']['encoder']['hidden_dim']}
            - Generator Base Channels: {self.config['model']['generator']['base_channels']}
            - Output Size: {self.config['model']['generator']['output_size']}x{self.config['model']['generator']['output_size']}
            
            **Parameters:**
            - Total: {total_params:,}
            - Trainable: {trainable_params:,}
            
            **Training Configuration:**
            - Batch Size: {self.config['training']['batch_size']}
            - Learning Rate: {self.config['training']['learning_rate']}
            - Loss Weights:
              - Reconstruction: {self.config['loss']['reconstruction_weight']}
              - Perceptual: {self.config['loss']['perceptual_weight']}
              - Adversarial: {self.config['loss']['adversarial_weight']}
            
            **Device:** {self.device}
            """
            
            return info
            
        except Exception as e:
            return f"❌ Error getting model info: {str(e)}"


def create_interface():
    """Crea l'interfaccia Gradio"""
    
    app = PikaPikaGeneratorApp()
    
    # Stile CSS per l'interfaccia
    css = """
    .gradio-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .header-banner {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 1rem;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
    }
    """
    
    with gr.Blocks(css=css, title="PikaPikaGenerator - Advanced Pokemon Sprite Generation") as demo:
        
        # Header
        gr.HTML("""
        <div class="header-banner">
            <h1> PikaPikaGenerator </h1>
            <h3>Advanced Text-to-Pokemon Sprite Generation</h3>
            <p>Using Encoder-Decoder Architecture with Attention Mechanism</p>
        </div>
        """)
        
        with gr.Tabs():
            
            # Single Generation Tab
            with gr.TabItem(" Generate Sprite"):
                with gr.Row():
                    with gr.Column(scale=2):
                        description_input = gr.Textbox(
                            label="Pokemon Description",
                            placeholder="Enter a detailed Pokemon description...",
                            lines=3
                        )
                        
                        with gr.Row():
                            noise_scale = gr.Slider(
                                minimum=0.1, maximum=2.0, value=1.0, step=0.1,
                                label="Noise Scale (Variation)"
                            )
                            show_attention = gr.Checkbox(
                                label="Show Attention Heatmap",
                                value=False
                            )
                        
                        generate_btn = gr.Button(" Generate Sprite", variant="primary", size="lg")
                    
                    with gr.Column(scale=1):
                        generated_image = gr.Image(label="Generated Sprite", type="pil")
                        attention_heatmap = gr.Image(label="Attention Heatmap", type="pil", visible=False)
                        status_output = gr.Textbox(label="Status", interactive=False)
                
                # Esempi
                gr.Examples(
                    examples=[
                        ["A small yellow electric mouse Pokemon with red cheeks and a lightning bolt shaped tail"],
                        ["A large blue turtle Pokemon with water cannons protruding from its shell"],
                        ["An orange dragon-type Pokemon with wings and a flame on its tail"],
                        ["A pink fairy Pokemon with ribbons and a sweet expression"],
                        ["A dark ghost Pokemon with purple flames and menacing red eyes"],
                        ["A steel-type bird Pokemon with sharp metallic feathers"],
                        ["A grass Pokemon that looks like a walking tree with leaves for hands"],
                        ["An ice-type Pokemon resembling a crystalline wolf with frozen breath"]
                    ],
                    inputs=[description_input],
                    label="Example Descriptions:"
                )
            
            # Batch Generation Tab
            with gr.TabItem(" Batch Generation"):
                with gr.Row():
                    with gr.Column():
                        batch_descriptions = gr.Textbox(
                            label="Pokemon Descriptions (one per line)",
                            placeholder="Enter multiple descriptions, one per line...",
                            lines=8
                        )
                        
                        with gr.Row():
                            num_variations = gr.Slider(
                                minimum=1, maximum=5, value=3, step=1,
                                label="Variations per Description"
                            )
                            batch_noise_scale = gr.Slider(
                                minimum=0.1, maximum=2.0, value=1.0, step=0.1,
                                label="Noise Scale"
                            )
                        
                        batch_generate_btn = gr.Button("🎯 Generate Batch", variant="primary")
                    
                    with gr.Column():
                        batch_gallery = gr.Gallery(
                            label="Generated Sprites",
                            show_label=True,
                            columns=4,
                            rows=2,
                            height="auto"
                        )
                        batch_status = gr.Textbox(label="Status", interactive=False)
            
            # Model Info Tab
            with gr.TabItem(" Model Info & Metrics"):
                with gr.Row():
                    with gr.Column():
                        info_btn = gr.Button(" Load Evaluation Metrics", variant="primary")
                        metrics_plot = gr.Image(label="Metrics Visualization", type="pil")
                        metrics_text = gr.Markdown()
                    
                    with gr.Column():
                        model_info_btn = gr.Button(" Get Model Info", variant="primary")
                        model_info_display = gr.Markdown()
        
        # Footer
        gr.HTML("""
        <div style='text-align: center; margin-top: 2rem; color: #666;'>
            <p>PikaPikaGenerator - Advanced Pokemon Sprite Generation using Deep Learning</p>
            <p>Built with PyTorch, Transformers, and Gradio</p>
        </div>
        """)
        
        # Event handlers
        def update_attention_visibility(show):
            return gr.update(visible=show)
        
        show_attention.change(
            fn=update_attention_visibility,
            inputs=[show_attention],
            outputs=[attention_heatmap]
        )
        
        generate_btn.click(
            fn=app.generate_sprite,
            inputs=[description_input, noise_scale, show_attention],
            outputs=[generated_image, attention_heatmap, status_output]
        )
        
        batch_generate_btn.click(
            fn=app.batch_generate,
            inputs=[batch_descriptions, num_variations, batch_noise_scale],
            outputs=[batch_gallery, batch_status]
        )
        
        info_btn.click(
            fn=app.load_evaluation_metrics,
            outputs=[metrics_plot, metrics_text]
        )
        
        model_info_btn.click(
            fn=app.get_model_info,
            outputs=[model_info_display]
        )
    
    return demo


def main():
    """Avvia l'app Gradio"""
    logger.info("Starting PikaPikaGenerator Gradio App...")
    
    demo = create_interface()
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )


if __name__ == "__main__":
    main()