# PikaPikaGenerator: Generazione Text-to-Pokémon Sprite

> Sistema avanzato di deep learning per la generazione di sprite Pokémon a partire da descrizioni testuali utilizzando un'architettura Encoder-Decoder all'avanguardia con meccanismo di Attenzione.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-v2.0+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

# Indice
- [Panoramica](#panoramica)
- [Architettura](#architettura)
- [Caratteristiche](#caratteristiche)
- [Installazione](#installazione)
- [Dataset](#dataset)
- [Utilizzo](#utilizzo)
- [Configurazione](#configurazione)
- [Training](#training)
- [Valutazione](#valutazione)
- [Interfaccia Demo](#interfaccia-demo)
- [Risultati](#risultati)
- [Struttura del Progetto](#struttura-del-progetto)
- [Risoluzione Problemi](#risoluzione-problemi)
- [Crediti](#crediti)

# Panoramica

PikaPikaGenerator è un progetto avanzato di deep learning sviluppato per il corso di Deep Learning al Politecnico di Bari (Codice Test: 2025_VI). Il sistema genera sprite Pokémon di alta qualità a partire da descrizioni testuali utilizzando una sofisticata architettura Encoder-Decoder con meccanismo di attenzione multi-head.

# Obiettivi Principali
- Implementare un encoder testuale basato su Transformer utilizzando BERT-mini pre-addestrato
- Sviluppare un decoder basato su CNN per la generazione progressiva di sprite
- Integrare meccanismi di attenzione per l'allineamento testo-immagine
- Creare una demo interattiva Gradio per la generazione in tempo reale

# Architettura

Il sistema è composto da tre componenti principali:

# 1. **Text Encoder** (basato su BERT)
- BERT pre-addestrato `prajjwal1/bert-mini` con embedding a 256 dimensioni
- 6 layer di attenzione con 12 teste di attenzione
- Layer di proiezione per l'adattamento dimensionale
- Regolarizzazione Dropout (0.1)

# 2. **Modulo di Attenzione**
- Meccanismo di attenzione multi-head a 8 teste
- Cross-attention tra caratteristiche testuali e visive
- Visualizzazione dei pesi di attenzione per interpretabilità

# 3. **Generatore di Sprite** (Decoder CNN)
- Upsampling progressivo da 4×4 a 256×256
- 6 blocchi residuali con normalizzazione spettrale
- Self-attention alle risoluzioni [32, 64]
- Pooling adattivo per dimensione output esatta

# Componenti Opzionali
- **Discriminatore**: Per training adversariale (architettura progressiva)
- **Loss Percettuale**: Utilizzo di LPIPS per migliore qualità visiva

# Caratteristiche

- **Generazione Text-to-Image**: Genera sprite Pokémon da descrizioni in linguaggio naturale
- **Supporto Multi-Risoluzione**: Genera sprite a 128, 192, 256 e 320 pixel
- **Training Progressivo**: Incrementa gradualmente la risoluzione durante il training
- **Visualizzazione Attenzione**: Visualizza su quali parole il modello si concentra
- **Metriche Complete**: Valutazione FID, IS, SSIM, PSNR, LPIPS
- **Demo Interattiva**: Interfaccia web basata su Gradio con generazione batch
- **Gestione Errori Robusta**: Gestione elegante di casi limite ed errori
- **Auto-Resume Training**: Riprende automaticamente dall'ultimo checkpoint


# Setup

# Creare Ambiente Virtuale
python -m venv venv
source venv/bin/activate  # Su Windows: venv\Scripts\activate
# Installare Dipendenze
pip install -r requirements.txt


# Utilizzo
# Avvio Rapido
python main.py all
# Esecuzione passo passo
# Preprocessing
python preprocessing.py
# Training
python train.py
# Riprendi da Checkpoint
python train.py --resume
# Valutazione del modello
python evaluate.py
# Demo Interattiva
python demo.py
oppure 
python app.py