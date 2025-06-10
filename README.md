# 🎭 Multimodal Video Emotion Recognition

A comprehensive deep learning framework for emotion recognition from video data, featuring 4 different model architectures optimized for RTX 4080 16GB.

## 🔥 Key Features

- **4 Model Architectures**: CNN, ViT from scratch, Pretrained ResNet50, Pretrained ViT
- **Professional Dataset**: GENEX Interview gaze replay videos with AFFDEX emotion annotations
- **RTX 4080 Optimized**: Mixed precision training with optimal batch sizes
- **Comprehensive Evaluation**: Statistical validation, attention visualization, performance comparison
- **Production Ready**: Complete training pipeline with experiment tracking

## 📊 Dataset Overview

- **5 Participants** with gaze replay videos and frame-level emotion annotations
- **140+ Facial Metrics** from professional AFFDEX facial coding system
- **7 Core Emotions**: Joy, Anger, Fear, Disgust, Sadness, Surprise, Contempt
- **Attention/Engagement** metrics for behavioral analysis
- **Temporal Alignment** between video frames and emotion labels

## 🏗️ Architecture

```
multimodal_video_ml/
├── data/                    # Data processing and storage
├── src/
│   ├── data/               # Dataset classes and preprocessing
│   ├── models/             # 4 model architectures
│   ├── training/           # Training loops and evaluation
│   └── utils/              # Configuration and utilities
├── scripts/                # Main execution scripts
├── experiments/            # Model checkpoints and logs
└── notebooks/              # Analysis notebooks
```

## 🚀 Quick Start

### 1. Installation

```bash
git clone [repository]
cd multimodal_video_ml
pip install -r requirements.txt
```

### 2. Data Preparation

```bash
python scripts/prepare_data.py
```

This will:
- Extract frames from videos at 30 FPS
- Detect and crop faces using MTCNN
- Parse emotion annotations from CSV files
- Create train/validation/test splits
- Generate processed datasets

### 3. Train Models

Train all 4 architectures:

```bash
# A.1: Simple CNN (Baseline)
python scripts/train_model.py --model cnn_simple

# A.2: ViT from Scratch
python scripts/train_model.py --model vit_scratch

# A.3: Pretrained ResNet50
python scripts/train_model.py --model resnet_pretrained

# A.4: Pretrained ViT (SOTA)
python scripts/train_model.py --model vit_pretrained
```

### 4. Comprehensive Evaluation

```bash
python scripts/evaluate_all_models.py
```

This generates:
- Performance comparison charts
- Statistical analysis reports
- Model efficiency metrics
- Deployment recommendations

## 🎯 Model Architectures

### A.1: Simple CNN (Baseline)
- **Purpose**: Establish baseline performance
- **Architecture**: 4 Conv blocks + Global Average Pooling
- **Parameters**: ~2M trainable
- **Batch Size**: 64 (RTX 4080 optimized)
- **Expected Performance**: 65-70% accuracy

### A.2: Vision Transformer (Scratch)
- **Purpose**: Modern attention-based approach
- **Architecture**: 6-layer transformer with patch embedding
- **Parameters**: ~12M trainable
- **Batch Size**: 32 (RTX 4080 optimized)
- **Expected Performance**: 70-75% accuracy

### A.3: Pretrained ResNet50
- **Purpose**: Transfer learning baseline
- **Architecture**: ImageNet pretrained + custom head
- **Parameters**: ~23.5M total, ~2M trainable
- **Batch Size**: 48 (RTX 4080 optimized)
- **Expected Performance**: 75-80% accuracy

### A.4: Pretrained ViT (SOTA) ⭐
- **Purpose**: State-of-the-art performance
- **Architecture**: ViT-Base with frozen backbone
- **Parameters**: ~86M total, ~5M trainable
- **Batch Size**: 24 (RTX 4080 optimized)
- **Expected Performance**: 80-85% accuracy

## 🔧 RTX 4080 16GB Optimizations

### Memory Management
- **Mixed Precision Training**: 2x speedup + 40% memory savings
- **Gradient Accumulation**: Effective larger batch sizes
- **Optimized Batch Sizes**: Model-specific tuning for 16GB VRAM
- **Smart Caching**: Efficient data loading with prefetching

### Performance Features
- **CUDA Optimizations**: Full GPU utilization
- **Dynamic Loss Scaling**: Stable mixed precision training
- **Gradient Clipping**: Prevents exploding gradients
- **Learning Rate Scheduling**: Cosine annealing with warm restarts

## 📈 Training Configuration

```yaml
# Optimized for RTX 4080 16GB
mixed_precision: true
gradient_accumulation_steps: 2-4
num_workers: 8
pin_memory: true

batch_sizes:
  cnn_simple: 64        # ~8GB usage
  vit_scratch: 32       # ~12GB usage
  resnet50: 48          # ~10GB usage
  vit_pretrained: 24    # ~14GB usage
```

## 📊 Evaluation Metrics

### Classification Metrics
- **Accuracy**: Overall correctness
- **F1-Score**: Balanced precision/recall
- **AUC-ROC**: Ranking quality
- **Balanced Accuracy**: Class-imbalance adjusted

### Advanced Analysis
- **Confusion Matrices**: Error pattern analysis
- **Attention Visualization**: Model interpretability
- **Calibration Metrics**: Confidence reliability
- **Cross-Participant Validation**: Generalization assessment

## 🎨 Visualization Features

### Attention Maps (ViT Models)
- **Spatial Attention**: Which image regions drive decisions
- **Layer-wise Analysis**: How attention evolves through network
- **Head Diversity**: Different attention patterns per head

### Performance Dashboards
- **Training Curves**: Loss and metric evolution
- **Model Comparison**: Side-by-side architecture performance
- **Error Analysis**: Failure case identification

## 🔬 Research Applications

### Academic Use Cases
- **Emotion Recognition Research**: Benchmark different architectures
- **Attention Mechanism Studies**: Visualize learned patterns
- **Transfer Learning Analysis**: Compare pretrained vs. scratch training
- **Multimodal Fusion**: Extend to audio/text modalities

### Industry Applications
- **Interview Assessment**: Automated candidate evaluation
- **Customer Experience**: Emotion tracking in user studies
- **Healthcare**: Mental health monitoring
- **Education**: Student engagement analysis

## 📁 Output Structure

```
experiments/
├── model_comparison/           # Cross-model analysis
│   ├── model_comparison.png    # Performance charts
│   ├── detailed_analysis.json  # Statistical insights
│   └── [model]_results.json    # Individual results
├── [model_name]/              # Per-model experiments
│   ├── checkpoints/           # Model weights
│   ├── logs/                  # Training logs
│   └── visualizations/        # Attention maps
```

## 🔍 Advanced Usage

### Custom Training Tasks

```bash
# Regression task (continuous attention scores)
python scripts/train_model.py --model vit_pretrained --task attention_regression

# Multi-label emotion classification
python scripts/train_model.py --model resnet_pretrained --task emotion_multilabel

# Custom hyperparameters
python scripts/train_model.py --model cnn_simple --batch_size 32 --learning_rate 1e-3 --epochs 50
```

### Resume Training

```bash
python scripts/train_model.py --model vit_pretrained --resume experiments/vit_pretrained/checkpoints/best_model.pth
```

### Evaluation Only

```bash
python scripts/train_model.py --model resnet_pretrained --eval_only --resume experiments/resnet_pretrained/checkpoints/best_model.pth
```

## 🐛 Troubleshooting

### Common Issues

**CUDA Out of Memory**
```bash
# Reduce batch size
python scripts/train_model.py --model vit_pretrained --batch_size 16
```

**Data Not Found**
```bash
# Ensure data preparation completed
python scripts/prepare_data.py
```

**Missing Dependencies**
```bash
# Install all requirements
pip install -r requirements.txt

# For pretrained ViT models
pip install timm
```

### Performance Optimization

**Faster Training**
- Enable mixed precision (default)
- Increase num_workers (default: 8)
- Use SSD storage for datasets
- Close unnecessary applications

**Better Accuracy**
- Increase training epochs
- Add data augmentation
- Tune learning rate schedules
- Ensemble multiple models

## 📖 Citation

If you use this framework in your research, please cite:

```bibtex
@software{multimodal_emotion_recognition,
  title={Multimodal Video Emotion Recognition Framework},
  author={[Your Name]},
  year={2024},
  url={[Repository URL]}
}
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📞 Support

- **Issues**: GitHub Issues for bug reports
- **Discussions**: GitHub Discussions for questions
- **Email**: [Your Email] for direct support

## 🙏 Acknowledgments

- **AFFDEX**: Professional facial coding system
- **GENEX Dataset**: Interview video data
- **PyTorch Team**: Deep learning framework
- **Hugging Face**: Pretrained model ecosystem
- **Weights & Biases**: Experiment tracking platform

---

**⚡ Optimized for RTX 4080 16GB | 🎯 Production Ready | 📊 Research Grade**