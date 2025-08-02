# Land Cover Segmentation - Technical Documentation

## Project Overview

This project implements a complete semantic segmentation system for land cover mapping using deep learning. The system supports 8 classes of land cover and achieves high-performance classification with transfer learning.

## Architecture

### Model Architectures
- **U-Net**: Classic encoder-decoder with skip connections
  - Supports custom and ResNet backbones (ResNet34, ResNet50)
  - Configurable input channels and bilinear upsampling
- **DeepLabV3**: Atrous convolutions with ASPP module
  - ResNet backbones with dilated convolutions
  - Auxiliary loss support for improved training
- **DeepLabV3+**: Enhanced decoder with low-level features
  - Combines benefits of encoder-decoder and ASPP

### Key Components

#### 1. Data Processing (`src/data/`)
- **Dataset**: Custom land cover dataset with RGB to class mapping
- **Transforms**: Comprehensive augmentation pipeline using Albumentations
- **DataLoader**: Efficient data loading with class balancing support

#### 2. Training System (`src/training/`)
- **Loss Functions**: Cross-entropy, Dice, Focal, IoU, Tversky, and combined losses
- **Optimizers**: Adam, AdamW, SGD, RMSprop with configurable parameters
- **Schedulers**: Cosine, polynomial, warmup, plateau, and custom schedulers

#### 3. Evaluation (`src/evaluation/`)
- **Metrics**: IoU, mIoU, pixel accuracy, F1-score, precision, recall, Dice coefficient
- **Evaluator**: Comprehensive evaluation with timing and visualization
- **Benchmarking**: Performance profiling and inference speed measurement

#### 4. Utilities (`src/utils/`)
- **Configuration**: YAML-based configuration management
- **Device**: Automatic device detection and management
- **Reproducibility**: Deterministic training with seed management
- **Visualization**: Prediction visualization and training history plots

## Land Cover Classes

| ID | Class | Color | Expected Distribution |
|----|--------|-------|---------------------|
| 0 | Bareland | #800000 | 1.5% |
| 1 | Rangeland | #00FF24 | 22.9% |
| 2 | Developed space | #949494 | 16.1% |
| 3 | Road | #FFFFFF | 6.7% |
| 4 | Tree | #226126 | 20.2% |
| 5 | Water | #0045FF | 3.3% |
| 6 | Agriculture land | #4BB549 | 13.7% |
| 7 | Building | #DE1F07 | 15.6% |

## Performance Targets

- **mIoU**: > 75% on validation set
- **Inference Time**: < 2 seconds per image on GPU
- **Model Size**: Optimized for deployment
- **Memory Usage**: Efficient GPU utilization

## Installation & Setup

### Requirements
- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU training)

### Quick Setup
```bash
# Clone repository
git clone https://github.com/Sanemi-A/CALIDADSOFTWARE.git
cd CALIDADSOFTWARE

# Setup development environment
python scripts/setup_dev.py --setup

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training
```bash
# Train U-Net model
python scripts/train.py --config config/unet_config.yaml

# Train DeepLabV3 model
python scripts/train.py --config config/deeplabv3_config.yaml

# Resume training from checkpoint
python scripts/train.py --config config/unet_config.yaml --resume models/checkpoint_epoch_50.pth
```

### Evaluation
```bash
# Evaluate model
python scripts/evaluate.py --config config/unet_config.yaml --model-path models/best_model.pth

# Evaluate with visualization
python scripts/evaluate.py --config config/unet_config.yaml --model-path models/best_model.pth --visualize

# Run performance benchmark
python scripts/evaluate.py --config config/unet_config.yaml --model-path models/best_model.pth --benchmark
```

### Prediction
```bash
# Single image prediction
python scripts/predict.py --image-path path/to/image.jpg --model-path models/best_model.pth --config config/unet_config.yaml

# Prediction with visualization
python scripts/predict.py --image-path path/to/image.jpg --model-path models/best_model.pth --config config/unet_config.yaml --visualize

# Prediction with analysis
python scripts/predict.py --image-path path/to/image.jpg --model-path models/best_model.pth --config config/unet_config.yaml --analyze
```

## Configuration

### Model Configuration
```yaml
model:
  name: "unet"              # Model architecture
  backbone: "resnet34"      # Backbone network
  pretrained: true          # Use pretrained weights
  num_classes: 8            # Number of output classes
  input_channels: 3         # Input image channels
```

### Training Configuration
```yaml
training:
  batch_size: 16           # Batch size
  epochs: 100              # Number of epochs
  learning_rate: 0.001     # Initial learning rate
  optimizer: "adam"        # Optimizer type
  weight_decay: 0.0001     # L2 regularization
```

### Data Configuration
```yaml
data:
  image_size: [512, 512]   # Input image size
  train_split: 0.7         # Training data ratio
  val_split: 0.2           # Validation data ratio
  test_split: 0.1          # Test data ratio
  num_workers: 4           # DataLoader workers
```

## Development

### Code Quality
- **Formatting**: Black with 100-character line length
- **Linting**: Flake8 with project-specific rules
- **Type Checking**: MyPy for static type analysis
- **Testing**: Pytest with coverage reporting

### Running Tests
```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src --cov-report=html

# Run specific test module
pytest tests/test_models.py -v
```

### Code Quality Checks
```bash
# Format code
black src/ scripts/ tests/

# Run linter
flake8 src/ scripts/ tests/ --max-line-length=100

# Type checking
mypy src/ --ignore-missing-imports

# Security checks
bandit -r src/
safety check
```

## Project Structure

```
├── src/                    # Source code
│   ├── models/            # Model architectures
│   │   ├── __init__.py
│   │   ├── unet.py
│   │   ├── deeplabv3.py
│   │   └── factory.py
│   ├── data/              # Data processing
│   │   ├── __init__.py
│   │   ├── dataset.py
│   │   ├── transforms.py
│   │   └── dataloader.py
│   ├── training/          # Training utilities
│   │   ├── __init__.py
│   │   ├── losses.py
│   │   └── optimizers.py
│   ├── evaluation/        # Evaluation metrics
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   └── evaluator.py
│   └── utils/             # Utility functions
│       ├── __init__.py
│       ├── config.py
│       ├── device.py
│       ├── reproducibility.py
│       └── visualization.py
├── config/                # Configuration files
│   ├── unet_config.yaml
│   └── deeplabv3_config.yaml
├── scripts/               # Training/evaluation scripts
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── setup_dev.py
│   └── pre_commit_hooks.py
├── tests/                 # Unit tests
│   ├── __init__.py
│   ├── test_config.py
│   ├── test_models.py
│   ├── test_metrics.py
│   └── test_data.py
├── notebooks/             # Jupyter notebooks
│   └── 01_data_exploration.ipynb
├── docs/                  # Documentation
├── requirements.txt       # Python dependencies
├── pytest.ini           # Test configuration
└── README.md             # Project documentation
```

## Monitoring & Logging

### Training Monitoring
- **TensorBoard**: Real-time training visualization
- **Weights & Biases**: Experiment tracking (optional)
- **Logging**: Comprehensive logging with configurable levels

### Metrics Tracking
- **Loss**: Training and validation loss curves
- **mIoU**: Mean Intersection over Union tracking
- **Per-class IoU**: Individual class performance
- **Learning Rate**: Learning rate scheduling visualization

## Deployment

### Model Export
```python
# Export trained model
torch.save({
    'model_state_dict': model.state_dict(),
    'config': config,
    'class_names': class_names,
    'class_colors': class_colors
}, 'deployed_model.pth')
```

### Inference Optimization
- **Mixed Precision**: Automatic mixed precision training
- **Model Compilation**: PyTorch 2.0 compilation support
- **TensorRT**: NVIDIA TensorRT optimization (when available)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with proper tests
4. Run quality checks: `python scripts/pre_commit_hooks.py`
5. Submit a pull request

## License

This project is licensed under the 0BSD License - see the LICENSE file for details.

## Support

For questions or issues:
1. Check the documentation
2. Run the development setup: `python scripts/setup_dev.py --all`
3. Create an issue on GitHub with detailed information