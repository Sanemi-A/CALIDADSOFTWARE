# Semantic Segmentation for Land Cover Mapping

A comprehensive deep learning system for semantic segmentation of satellite imagery to classify land cover into 8 distinct classes using transfer learning.

## Project Overview

This project implements a state-of-the-art semantic segmentation system for global land cover mapping using satellite imagery. The system utilizes transfer learning with pre-trained architectures to achieve high-precision classification across 8 land cover classes.

### Land Cover Classes

| Class | Color Code | Distribution | Description |
|-------|------------|--------------|-------------|
| Bareland | #800000 | 1.5% | Exposed soil and rock surfaces |
| Rangeland | #00FF24 | 22.9% | Natural grasslands and shrublands |
| Developed space | #949494 | 16.1% | Urban and developed areas |
| Road | #FFFFFF | 6.7% | Transportation infrastructure |
| Tree | #226126 | 20.2% | Forest and tree coverage |
| Water | #0045FF | 3.3% | Water bodies and aquatic areas |
| Agriculture land | #4BB549 | 13.7% | Cultivated agricultural areas |
| Building | #DE1F07 | 15.6% | Residential and commercial structures |

## Features

- **Transfer Learning**: Utilizes pre-trained U-Net and DeepLab architectures
- **High Performance**: Achieves >75% mIoU on validation data
- **Fast Inference**: <2 seconds per image on GPU
- **Robust Pipeline**: Complete data processing and augmentation pipeline
- **Comprehensive Evaluation**: Multiple metrics including IoU, F1-score, and confusion matrices
- **Modular Design**: Clean, maintainable, and extensible codebase
- **Quality Assurance**: Comprehensive testing and code quality standards

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Sanemi-A/CALIDADSOFTWARE.git
cd CALIDADSOFTWARE
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Training a Model

```bash
python scripts/train.py --config config/unet_config.yaml
```

### Evaluating a Model

```bash
python scripts/evaluate.py --model-path models/best_model.pth --config config/unet_config.yaml
```

### Making Predictions

```bash
python scripts/predict.py --image-path path/to/image.jpg --model-path models/best_model.pth
```

## Project Structure

```
├── src/                    # Source code
│   ├── models/            # Model architectures
│   ├── data/              # Data loading and processing
│   ├── training/          # Training utilities
│   ├── evaluation/        # Evaluation metrics and tools
│   └── utils/             # General utilities
├── config/                # Configuration files
├── scripts/               # Training and evaluation scripts
├── notebooks/             # Jupyter notebooks for analysis
├── tests/                 # Unit tests
├── docs/                  # Documentation
└── requirements.txt       # Python dependencies
```

## Configuration

The system uses YAML configuration files to manage hyperparameters, model settings, and training configurations. Example configurations are provided in the `config/` directory.

### Key Configuration Options

- **Model Architecture**: U-Net, DeepLab, or custom architectures
- **Training Parameters**: Learning rate, batch size, epochs
- **Data Augmentation**: Rotation, flipping, color transforms
- **Optimization**: Adam, SGD, learning rate scheduling

## Evaluation Metrics

The system provides comprehensive evaluation metrics:

- **IoU (Intersection over Union)** per class
- **mIoU (mean IoU)** global performance
- **Pixel Accuracy** overall classification accuracy
- **F1-Score** per class performance
- **Confusion Matrix** detailed classification analysis

## Performance Targets

- **mIoU**: >75% on validation set
- **Inference Time**: <2 seconds per image on GPU
- **Memory Usage**: Optimized for standard GPU memory
- **Reproducibility**: Deterministic results with fixed seeds

## Development

### Code Quality

The project follows strict code quality standards:

- **PEP 8** compliance for Python style
- **Type hints** for better code documentation
- **Unit tests** with pytest framework
- **Pre-commit hooks** for automated checks

### Testing

Run the test suite:

```bash
pytest tests/ -v --cov=src/
```

### Code Formatting

Format code with Black:

```bash
black src/ scripts/ tests/
```

Check code style:

```bash
flake8 src/ scripts/ tests/
mypy src/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## License

This project is licensed under the 0BSD License - see the LICENSE file for details.

## Acknowledgments

- Dataset source: Kaggle Land Cover Classification
- Pre-trained models: PyTorch and TensorFlow model repositories
- Inspiration: Recent advances in semantic segmentation research

## Contact

For questions or support, please open an issue on GitHub.