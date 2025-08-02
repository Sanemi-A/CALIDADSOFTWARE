#!/usr/bin/env python3
"""
Development setup script for the land cover segmentation project.

This script helps set up the development environment and run basic checks.
"""

import subprocess
import sys
import os
from pathlib import Path
import argparse


def run_command(command, description, check=True):
    """Run a command and handle errors."""
    print(f"\n{description}...")
    try:
        result = subprocess.run(command, shell=True, check=check, 
                              capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        if e.stdout:
            print(f"stdout: {e.stdout}")
        if e.stderr:
            print(f"stderr: {e.stderr}")
        return False


def setup_environment():
    """Set up the development environment."""
    print("Setting up development environment...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher is required")
        return False
    
    print(f"Python version: {sys.version}")
    
    # Install dependencies
    commands = [
        ("pip install --upgrade pip", "Upgrading pip"),
        ("pip install -r requirements.txt", "Installing main dependencies"),
        ("pip install pytest pytest-cov black flake8 mypy bandit safety", "Installing development tools")
    ]
    
    for cmd, desc in commands:
        if not run_command(cmd, desc):
            print(f"Failed to execute: {cmd}")
            return False
    
    print("\n✅ Environment setup completed successfully!")
    return True


def run_tests():
    """Run the test suite."""
    print("\nRunning test suite...")
    
    # Create test directories if they don't exist
    test_dirs = ['logs', 'models', 'outputs']
    for dir_name in test_dirs:
        Path(dir_name).mkdir(exist_ok=True)
    
    return run_command("python -m pytest tests/ -v", "Running unit tests")


def run_linting():
    """Run code quality checks."""
    print("\nRunning code quality checks...")
    
    checks = [
        ("python -m black --check src/ scripts/ tests/", "Checking code formatting"),
        ("python -m flake8 src/ scripts/ tests/ --max-line-length=100 --ignore=E203,W503", "Running linter"),
        ("python -m mypy src/ --ignore-missing-imports", "Running type checker")
    ]
    
    all_passed = True
    for cmd, desc in checks:
        if not run_command(cmd, desc, check=False):
            all_passed = False
    
    return all_passed


def run_security_checks():
    """Run security checks."""
    print("\nRunning security checks...")
    
    checks = [
        ("python -m safety check", "Checking for security vulnerabilities"),
        ("python -m bandit -r src/ -f json", "Running security linter")
    ]
    
    all_passed = True
    for cmd, desc in checks:
        if not run_command(cmd, desc, check=False):
            all_passed = False
    
    return all_passed


def test_model_creation():
    """Test basic model creation."""
    print("\nTesting model creation...")
    
    test_script = '''
import sys
from pathlib import Path
sys.path.append(".")

try:
    from src.models import create_model
    from src.utils import load_config
    
    # Test U-Net creation
    config = {
        "name": "unet",
        "num_classes": 8,
        "backbone": "custom",
        "input_channels": 3
    }
    
    model = create_model(config)
    print(f"✅ U-Net model created successfully: {model.__class__.__name__}")
    
    # Test configuration loading
    config = load_config("config/unet_config.yaml")
    print(f"✅ Configuration loaded successfully")
    
    print("✅ All basic functionality tests passed!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)
'''
    
    return run_command(f'python -c "{test_script}"', "Testing basic functionality")


def create_sample_data():
    """Create sample data for testing."""
    print("\nCreating sample data structure...")
    
    # Create data directories
    data_dirs = [
        'data/train/images',
        'data/train/masks',
        'data/val/images', 
        'data/val/masks',
        'data/test/images',
        'data/test/masks'
    ]
    
    for dir_path in data_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    print("✅ Sample data structure created")
    return True


def generate_docs():
    """Generate basic documentation."""
    print("\nGenerating documentation...")
    
    docs_content = """
# Land Cover Segmentation Documentation

## Project Structure

```
├── src/                 # Source code
│   ├── models/         # Model architectures
│   ├── data/           # Data processing
│   ├── training/       # Training utilities
│   ├── evaluation/     # Evaluation metrics
│   └── utils/          # Utility functions
├── config/             # Configuration files
├── scripts/            # Training/evaluation scripts
├── tests/              # Unit tests
├── notebooks/          # Jupyter notebooks
└── docs/               # Documentation
```

## Quick Start

1. **Setup Environment:**
   ```bash
   python scripts/setup_dev.py --setup
   ```

2. **Train a Model:**
   ```bash
   python scripts/train.py --config config/unet_config.yaml
   ```

3. **Evaluate Model:**
   ```bash
   python scripts/evaluate.py --config config/unet_config.yaml --model-path models/best_model.pth
   ```

4. **Make Predictions:**
   ```bash
   python scripts/predict.py --image-path path/to/image.jpg --model-path models/best_model.pth --config config/unet_config.yaml
   ```

## Model Architectures

- **U-Net**: Classic encoder-decoder with skip connections
- **DeepLabV3**: Atrous convolutions with ASPP module
- **DeepLabV3+**: Enhanced decoder with low-level features

## Performance Targets

- mIoU > 75% on validation set
- Inference time < 2 seconds per image on GPU
- Model size optimized for deployment
"""
    
    docs_dir = Path('docs')
    docs_dir.mkdir(exist_ok=True)
    
    with open(docs_dir / 'README.md', 'w') as f:
        f.write(docs_content)
    
    print("✅ Basic documentation generated")
    return True


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Development setup script")
    parser.add_argument("--setup", action="store_true", help="Set up development environment")
    parser.add_argument("--test", action="store_true", help="Run tests")
    parser.add_argument("--lint", action="store_true", help="Run linting")
    parser.add_argument("--security", action="store_true", help="Run security checks")
    parser.add_argument("--all", action="store_true", help="Run all checks")
    parser.add_argument("--create-sample-data", action="store_true", help="Create sample data structure")
    parser.add_argument("--docs", action="store_true", help="Generate documentation")
    
    args = parser.parse_args()
    
    if not any(vars(args).values()):
        # Default behavior: run basic setup and tests
        args.setup = True
        args.test = True
        args.create_sample_data = True
    
    success = True
    
    if args.setup or args.all:
        success &= setup_environment()
    
    if args.create_sample_data or args.all:
        success &= create_sample_data()
    
    if args.test or args.all:
        success &= test_model_creation()
        success &= run_tests()
    
    if args.lint or args.all:
        success &= run_linting()
    
    if args.security or args.all:
        success &= run_security_checks()
    
    if args.docs or args.all:
        success &= generate_docs()
    
    if success:
        print("\n🎉 All development setup tasks completed successfully!")
        print("\nNext steps:")
        print("1. Add your training data to the data/ directory")
        print("2. Train a model: python scripts/train.py --config config/unet_config.yaml")
        print("3. Evaluate the model: python scripts/evaluate.py --config config/unet_config.yaml --model-path models/best_model.pth")
    else:
        print("\n❌ Some tasks failed. Please check the output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()