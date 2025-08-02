"""Pre-commit hooks configuration script."""

import subprocess
import sys
from pathlib import Path


def run_black():
    """Run Black code formatter."""
    print("Running Black formatter...")
    result = subprocess.run([
        'black', 'src/', 'scripts/', 'tests/', '--line-length', '100'
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        print("Black formatting failed:")
        print(result.stdout)
        print(result.stderr)
        return False
    
    print("Black formatting passed")
    return True


def run_flake8():
    """Run Flake8 linter."""
    print("Running Flake8 linter...")
    result = subprocess.run([
        'flake8', 'src/', 'scripts/', 'tests/', 
        '--max-line-length=100', '--ignore=E203,W503'
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        print("Flake8 linting failed:")
        print(result.stdout)
        print(result.stderr)
        return False
    
    print("Flake8 linting passed")
    return True


def run_mypy():
    """Run MyPy type checker."""
    print("Running MyPy type checker...")
    result = subprocess.run([
        'mypy', 'src/', '--ignore-missing-imports'
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        print("MyPy type checking failed:")
        print(result.stdout)
        print(result.stderr)
        return False
    
    print("MyPy type checking passed")
    return True


def run_tests():
    """Run unit tests."""
    print("Running unit tests...")
    result = subprocess.run([
        'pytest', 'tests/', '-v', '--tb=short'
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        print("Unit tests failed:")
        print(result.stdout)
        print(result.stderr)
        return False
    
    print("Unit tests passed")
    return True


def main():
    """Main pre-commit hook."""
    print("Running pre-commit hooks...")
    
    hooks = [
        run_black,
        run_flake8,
        run_mypy,
        run_tests
    ]
    
    failed_hooks = []
    
    for hook in hooks:
        try:
            if not hook():
                failed_hooks.append(hook.__name__)
        except Exception as e:
            print(f"Error running {hook.__name__}: {e}")
            failed_hooks.append(hook.__name__)
    
    if failed_hooks:
        print(f"\nPre-commit hooks failed: {', '.join(failed_hooks)}")
        print("Please fix the issues before committing.")
        sys.exit(1)
    else:
        print("\nAll pre-commit hooks passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()