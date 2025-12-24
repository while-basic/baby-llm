#----------------------------------------------------------------------------
#File:       setup.py
#Project:    Baby LLM - Unified Neural Child Development System
#Created by: Celaya Solutions, 2025
#Author:     Christopher Celaya <chris@chriscelaya.com>
#Description: Setup script for installing the package
#Version:    1.0.0
#License:    MIT
#Last Update: January 2025
#----------------------------------------------------------------------------

"""Setup script for neural-child-unified package.

This is a compatibility script. The package uses pyproject.toml for configuration.
Install with: pip install -e .
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

setup(
    name="neural-child-unified",
    version="1.0.0",
    description="A unified neural child development system modeling child cognitive development",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Celaya Solutions",
    author_email="chris@chriscelaya.com",
    url="https://celayasolutions.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.11",
    install_requires=[
        "torch>=2.0.0",
        "pydantic>=2.0.0",
        "pyyaml>=6.0",
        "numpy>=1.20.0,<2.0.0",
        "flask>=3.0.0",
        "flask-cors>=4.0.0",
        "requests>=2.25.0",
        "python-dotenv>=1.0.0",
        "networkx>=2.8.0",
        "scikit-learn>=1.0.0",
        "chromadb>=0.4.0",
        "ollama>=0.1.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "mypy>=1.4.0",
            "ruff>=0.1.0",
        ],
        "viz": [
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)

