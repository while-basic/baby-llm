#!/bin/bash
#----------------------------------------------------------------------------
#File:       install_gpu_pytorch.sh
#Project:     NeuralChild
#Created by:  Celaya Solutions, 2025
#Author:      Christopher Celaya <chris@chriscelaya.com>
#Description: Install GPU-enabled PyTorch for NeuralChild
#Version:     1.0.0
#License:     MIT
#Last Update: November 2025
#----------------------------------------------------------------------------

echo "🔧 Installing GPU-enabled PyTorch for NeuralChild..."
echo ""

# Uninstall CPU version
echo "1. Uninstalling CPU-only PyTorch..."
python3 -m pip uninstall torch torchvision torchaudio -y

# Install CUDA-enabled PyTorch
# Using CUDA 12.1 (compatible with CUDA 12.x)
echo ""
echo "2. Installing PyTorch with CUDA 12.1 support..."
python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify installation
echo ""
echo "3. Verifying GPU availability..."
python3 << EOF
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
    print(f"GPU count: {torch.cuda.device_count()}")
else:
    print("⚠️  GPU not available - falling back to CPU")
EOF

echo ""
echo "✅ Installation complete!"
echo ""
echo "To test NeuralChild with GPU:"
echo "  python3 neuralchild/cli.py run --steps 10"

