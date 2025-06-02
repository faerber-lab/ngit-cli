#!/bin/bash

echo "🚀 Installing Working CPU Optimizations Only"
echo "============================================="

# Update pip
pip install --upgrade pip

echo "📦 Installing core dependencies..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

echo "📦 Installing transformers..."
pip install transformers

echo "📦 Installing psutil for memory monitoring..."
pip install psutil

echo "📦 Installing Intel Extension for PyTorch (optional but recommended)..."
pip install intel_extension_for_pytorch || echo "⚠ Intel Extension installation failed, continuing..."

echo "✅ Core installation complete!"
echo ""
echo "🔧 Set these environment variables for best performance:"
echo "export OMP_NUM_THREADS=32"
echo "export MKL_NUM_THREADS=32"
echo "export KMP_AFFINITY=granularity=fine,compact,1,0"
echo ""
echo "🚀 Run with NUMA binding for maximum performance:"
echo "numactl --cpunodebind=0 --membind=0 python your_script.py"