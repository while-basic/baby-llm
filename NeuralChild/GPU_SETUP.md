# GPU Setup for NeuralChild

## Quick Installation

Run the installation script to install GPU-enabled PyTorch:

```bash
cd NeuralChild
./install_gpu_pytorch.sh
```

Or manually:

```bash
# Uninstall CPU version
python3 -m pip uninstall torch torchvision torchaudio -y

# Install CUDA-enabled PyTorch (CUDA 12.1)
python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Verify GPU Availability

```bash
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

## GPU Priority Implementation

NeuralChild automatically prioritizes GPU usage:

1. **Base Neural Network Class**: All networks detect and use GPU if available
   - Location: `neuralchild/core/neural_network.py`
   - Networks automatically move to GPU device during initialization
   - Logs GPU device name on initialization

2. **Tensor Operations**: All tensors follow the network's device
   - Input tensors automatically use the network's device
   - Internal tensor creation uses `device=x.device` or `self.device`

3. **Network Implementations**: All four networks support GPU:
   - ConsciousnessNetwork
   - EmotionsNetwork
   - PerceptionNetwork
   - ThoughtsNetwork

## Performance Notes

- **GPU**: Significantly faster for neural network operations, especially with larger models
- **CPU**: Falls back automatically if GPU is not available
- **Mixed Precision**: Can be added for even better performance on modern GPUs

## Troubleshooting

### GPU not detected
- Ensure NVIDIA drivers are installed: `nvidia-smi`
- Verify CUDA toolkit is available
- Check PyTorch CUDA version matches your CUDA installation

### Out of Memory
- Reduce network dimensions in `config.yaml`
- Use smaller batch sizes
- Enable gradient checkpointing (future enhancement)

## Testing

Run a test simulation to verify GPU usage:

```bash
python3 neuralchild/cli.py run --steps 10
```

Check the logs for "using GPU" messages during network initialization.

