FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-devel

WORKDIR /workspace/finetune-zamba2

# Install git and other essentials
RUN apt-get update && apt-get install -y \
    git \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    transformers \
    accelerate \
    datasets \
    bitsandbytes \
    optuna \
    scikit-learn \
    matplotlib \
    scipy \
    huggingface-hub \
    torch-distributed

# Set environment variables for better GPU utilization
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

# Enable multi-GPU support
ENV NCCL_DEBUG=INFO
ENV NCCL_SOCKET_IFNAME=eth0
ENV NCCL_IB_DISABLE=1
ENV MASTER_ADDR=localhost
ENV MASTER_PORT=29500

CMD ["python", "-m", "torch.distributed.launch", "--nproc_per_node=auto", "finetune-zamba2/finetune.py"]