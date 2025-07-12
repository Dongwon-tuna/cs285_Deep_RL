#  CartPole Stochastic Policy (PyTorch + Gym)

This project implements a stochastic policy using Categorical distributions to solve the classic CartPole control problem using PyTorch.

---

##  Environment Setup (with Conda)

We recommend using a Conda environment to manage dependencies and ensure compatibility.

### 1️⃣ Create and activate a new Conda environment

```bash
conda create -n cartpole_env python=3.10
conda activate cartpole_env
pip install -r requirements.txt

```

⚡ **Optional: Upgrade PyTorch to CUDA version (for GPU users)**

> ℹ️ This project was developed using **NVIDIA RTX 3090** with **CUDA 12.8**.  
> The default `requirements.txt` installs the CPU-compatible version of PyTorch.  

