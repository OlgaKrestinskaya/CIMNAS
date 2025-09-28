# CIMNAS
# CIMNAS: A Joint Framework for Compute-In-Memory-Aware Neural Architecture Search

This repository contains the official implementation of  
**CIMNAS: A Joint Model-Quantization-Hardware Optimization Framework for CIM Architectures**.

> **Authors:** Olga Krestinskaya, Mohammed E. Fouda, Ahmed Eltawil, and Khaled N. Salama  
> Accepted to **IEEE TCAS-AI**

---

## Overview

CIMNAS simultaneously searches across **software parameters**, **quantization policies**, and a broad range of **hardware parameters**,  
incorporating **device-, circuit-, and architecture-level co-optimizations**.  
The framework is built on top of:

- [**CiMLoop**](https://github.com/mit-emze/cimloop/tree/main)  
- [**APQ**](https://github.com/mit-han-lab/apq/tree/master?tab=readme-ov-file#dataset-and-model-preparation)

> _Illustrative figure to be added here._

- **Paper:** (link will be added)  
- **Citation:** (to be added)

---

## Repository Structure

```
CIMNAS/
-- main/
-- --CIMNAS.ipynb                  # Example: how to run CIMNAS framework
-- --checking_singleHardwareArchitecture.ipynb  # Test a single architecture
-- --testAndInstallations.ipynb    # Installation and debugging instructions
-- --real_accuracy/                # Quantization-aware fine-tuning
-- --APQ/                          # Accuracy predictor & OFA models
-- models/                         # Models supported by CiMLoop + modifications
-- dataset/                        # Placeholder for ImageNet data
```

- **Accuracy predictor**: CIMNAS operates on predicted accuracy.  
  Actual accuracy can be obtained via **quantization-aware fine-tuning** (as in APQ) -  
  see `main/real_accuracy` for instructions.

---

## Requirements

- **Processor:** Multi-core CPU (64 cores used in our experiments recommended for faster search)
- **GPU:** Recommended for speed (not mandatory, but accuracy prediction will be slower without it)
- **GPU required for quantization-aware fine-tuning:** Needed to obtain the actual accuracy of the network found by the search (which is based on the accuracy predictor).

Required libraries are listed in:  
`CIMNAS/main/testAndInstallations.ipynb`

---

## Data and Models

Before running the code:

1. **ImageNet dataset**
   Download [ImageNet dataset](http://www.image-net.org/) (as shown in [APQ](https://github.com/mit-han-lab/apq/tree/master?tab=readme-ov-file#dataset-and-model-preparation)) and place in:
   ```
   CIMNAS/dataset
   ```
   (currently only a few example files are included in the repo).

2. **Model checkpoints**  
   If _CIMNAS/main/APQ/models_ doesn't contain _acc_quant.pt_ and _imagenet-OFA_:
   Download checkpoints for:
   - Quantization-aware predictor [(`acc_quant.pt`)](https://drive.google.com/file/d/1onIxkfLF-QCxi9YxzwQt6SpAaYNJBUDs/view?usp=sharing)
   - Once-for-All network [(`imagenet-OFA`)](https://drive.google.com/file/d/1k9tv1ISsB-QDENspiuR82rDvaIYGIKD5/view?usp=sharing)
   
   Place them in:
   ```
   CIMNAS/main/APQ/models
   ```
   If you plan to run quantization-aware fine-tuning, also place these models in:
   ```
   CIMNAS/main/real_accuracy/models
   ```
   and make ImageNet available in  
   `CIMNAS/main/real_accuracy/dataset/imagenet`  
   (you can use a symbolic link to avoid duplication).

---

## Installation and Initial Run

CIMNAS is built on [**CiMLoop**](https://github.com/mit-emze/cimloop/tree/main), which itself depends on [**Timeloop** and **Accelergy**](https://github.com/Accelergy-Project/timeloop-accelergy-exercises).  
These require Docker with `sudo` (admin) access.

Follow the guidelines in _docker-compose.yaml_ file in **Run as follows** section to set up USER_UID and USER_GID.

Follow steps similar to [CiMLoop](https://github.com/mit-emze/cimloop):

```bash
git clone https://github.com/OlgaKrestinskaya/CIMNAS.git
cd CIMNAS
export DOCKER_ARCH=<your processor architecture>  # e.g., amd64
docker-compose pull
docker-compose up
```

> **ARM64** is supported by Timeloop and Accelergy Docker,  
> but is marked *unstable* - building from source is recommended on ARM as per [CiMLoop](https://github.com/mit-emze/cimloop).

Access JupyterLab from your browser (unless port mapping is changed):  
`http://127.0.0.1:8888/lab`

---

## GPU Support

The `docker-compose.yaml` in this repository has been updated to support GPU.  
If you do **not** want GPU support, you can use CiMLoop's original instructions (you can replace the _docker-compose.yaml_ file content by the content in _docker-compose.yaml_ file from [CiMLoop](https://github.com/mit-emze/cimloop)),  
but you must adjust the PyTorch code where GPU support is set by default.

**Troubleshooting NVIDIA GPU with Docker**

If you see an error like:

```
Error response from daemon: failed to create task for container: ...
nvidia-container-cli: initialization error: load library failed: libnvidia-ml.so.1: cannot open shared object file
```

Reinstall Docker and verify GPU support as described here:  
[Medium guide](https://medium.com/@jared.ratner2/setting-up-docker-and-docker-compose-with-nvidia-gpu-support-on-linux-716db95c0f7c)

Check NVIDIA support:
```bash
sudo docker run --rm --gpus all nvidia/cuda:11.0.3-base-ubuntu20.04 nvidia-smi
```
(or adapt to your CUDA/Ubuntu version).

---

## Usage

- **Installations and debugging:**  
  `CIMNAS/main/testAndInstallations.ipynb`
  
- **Run CIMNAS framework example:**  
  `CIMNAS/main/CIMNAS.ipynb`

- **Test a single architecture:**  
  `CIMNAS/main/checking_singleHardwareArchitecture.ipynb`

For quantization-aware fine-tuning (to get actual accuracy):  
follow `CIMNAS/main/real_accuracy` instructions.

If you need code for **two-stage search**, **expert-like search**, or other experiments described in the paper, please submit a request.

---

## Acknowledgments



CIMNAS builds on:

- [CiMLoop](https://github.com/mit-emze/cimloop)
- [APQ](https://github.com/mit-han-lab/apq)

We thank the maintainers of these projects for their foundational work.

---
