# ECT: Consistency Models Made Easy

Pytorch implementation for [Easy Consistency Tuning (ECT)](https://www.notion.so/gsunshine/Consistency-Models-Made-Easy-954205c0b4a24c009f78719f43b419cc).

ECT unlocks state-of-the-art (SoTA) few-step generative abilities through a simple yet principled approach. 
With minimal tuning costs, ECT demonstrates promising early results and scales with training FLOPs and model sizes.

Try your own [Consistency Models](https://arxiv.org/abs/2303.01469)! You only need to fine-tune a bit. :D

<div align="center">
    <img src="./assets/learning_scheme.jpg" width="1000" alt="Comparison of Learning Schemes">
</div>

## Introduction

This repository is organized in a multi-branch structure, with each branch offering a minimal implementation for a specific purpose. 
The current branches support the following training protocols:

- `main`: ECT on CIFAR-10. Best for understanding CMs and fast prototyping.
- `amp`: Mixed Precision training with GradScaler on CIFAR-10.
- `imgnet`: ECT ImageNet 64x64.

## ⭐ Update ⭐

Baking more in the oven. 🙃 

- **2024.10.12** - Add ECT code for ImgNet 64x64. Switch to the `imgnet` [branch](https://github.com/locuslab/ect/tree/imgnet): `git checkout imgnet`.
- **2024.09.23** - Add Gradscaler for Mixed Precision Training. To use mixed precision with GradScaler, switch to the `amp` [branch](https://github.com/locuslab/ect/tree/amp): `git checkout amp`.
- **2024.04.27** - Upgrade environment to Pytorch 2.3.0.
- **2024.04.12** - ECMs can now surpass SoTA GANs using 1 model step and SoTA Diffusion Models using 2 model steps on CIFAR10. Checkpoints available.

## Environment

You can run the following command to set up the Python environment through `conda`. 
Pytorch 2.3.0 and Python 3.9.18 will be installed.

```bash
conda env create -f env.yml
```

## Datasets

Prepare the dataset in the EDM's format. See a reference [here](https://github.com/NVlabs/edm?tab=readme-ov-file#preparing-datasets).

## Training

Run the following command to tune your SoTA 2-step ECM and match Consistency Distillation (CD) within 1 A100 GPU hour. 

```bash
bash run_ecm_1hour.sh 1 <PORT> --desc bs128.1hour
```

Run the following command to run ECT at batch size 128 and 200k iterations. NGPUs=2/4 is recommended. 

```bash
bash run_ecm.sh <NGPUs> <PORT> --desc bs128.200k
```

Replace NGPUs and PORT with the number of GPUs used for training and the port number for DDP sync.

### Half Precision Training

In this branch, we enable fp16 training with AMP GradScaler for more stable training dynamics.
To enable fp16 and GradScaler, add the following arguments to your script:

```bash
bash run_ecm_1hour.sh 1 <PORT> --desc bs128.1hour --fp16=True --enable_amp=True
```

For more information, please refer to this [PR](https://github.com/locuslab/ect/pull/13). 
Full support for Automatic Mixed Precision (AMP) will be added later.

## Evaluation

Run the following command to calculate FID of a pretrained checkpoint. 

```bash
bash eval_ecm.sh <NGPUs> <PORT> --resume <CKPT_PATH> 
```

## Generative Performance

### FID Evaluation


Taking the models trained by ECT as ECM, we compare ECMs' unconditional image generation capabilities with SoTA generative models on the CIFAR10 dataset, including popular diffusion models w/ advanced samplers, diffusion distillations, and consistency models on the CIFAR10 dataset.

| Method | FID | NFE | Model  | Params | Batch Size | Schedule |
| :----  | :-- | :-- |:----   | :----- | :--------- | :------- |
| Score SDE | 2.38 | 2000 | NCSN++ | 56.4M | 128 | ~1600k | 
| Score SDE-deep | 2.20 | 2000 | NCSN++ (2 $\times$ depth) | > 100M | 128 | ~1600k |
| EDM                | 2.01 | 35 | DDPM++ | 56.4M | 512 | 400k |
| PD                 | 8.34 | 1  | DDPM++ | 56.4M | 512 | 800k | 
| Diff-Instruct      | 4.53 | 1  | DDPM++ | 56.4M | 512 | 800k | 
| CD (LPIPS)         | 3.55 | 1  | NCSN++ | 56.4M | 512 | 800k | 
| CD (LPIPS)         | 2.93 | 2  | NCSN++ | 56.4M | 512 | 800k | 
| iCT-deep           | 2.51 | 1  | NCSN++ (2 $\times$ depth) | > 100M | 1024 | 400k | 
| iCT-deep           | 2.24 | 2  | NCSN++ (2 $\times$ depth) | > 100M | 1024 | 400k | 
| ECM (100k)         | 4.54 | 1  | DDPM++ | 55.7M | 128 | 100k |
| ECM (200k)         | 3.86 | 1  | DDPM++ | 55.7M | 128 | 200k |
| ECM (400k)         | 3.60 | 1  | DDPM++ | 55.7M | 128 | 400k |
| ECM (100k)         | 2.20 | 2  | DDPM++ | 55.7M | 128 | 100k | 
| ECM (200k)         | 2.15 | 2  | DDPM++ | 55.7M | 128 | 200k | 
| ECM (400k)         | 2.11 | 2  | DDPM++ | 55.7M | 128 | 400k | 

### $\mathrm{FD}_{\text{DINOv2}}$ Evaluation

Since DINOv2 could produce evaluation better aligned with human vision, we evaluate the image fidelity using Fréchet Distance in the latent space of SoTA open-source representation model DINOv2, denoted as 
$\mathrm{FD}_{\text{DINOv2}}$.

---

Using [dgm-eval](https://github.com/layer6ai-labs/dgm-eval/tree/master), we have $\mathrm{FD}_{\text{DINOv2}}$ against SoTA Diffusion Models and GANs.

| Method |  $\mathrm{FD}_{\text{DINOv2}}$  | NFE | 
| :----  |  :-- | :-- |
| [EDM](https://github.com/NVlabs/edm)                                        | 145.20 | 35  |
| [StyleGAN-XL](https://github.com/autonomousvision/stylegan-xl/tree/main)    | 204.60 | 1   |
ECM                                                                           | 198.51 | 1   | 
ECM                                                                           | 128.63 | 2   |

Without combining with other generative mechanisms like GANs or diffusion distillation like Score Distillation, ECT is capable of generating high-quality samples much faster than SoTA diffusion models and much better than ~~SoTA GANs~~ SoTA Diffusion Models and GANs.

## Checkpoints

- CIFAR10 $\mathrm{FD}_{\text{DINOv2}}$ [checkpoint](https://drive.google.com/file/d/1WN_eLTrcl-vB7fMc1HADpacgcO4SNJ_1/view?usp=sharing).

## Contact

Feel free to drop me an email at zhengyanggeng@gmail.com if you have additional questions or are interested in collaboration. You can find me on [Twitter](https://twitter.com/ZhengyangGeng) or [WeChat](https://github.com/Gsunshine/Enjoy-Hamburger/blob/main/assets/WeChat.jpg).

## Citation

```bibtex
@article{ect,
  title={Consistency Models Made Easy},
  author={Geng, Zhengyang and Pokle, Ashwini and Luo, William and Lin, Justin and Kolter, J Zico},
  journal={arXiv preprint arXiv:2406.14548},
  year={2024}
}
```

