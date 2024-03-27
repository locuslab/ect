# ECM: Consistency Models Made Easy

Pytorch implementation for Easy Consistency Tuning (ECT).

ECT is the key to unlocking SoTA few-step generative capabilities through a simple yet principled approach. With just a negligible tuning cost, ECT demonstrates promising early results while benefiting from the scaling in training FLOPs to continuously enhance its few-step generation capability.

You only need to fine-tune a bit. :D

## Datasets

Prepare the dataset to EDM's format. See a reference [here](https://github.com/NVlabs/edm?tab=readme-ov-file#preparing-datasets).

## Training

Run the following command to run ECM at batch size 128 and 200k iterations. NGPUs=2/4 is recommended. 

```bash
bash run_ecm.sh <NGPUs> <PORT> --desc bs128.200k
```

Replace NGPUs and PORT with the number of GPUs used for training and the port number for DDP sync.

## Evaluation

Run the following command to calculate FID of a pretrain checkpoint ECM. 

```bash
bash eval_ecm.sh <NGPUs> <PORT> --resume <CKPT_PATH> 
```

## Generative Performance

We compare ECMs' unconditional image generation capabilities with SoTA generative models on the CIFAR10 dataset, including popular diffusion models w/ advanced samplers, diffusion distillations, and consistency models on the CIFAR10 dataset.

| Method |  FID | NFE | Model  | Params | Batch Size | Schedule |
| :----  |  :-- | :-- |:---   | :----- | :--------- | :------- |
| Score SDE | 2.38 | 2000 | NCSN++ | 56.4M | 128 | >1500k | 
| Score SDE-deep | 2.20 | 2000 | NCSN++ (2 $\times$ depth) | > 100M | 128 | >1500k |
| EDM                | 8.34 | 1 | DDPM++ | 56.4M | 512 | 800k |
| PD                 | 8.34 | 1 | DDPM++ | 56.4M | 512 | 800k | 
| Diff-Instruct      | 4.53 | 1 | DDPM++ | 56.4M | 512 | 800k | 
| CD (LPIPS)         | 3.55 | 1 | NCSN++ | 56.4M | 512 | 800k | 
| CD (LPIPS)         | 2.93 | 2 | NCSN++ | 56.4M | 512 | 800k | 
| iCT-deep           | 2.51 | 1 | NCSN++ (2 $\times$ depth) | > 100M | 1024 | 400k | 
| iCT-deep           | 2.24 | 2 | NCSN++ (2 $\times$ depth) | > 100M | 1024 | 400k | 
| ECM (100k)         | 4.54 | 1 | DDPM++ | 55.7M | 128 | 100k |
| ECM (100k)         | 2.15 | 2 | DDPM++ | 55.7M | 128 | 100k | 
| ECM (200k)         | 3.86 | 1 | DDPM++ | 55.7M | 128 | 200k |
| ECM (200k)         | 2.15 | 2 | DDPM++ | 55.7M | 128 | 200k | 
| ECM (400k)         | 3.60 | 1 | DDPM++ | 55.7M | 128 | 400k |
| ECM (400k)         | 2.11 | 2 | DDPM++ | 55.7M | 128 | 400k | 

