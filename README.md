# ECT: Consistency Models Made Easy

Pytorch implementation for [Easy Consistency Tuning (ECT)](https://www.notion.so/gsunshine/Consistency-Models-Made-Easy-954205c0b4a24c009f78719f43b419cc).

ECT unlocks SoTA few-step generative abilities through a simple yet principled approach. With a negligible tuning cost, ECT demonstrates promising early results while benefiting from the scaling to improve its few-step generation capability.

Try your own [Consistency Models](https://arxiv.org/abs/2303.01469)! You only need to fine-tune a bit. :D

<div align="center">
    <img src="./assets/learning_scheme.jpg" width="1000" alt="Comparison of Learning Schemes">
</div>

## Introduction
v
This branch provides a minimal setup to tune CMs on the ImageNet 64x64 dataset. The dataset preparation follows the format used by ADM and EDM2.

## Environment

You can run the following command to set up the Python environment through `conda`. 
Pytorch 2.3.0 and Python 3.9.18 will be installed.

```bash
conda env create -f env.yml
```

## Datasets

To prepare the ImageNet 64x64 dataset, please follow the steps below: 

1. Download the the ILSVRC2012 data archive from [Kaggle](https://www.kaggle.com/competitions/imagenet-object-localization-challenge/data) and extract it.
2. Clone the EDM2 repository
    -   ```bash
        git clone https://github.com/NVlabs/edm2.git
        mv data_prep.sh edm2
        ```
3. Configure paths in `data_prep.sh`.
    - `DATA_DIR`: Set this to the path where you saved the extracted ImageNet train set.
    - `DEST_DIR`: Set this to the directory where you want to store the processed ImageNet 64x64 dataset.
4. Run `bash data_prep.sh`.
5. Link your `zip` dataset to the local `datasets` directory.
    ```bash
    mkdir datasets
    ln -s $DEST_DIR/edm2-imagenet-64x64.zip ./datasets/
    ```

## Training

Run the following command to run ECT at batch size 128 and 100k iterations. NGPUs=4/8 is recommended. 

```bash
bash run_imgnet_ecm.sh <NGPUs> <PORT> --desc bs128.100k
```

- Replace `<NGPUs>` and `<PORT>` with the number of GPUs used for training and the port number for DDP sync. 
- Modify `--preset` to the model configs to launch.

## Evaluation

Run the following command to calculate FID of a pretrained checkpoint. 

```bash
bash eval_imgnet_ecm.sh <NGPUs> <PORT> --resume <CKPT_PATH> 
```

- Use `--resume_pkl` to specify a snapshot (.pkl) or the directory containing multiple snapshots for evaluation.
- Use `--resume` to specify a checkpoint (.pt) or the directory containing multiple checkpoints for evaluation.

## Checkpoints

- ImgNet 64x64 ECM-XL [checkpoints](https://drive.google.com/drive/folders/137l1CdsC25Ez7CCIfN4XuqrNmYbOscWD?usp=sharing) trained at bs1024 for 100k iterations.

## Contact

Feel free to drop me an email at zhengyanggeng@gmail.com if you have additional questions or are interested in collaboration. You can also find me on [Twitter](https://twitter.com/ZhengyangGeng) or [WeChat](https://github.com/Gsunshine/Enjoy-Hamburger/blob/main/assets/WeChat.jpg).

## Citation

```bibtex
@article{ect,
  title={Consistency Models Made Easy},
  author={Geng, Zhengyang and Pokle, Ashwini and Luo, William and Lin, Justin and Kolter, J Zico},
  journal={arXiv preprint arXiv:2406.14548},
  year={2024}
}
```
