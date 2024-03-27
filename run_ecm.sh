
torchrun --nnodes=1 --nproc_per_node=$1 --rdzv_backend=c10d --rdzv_endpoint=localhost:$2 ct_train.py  \
    --outdir=ct-runs --data=datasets/cifar10-32x32.zip  \
    --cond=0 --arch=ddpmpp --metrics=fid50k_full        \
    --transfer=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-uncond-vp.pkl    \
    --duration=25.6 --tick=12.8 --double=250 --batch=128 --lr=0.0001 --optim=RAdam --dropout=0.2 --augment=0.0 \
    ${@:3}
