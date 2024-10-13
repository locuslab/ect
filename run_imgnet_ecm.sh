
torchrun --nnodes=1 --nproc_per_node=$1 --rdzv_backend=c10d --rdzv_endpoint=localhost:$2 ct_train.py        \
    --outdir=ct-runs --data=datasets/edm2-imagenet-64x64.zip                \
    --cond=1 --arch=edm2 --preset=edm2-img64-s                              \
    --fp16=0 --cache=True --mid_t=1.526                                     \
    --duration=12.8 --tick=6.4 --batch=128 --batch-gpu=32                   \
    --double 500 --snap 2000 --dump 500 --optim Adam --wt snrpk             \
    --transfer=https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions/edm2-img64-s-1073741-0.075.pkl \
    ${@:3}
