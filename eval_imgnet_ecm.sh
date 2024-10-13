
torchrun --nnodes=1 --nproc_per_node=$1 --rdzv_backend=c10d --rdzv_endpoint=localhost:$2 ct_eval.py  \
    --outdir=ct-evals --data=datasets/edm2-imagenet-64x64.zip       \
    --cond=1 --arch=edm2 --preset=edm2-img64-s                      \
    --fp16=1 --mid_t=1.526                                          \
    ${@:3}
