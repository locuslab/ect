import os
import re
import json
import click

import pickle
import psutil
import functools
import PIL.Image
import numpy as np

import torch
import dnnlib
from torch_utils import distributed as dist
from torch_utils import training_stats
from torch_utils import misc

from metrics import metric_main

import warnings
warnings.filterwarnings('ignore', 'Grad strides do not match bucket view strides') # False warning printed by PyTorch 1.12.

#----------------------------------------------------------------------------
# Configuration presets.

config_presets = {
   'edm2-img64-s':     dnnlib.EasyDict(channels=192, lr=0.0010, betas=(0.9, 0.99), decay=2000, dropout=0.40, mean=-0.8, std=1.6),
   'edm2-img64-m':     dnnlib.EasyDict(channels=256, lr=0.0009, betas=(0.9, 0.99), decay=2000, dropout=0.50, mean=-0.8, std=1.6),
   'edm2-img64-l':     dnnlib.EasyDict(channels=320, lr=0.0008, betas=(0.9, 0.99), decay=2000, dropout=0.50, mean=-0.8, std=1.6),
   'edm2-img64-xl':    dnnlib.EasyDict(channels=384, lr=0.0007, betas=(0.9, 0.99), decay=2000, dropout=0.50, mean=-0.8, std=1.6),
}

#----------------------------------------------------------------------------
# Parse a comma separated list of numbers or ranges and return a list of ints.
# Example: '1,2,5-10' returns [1, 2, 5, 6, 7, 8, 9, 10]

def parse_int_list(s):
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges


class CommaSeparatedList(click.ParamType):
    name = 'list'

    def convert(self, value, param, ctx):
        _ = param, ctx
        if value is None or value.lower() == 'none' or value == '':
            return []
        return value.split(',')

#----------------------------------------------------------------------------

@click.command()

# Main options.
@click.option('--outdir',        help='Where to save the results', metavar='DIR',                   type=str, required=True)
@click.option('--data',          help='Path to the dataset', metavar='ZIP|DIR',                     type=str, required=True)
@click.option('--cond',          help='Train class-conditional model', metavar='BOOL',              type=bool, default=False, show_default=True)
@click.option('--arch',          help='Network architecture', metavar='edm2',                       type=click.Choice(['edm2']), default='edm2', show_default=True)
@click.option('--preset',        help='Configuration preset', metavar='STR',                        type=str, default='', show_default=True)

# Hyperparameters.
@click.option('--cbase',         help='Channel multiplier  [default: varies]', metavar='INT',       type=int)
@click.option('--cres',          help='Channels per resolution  [default: varies]', metavar='LIST', type=parse_int_list)
@click.option('--dropout',       help='Dropout probability', metavar='FLOAT',                       type=click.FloatRange(min=0, max=1), default=0.13, show_default=True)
@click.option('--augment',       help='Augment probability', metavar='FLOAT',                       type=click.FloatRange(min=0, max=1), default=0, show_default=True)
@click.option('--xflip',         help='Enable dataset x-flips', metavar='BOOL',                     type=bool, default=False, show_default=True)

# Model Hyperparameters
@click.option('--mean',          help='P_mean of Log Normal Distribution', metavar='FLOAT',         type=click.FloatRange(), default=-1.1, show_default=True)
@click.option('--std',           help='P_std of Log Normal Distribution', metavar='FLOAT',          type=click.FloatRange(), default=2.0, show_default=True)

# Performance-related.
@click.option('--fp16',          help='Enable mixed-precision training', metavar='BOOL',            type=bool, default=False, show_default=True)
@click.option('--bench',         help='Enable cuDNN benchmarking', metavar='BOOL',                  type=bool, default=True, show_default=True)
@click.option('--cache',         help='Cache dataset in CPU memory', metavar='BOOL',                type=bool, default=True, show_default=True)
@click.option('--workers',       help='DataLoader worker processes', metavar='INT',                 type=click.IntRange(min=1), default=1, show_default=True)

# I/O-related.
@click.option('--desc',          help='String to include in result dir name', metavar='STR',        type=str)
@click.option('--nosubdir',      help='Do not create a subdirectory for results',                   is_flag=True)
@click.option('--seed',          help='Random seed  [default: random]', metavar='INT',              type=int)
@click.option('--resume',        help='Load network pickles', metavar='PKL|URL|DIR',                type=str)
@click.option('--resume_pkl',    help='Load network pickles', metavar='PKL|URL|DIR',                type=str)
@click.option('-n', '--dry_run', help='Print training options and exit',                            is_flag=True)

# Evaluation
@click.option('--mid_t',         help='Sampler steps [default: 0.821]', metavar='FLOAT',            multiple=True, default=[0.821], show_default=True)
@click.option('--metrics',       help='Comma-separated list or "none" [default: fid50k_full]',      type=CommaSeparatedList(), default='', show_default=True)
@click.option('--eval_repeat',   help='How many repeats of evaluations', metavar='TICKS',           type=click.IntRange(min=1), default=1, show_default=True)


def main(**kwargs):
    """Train ECMs using the techniques described in the 
    blog "Consistency Models Made Easy".
    """   
    opts = dnnlib.EasyDict(kwargs)
    torch.multiprocessing.set_start_method('spawn')
    dist.init()

    # Preset.
    if opts.preset:
        if opts.preset not in config_presets:
            raise click.ClickException(f'Invalid configuration preset "{opts.preset}"')
        for key, value in config_presets[opts.preset].items():
            opts[key] = value
    
   # Initialize config dict.
    c = dnnlib.EasyDict()
    c.dataset_kwargs = dnnlib.EasyDict(class_name='training.dataset.ImageFolderDataset', path=opts.data, use_labels=opts.cond, xflip=opts.xflip, cache=opts.cache)

    # Validate dataset options.
    try:
        dataset_obj = dnnlib.util.construct_class_by_name(**c.dataset_kwargs)
        dataset_name = dataset_obj.name
        c.dataset_kwargs.resolution = dataset_obj.resolution # be explicit about dataset resolution
        c.dataset_kwargs.max_size = len(dataset_obj) # be explicit about dataset size
        if opts.cond and not dataset_obj.has_labels:
            raise click.ClickException('--cond=True requires labels specified in dataset.json')
        del dataset_obj # conserve memory
    except IOError as err:
        raise click.ClickException(f'--data: {err}')

    # Network architecture.
    c.network_kwargs = dnnlib.EasyDict(class_name='training.networks.ECMPrecond')
    
    assert opts.arch == 'edm2'
    c.network_kwargs.update(class_name='training.networks_edm2.Precond', model_channels=opts.channels)

    # Network options.
    if opts.cbase is not None:
        c.network_kwargs.model_channels = opts.cbase
    if opts.cres is not None:
        c.network_kwargs.channel_mult = opts.cres
    if opts.augment:
        c.network_kwargs.augment_dim = 9
    c.network_kwargs.update(dropout=opts.dropout, use_fp16=opts.fp16)

    # Trainig options.
    c.update(cudnn_benchmark=opts.bench)
    c.update(mid_t=opts.mid_t, metrics=opts.metrics)

    # Random seed.
    if opts.seed is not None:
        c.seed = opts.seed
    else:
        seed = torch.randint(1 << 31, size=[], device=torch.device('cuda'))
        torch.distributed.broadcast(seed, src=0)
        c.seed = int(seed)

    # Description string.
    cond_str = 'cond' if c.dataset_kwargs.use_labels else 'uncond'
    dtype_str = 'fp16' if c.network_kwargs.use_fp16 else 'fp32'
    desc = f'{dataset_name:s}-{cond_str:s}-{opts.arch:s}-{opts.preset:s}-gpus{dist.get_world_size():d}-{dtype_str:s}'
    if opts.desc is not None:
        desc += f'-{opts.desc}'

    # Pick output directory.
    if dist.get_rank() != 0:
        c.run_dir = None
    elif opts.nosubdir:
        c.run_dir = opts.outdir
    else:
        prev_run_dirs = []
        if os.path.isdir(opts.outdir):
            prev_run_dirs = [x for x in os.listdir(opts.outdir) if os.path.isdir(os.path.join(opts.outdir, x))]
        prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
        prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
        cur_run_id = max(prev_run_ids, default=-1) + 1
        c.run_dir = os.path.join(opts.outdir, f'{cur_run_id:05d}-{desc}')
        assert not os.path.exists(c.run_dir)

    # Print options.
    dist.print0()
    dist.print0('Evaluarion options:')
    dist.print0(json.dumps(c, indent=2))
    dist.print0()
    dist.print0(f'Output directory:        {c.run_dir}')
    dist.print0(f'Dataset path:            {c.dataset_kwargs.path}')
    dist.print0(f'Class-conditional:       {c.dataset_kwargs.use_labels}')
    dist.print0(f'Network architecture:    {opts.arch}')
    dist.print0(f'Number of GPUs:          {dist.get_world_size()}')
    dist.print0(f'Mixed-precision:         {c.network_kwargs.use_fp16}')
    dist.print0()

    # Dry run?
    if opts.dry_run:
        dist.print0('Dry run; exiting.')
        return

    # Create output directory.
    dist.print0('Creating output directory...')
    if dist.get_rank() == 0:
        os.makedirs(c.run_dir, exist_ok=True)
        with open(os.path.join(c.run_dir, 'training_options.json'), 'wt') as f:
            json.dump(c, f, indent=2)
        dnnlib.util.Logger(file_name=os.path.join(c.run_dir, 'log.txt'), file_mode='a', should_flush=True)

    # Checkpoints to evaluate.
    c.eval_repeat = opts.eval_repeat
    
    if opts.resume_pkl is not None:
        if os.path.isdir(opts.resume_pkl):
            # Load all snapshots from the directory
            checkpoint_paths = [os.path.join(opts.resume_pkl, f) for f in os.listdir(opts.resume_pkl) if f.endswith('.pkl')]
        else:
            # Load a single snapshot
            checkpoint_paths = [opts.resume_pkl]

        checkpoint_paths = sorted(checkpoint_paths)
        for ckpt_path in checkpoint_paths:
            c.resume_pkl = ckpt_path
            evaluation_pkl(**c)

    if opts.resume is not None:
        if os.path.isdir(opts.resume):
            # Load all checkpoints from the directory
            checkpoint_paths = [os.path.join(opts.resume, f) for f in os.listdir(opts.resume) if f.endswith('.pt')]
        else:
            # Load a single checkpoint
            checkpoint_paths = [opts.resume]

        checkpoint_paths = sorted(checkpoint_paths)
        for ckpt_path in checkpoint_paths:
            c.resume_pt = ckpt_path
            evaluation_pt(**c)


#----------------------------------------------------------------------------

def setup_snapshot_image_grid(training_set, random_seed=0):
    rnd = np.random.RandomState(random_seed)
    gw = np.clip(7680 // training_set.image_shape[2], 7, 16)
    gh = np.clip(4320 // training_set.image_shape[1], 4, 16)

    # No labels => show random subset of training samples.
    if not training_set.has_labels:
        all_indices = list(range(len(training_set)))
        rnd.shuffle(all_indices)
        grid_indices = [all_indices[i % len(all_indices)] for i in range(gw * gh)]

    else:
        # Group training samples by label.
        label_groups = dict() # label => [idx, ...]
        for idx in range(len(training_set)):
            label = tuple(training_set.get_details(idx).raw_label.flat[::-1])
            if label not in label_groups:
                label_groups[label] = []
            label_groups[label].append(idx)

        # Reorder.
        label_order = sorted(label_groups.keys())
        for label in label_order:
            rnd.shuffle(label_groups[label])

        # Organize into grid.
        grid_indices = []
        for y in range(gh):
            label = label_order[y % len(label_order)]
            indices = label_groups[label]
            grid_indices += [indices[x % len(indices)] for x in range(gw)]
            label_groups[label] = [indices[(i + gw) % len(indices)] for i in range(len(indices))]

    # Load data.
    images, labels = zip(*[training_set[i] for i in grid_indices])
    return (gw, gh), np.stack(images), np.stack(labels)
    
#----------------------------------------------------------------------------

def save_image_grid(img, fname, drange, grid_size):
    lo, hi = drange
    img = np.asarray(img, dtype=np.float32)
    img = (img - lo) * (255 / (hi - lo))
    img = np.rint(img).clip(0, 255).astype(np.uint8)

    gw, gh = grid_size
    _N, C, H, W = img.shape
    img = img.reshape(gh, gw, C, H, W)
    img = img.transpose(0, 3, 1, 4, 2)
    img = img.reshape(gh * H, gw * W, C)

    assert C in [1, 3]
    if C == 1:
        PIL.Image.fromarray(img[:, :, 0], 'L').save(fname)
    if C == 3:
        PIL.Image.fromarray(img, 'RGB').save(fname)

#----------------------------------------------------------------------------

@torch.no_grad()
def generator_fn(
    net, latents, class_labels=None, 
    t_max=80, mid_t=None
):
    # Time step discretization.
    mid_t = [] if mid_t is None else mid_t
    t_steps = torch.tensor([t_max]+list(mid_t), dtype=torch.float64, device=latents.device)

    # t_0 = T, t_N = 0
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])

    # Sampling steps 
    x = latents.to(torch.float64) * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x = net(x, t_cur, class_labels).to(torch.float64)
        if t_next > 0:
            x = x + t_next * torch.randn_like(x) 
    return x

#----------------------------------------------------------------------------

def evaluation_pkl(
    run_dir             = '.',      # Output directory.
    dataset_kwargs      = {},       # Options for training set.
    network_kwargs      = {},       # Options for model and preconditioning.
    batch_size          = 512,      # Total batch size for one training iteration.
    seed                = 0,        # Global random seed.
    resume_pkl          = None,     # Resume from the given network snapshot, None = random initialization.
    mid_t               = None,     # Intermediate t for few-step generation.
    metrics             = None,     # Metrics for evaluation.
    eval_repeat         = 1,        # Number of evaluations.
    cudnn_benchmark     = True,     # Enable torch.backends.cudnn.benchmark?
    device              = torch.device('cuda'),
):
    # Initialize.
    np.random.seed((seed * dist.get_world_size() + dist.get_rank()) % (1 << 31))
    torch.manual_seed(np.random.randint(1 << 31))
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True

    # Select batch size per GPU.
    batch_gpu = batch_size // dist.get_world_size()

    # Load dataset.
    dist.print0('Loading dataset...')
    dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs) # subclass of training.dataset.Dataset

    # Construct network.
    dist.print0('Constructing network...')
    interface_kwargs = dict(img_resolution=dataset_obj.resolution, img_channels=dataset_obj.num_channels, label_dim=dataset_obj.label_dim)
    net = dnnlib.util.construct_class_by_name(**network_kwargs, **interface_kwargs) # subclass of torch.nn.Module
    net.eval().requires_grad_(False).to(device)
  
    # Export sample images.
    grid_size = None
    grid_z = None
    grid_c = None
        
    if dist.get_rank() == 0:
        dist.print0('Exporting sample images...')
        grid_size, images, labels = setup_snapshot_image_grid(training_set=dataset_obj)
        save_image_grid(images, os.path.join(run_dir, 'data.png'), drange=[0,255], grid_size=grid_size)
        
        grid_z = torch.randn([labels.shape[0], net.img_channels, net.img_resolution, net.img_resolution], device=device)
        grid_z = grid_z.split(batch_gpu)
        
        grid_c = torch.from_numpy(labels).to(device)
        grid_c = grid_c.split(batch_gpu)
    
    # Resume training from previous snapshot.
    if resume_pkl is not None:
        dist.print0(f'Loading network weights from "{resume_pkl}"...')
        if dist.get_rank() != 0:
            torch.distributed.barrier() # rank 0 goes first
        with dnnlib.util.open_url(resume_pkl, verbose=(dist.get_rank() == 0)) as f:
            data = pickle.load(f)
        if dist.get_rank() == 0:
            torch.distributed.barrier() # other ranks follow
        misc.copy_params_and_buffers(src_module=data['ema'], dst_module=net, require_all=False)
        del data # conserve memory
    
    # Few-step Evaluation.
    few_step_fn = functools.partial(generator_fn, mid_t=mid_t)
    
    if dist.get_rank() == 0:
        dist.print0('Exporting final sample images...')
        images = [few_step_fn(net, z, c).cpu() for z, c in zip(grid_z, grid_c)]
        images = torch.cat(images).numpy()
        img_path = resume_pkl.split('/')[-1].split('.pkl')[0]
        save_image_grid(images, os.path.join(run_dir, f'{img_path}.png'), drange=[-1,1], grid_size=grid_size)
        del images

    dist.print0('Evaluating few-step generation...')
    for _ in range(eval_repeat):
        one_step_results = metric_main.calc_metric(metric='fid50k_full',
                generator_fn=generator_fn, G=net, G_kwargs={},
                dataset_kwargs=dataset_kwargs, num_gpus=dist.get_world_size(), rank=dist.get_rank(), device=device)
        if dist.get_rank() == 0:
            metric_main.report_metric(one_step_results, run_dir=run_dir, snapshot_pkl=f'{resume_pkl}')
        
        few_step_fn = functools.partial(generator_fn, mid_t=mid_t)
        few_step_results = metric_main.calc_metric(metric='two_step_fid50k_full', 
                generator_fn=few_step_fn, G=net, G_kwargs={},
                dataset_kwargs=dataset_kwargs, num_gpus=dist.get_world_size(), rank=dist.get_rank(), device=device)
        if dist.get_rank() == 0:
            metric_main.report_metric(few_step_results, run_dir=run_dir, snapshot_pkl=f'{resume_pkl}')
        
        for metric in metrics:
            result_dict = metric_main.calc_metric(metric=metric, 
                generator_fn=few_step_fn, G=net, G_kwargs={},
                dataset_kwargs=dataset_kwargs, num_gpus=dist.get_world_size(), rank=dist.get_rank(), device=device)
            if dist.get_rank() == 0:
                metric_main.report_metric(result_dict, run_dir=run_dir, snapshot_pkl=f'{resume_pkl}-2step')

    # Done.
    dist.print0()
    dist.print0('Exiting...')


def evaluation_pt(
    run_dir             = '.',      # Output directory.
    dataset_kwargs      = {},       # Options for training set.
    network_kwargs      = {},       # Options for model and preconditioning.
    batch_size          = 512,      # Total batch size for one training iteration.
    seed                = 0,        # Global random seed.
    resume_pt           = None,      # Resume from the given checkpoint, None = random initialization.
    mid_t               = None,     # Intermediate t for few-step generation.
    metrics             = None,     # Metrics for evaluation.
    eval_repeat         = 1,        # Number of evaluations.
    cudnn_benchmark     = True,     # Enable torch.backends.cudnn.benchmark?
    device              = torch.device('cuda'),
):
    # Initialize.
    np.random.seed((seed * dist.get_world_size() + dist.get_rank()) % (1 << 31))
    torch.manual_seed(np.random.randint(1 << 31))
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True

    # Select batch size per GPU.
    batch_gpu = batch_size // dist.get_world_size()

    # Load dataset.
    dist.print0('Loading dataset...')
    dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs) # subclass of training.dataset.Dataset

    # Construct network.
    dist.print0('Constructing network...')
    interface_kwargs = dict(img_resolution=dataset_obj.resolution, img_channels=dataset_obj.num_channels, label_dim=dataset_obj.label_dim)
    net = dnnlib.util.construct_class_by_name(**network_kwargs, **interface_kwargs) # subclass of torch.nn.Module
    net.eval().requires_grad_(False).to(device)
    
    dist.print0('Setting up EMA...')
    ema_kwargs = dict(class_name='training.phema.PowerFunctionEMA')
    ema        = dnnlib.util.construct_class_by_name(net=net, **ema_kwargs)

    # Export sample images.
    grid_size = None
    grid_z = None
    grid_c = None
        
    if dist.get_rank() == 0:
        dist.print0('Exporting sample images...')
        grid_size, images, labels = setup_snapshot_image_grid(training_set=dataset_obj)
        save_image_grid(images, os.path.join(run_dir, 'data.png'), drange=[0,255], grid_size=grid_size)
        
        grid_z = torch.randn([labels.shape[0], net.img_channels, net.img_resolution, net.img_resolution], device=device)
        grid_z = grid_z.split(batch_gpu)
        
        grid_c = torch.from_numpy(labels).to(device)
        grid_c = grid_c.split(batch_gpu)
    
    if resume_pt is not None:
        data = torch.load(resume_pt, map_location=torch.device('cpu'))
        
        misc.copy_params_and_buffers(src_module=data['net'], dst_module=net, require_all=True)
        ema.load_state_dict(data['ema_state'])
        del data # conserve memory
 
    # Few-step Evaluation.
    few_step_fn = functools.partial(generator_fn, mid_t=mid_t)
    
    # Sample Img
    dist.print0(f'Using mid ts of {mid_t}...')
    if dist.get_rank() == 0:
        ema_list = ema.get()
        for ema_net, ema_suffix in ema_list:
            images = [generator_fn(ema_net, z, c).cpu() for z, c in zip(grid_z, grid_c)]
            images = torch.cat(images).numpy()
            save_image_grid(images, os.path.join(run_dir, f'1_step_ema{ema_suffix}.png'), drange=[-1,1], grid_size=grid_size)

            few_step_fn = functools.partial(generator_fn, mid_t=mid_t)
            images = [few_step_fn(ema_net, z, c).cpu() for z, c in zip(grid_z, grid_c)]
            images = torch.cat(images).numpy()
            save_image_grid(images, os.path.join(run_dir, f'2_step_ema{ema_suffix}.png'), drange=[-1,1], grid_size=grid_size)
        del images

    dist.print0('Evaluating PH-EMA models...')
    for _ in range(eval_repeat):
        ema_list = ema.get()
        for ema_net, ema_suffix in ema_list:
            one_step_results = metric_main.calc_metric(metric='fid50k_full',
                    generator_fn=generator_fn, G=ema_net, G_kwargs={},
                    dataset_kwargs=dataset_kwargs, num_gpus=dist.get_world_size(), rank=dist.get_rank(), device=device)
            if dist.get_rank() == 0:
                metric_main.report_metric(one_step_results, run_dir=run_dir, snapshot_pkl=f'{ema_suffix}')

            few_step_fn = functools.partial(generator_fn, mid_t=mid_t)
            few_step_results = metric_main.calc_metric(metric='two_step_fid50k_full', 
                generator_fn=few_step_fn, G=ema_net, G_kwargs={},
                dataset_kwargs=dataset_kwargs, num_gpus=dist.get_world_size(), rank=dist.get_rank(), device=device)
            if dist.get_rank() == 0:
                metric_main.report_metric(few_step_results, run_dir=run_dir, snapshot_pkl=f'{ema_suffix}')

    # Done.
    dist.print0()
    dist.print0('Exiting...')


#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
