import os
import time
import copy
import json
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
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])

    # Sampling steps 
    x = latents.to(torch.float64) * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x = net(x, t_cur, class_labels).to(torch.float64)
        if t_next > 0:
            x = x + t_next * torch.randn_like(x) 
    return x

#----------------------------------------------------------------------------

def training_loop(
    run_dir             = '.',      # Output directory.
    dataset_kwargs      = {},       # Options for training set.
    data_loader_kwargs  = {},       # Options for torch.utils.data.DataLoader.
    network_kwargs      = {},       # Options for model and preconditioning.
    loss_kwargs         = {},       # Options for loss function.
    optimizer_kwargs    = {},       # Options for optimizer.
    augment_kwargs      = None,     # Options for augmentation pipeline, None = disable.
    seed                = 0,        # Global random seed.
    batch_size          = 512,      # Total batch size for one training iteration.
    batch_gpu           = None,     # Limit batch size per GPU, None = no limit.
    total_kimg          = 200000,   # Training duration, measured in thousands of training images.
    ema_beta            = 0.9999,   # EMA decay rate. Overwritten by ema_halflife_kimg.
    ema_halflife_kimg   = None,     # Half-life of the exponential moving average (EMA) of model weights.
    ema_rampup_ratio    = None,     # EMA ramp-up coefficient, None = no rampup.
    lr_rampup_kimg      = 0,        # Learning rate ramp-up duration.
    loss_scaling        = 1,        # Loss scaling factor for reducing FP16 under/overflows.
    kimg_per_tick       = 50,       # Interval of progress prints.
    snapshot_ticks      = 500,      # How often to save network snapshots, None = disable.
    state_dump_ticks    = 500,      # How often to dump training state, None = disable.
    ckpt_ticks          = 100,      # How often to save latest checkpoints, None = disable.
    sample_ticks        = 50,       # How often to sample images, None = disable.
    eval_ticks          = 500,      # How often to evaluate models, None = disable.
    double_ticks        = 500,      # How often to evaluate models, None = disable.
    resume_pkl          = None,     # Start from the given network snapshot, None = random initialization.
    resume_state_dump   = None,     # Start from the given training state, None = reset training state.
    resume_tick         = 0,        # Start from the given training progress.
    mid_t               = None,     # Intermediate t for few-step generation.
    metrics             = None,     # Metrics for evaluation.
    cudnn_benchmark     = True,     # Enable torch.backends.cudnn.benchmark?
    enable_tf32         = False,     # Enable tf32 for A100/H100 GPUs?
    device              = torch.device('cuda'),
):
    # Initialize.
    start_time = time.time()
    np.random.seed((seed * dist.get_world_size() + dist.get_rank()) % (1 << 31))
    torch.manual_seed(np.random.randint(1 << 31))
    torch.backends.cudnn.benchmark = cudnn_benchmark

    # Enable these to speed up on A100 GPUs
    dist.print0(f'Enable tf32: {enable_tf32}')
    torch.backends.cudnn.allow_tf32 = enable_tf32
    torch.backends.cuda.matmul.allow_tf32 = enable_tf32
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = enable_tf32

    # Select batch size per GPU.
    batch_gpu_total = batch_size // dist.get_world_size()
    if batch_gpu is None or batch_gpu > batch_gpu_total:
        batch_gpu = batch_gpu_total
    num_accumulation_rounds = batch_gpu_total // batch_gpu
    assert batch_size == batch_gpu * num_accumulation_rounds * dist.get_world_size()

    # Load dataset.
    dist.print0('Loading dataset...')
    dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs) # subclass of training.dataset.Dataset
    dataset_sampler = misc.InfiniteSampler(dataset=dataset_obj, rank=dist.get_rank(), num_replicas=dist.get_world_size(), seed=seed)
    dataset_iterator = iter(torch.utils.data.DataLoader(dataset=dataset_obj, sampler=dataset_sampler, batch_size=batch_gpu, **data_loader_kwargs))

    # Construct network.
    dist.print0('Constructing network...')
    interface_kwargs = dict(img_resolution=dataset_obj.resolution, img_channels=dataset_obj.num_channels, label_dim=dataset_obj.label_dim)
    net = dnnlib.util.construct_class_by_name(**network_kwargs, **interface_kwargs) # subclass of torch.nn.Module
    net.train().requires_grad_(True).to(device)
    
    # Setup optimizer.
    dist.print0('Setting up optimizer...')
    loss_fn = dnnlib.util.construct_class_by_name(**loss_kwargs)
    optimizer = dnnlib.util.construct_class_by_name(params=net.parameters(), **optimizer_kwargs) # subclass of torch.optim.Optimizer
    augment_pipe = dnnlib.util.construct_class_by_name(**augment_kwargs) if augment_kwargs is not None else None # training.augment.AugmentPipe
    
    dist.print0('Setting up DDP...')
    ddp = torch.nn.parallel.DistributedDataParallel(net, device_ids=[device], broadcast_buffers=False)
    ema = copy.deepcopy(net).eval().requires_grad_(False)
    
    # Stats
    if dist.get_rank() == 0:
        with torch.no_grad():
            images = torch.zeros([batch_gpu, net.img_channels, net.img_resolution, net.img_resolution], device=device)
            sigma = torch.ones([batch_gpu], device=device)
            labels = torch.zeros([batch_gpu, net.label_dim], device=device)
            misc.print_module_summary(net, [images, sigma, labels], max_nesting=2)

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
        misc.copy_params_and_buffers(src_module=data['ema'], dst_module=ema, require_all=False)
        del data # conserve memory
    if resume_state_dump:
        dist.print0(f'Loading training state from "{resume_state_dump}"...')
        data = torch.load(resume_state_dump, map_location=torch.device('cpu'))
        misc.copy_params_and_buffers(src_module=data['net'], dst_module=net, require_all=True)
        optimizer.load_state_dict(data['optimizer_state'])
        del data # conserve memory
    
    # Export sample images.
    grid_size = None
    grid_z = None
    grid_c = None
        
    if dist.get_rank() == 0:
        dist.print0('Exporting sample images...')
        grid_size, images, labels = setup_snapshot_image_grid(training_set=dataset_obj)
        save_image_grid(images, os.path.join(run_dir, 'data.png'), drange=[0,255], grid_size=grid_size)
        
        grid_z = torch.randn([labels.shape[0], ema.img_channels, ema.img_resolution, ema.img_resolution], device=device)
        grid_z = grid_z.split(batch_gpu)
        
        grid_c = torch.from_numpy(labels).to(device)
        grid_c = grid_c.split(batch_gpu)
        
        images = [generator_fn(ema, z, c).cpu() for z, c in zip(grid_z, grid_c)]
        images = torch.cat(images).numpy()
        save_image_grid(images, os.path.join(run_dir, 'model_init.png'), drange=[-1,1], grid_size=grid_size)
        del images

    # Train.
    dist.print0(f'Training for {total_kimg} kimg...')
    dist.print0()
    cur_nimg = resume_tick * kimg_per_tick * 1000
    cur_tick = resume_tick
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    dist.update_progress(cur_nimg / 1000, total_kimg)
    stats_jsonl = None

    # Prepare for the mapping fn p(r|t).
    dist.print0(f'Reduce dt every {double_ticks} ticks.')
    
    def update_scheduler(loss_fn):
        loss_fn.update_schedule(stage)
        dist.print0(f'Update scheduler at {cur_tick} ticks, {cur_nimg / 1e3} kimg, ratio {loss_fn.ratio}')
        
    stage = cur_tick // double_ticks
    update_scheduler(loss_fn)

    while True:

        # Accumulate gradients.
        optimizer.zero_grad(set_to_none=True)
        for round_idx in range(num_accumulation_rounds):
            with misc.ddp_sync(ddp, (round_idx == num_accumulation_rounds - 1)):
                images, labels = next(dataset_iterator)
                images = images.to(device).to(torch.float32) / 127.5 - 1
                labels = labels.to(device)

                loss = loss_fn(net=ddp, images=images, labels=labels, augment_pipe=augment_pipe)
                training_stats.report('Loss/loss', loss)
                loss.mul(loss_scaling).mean().backward()

        # LR scheduler (if needed in the future)
        # for g in optimizer.param_groups:
        #     g['lr'] = optimizer_kwargs['lr'] * min(cur_nimg / max(lr_rampup_kimg * 1000, 1e-8), 1)
        
        # Update weights.
        for param in net.parameters():
            if param.grad is not None:
                torch.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
        optimizer.step()

        # Update EMA.
        if ema_halflife_kimg is not None:
            ema_halflife_nimg = ema_halflife_kimg * 1000
            if ema_rampup_ratio is not None:
                ema_halflife_nimg = min(ema_halflife_nimg, cur_nimg * ema_rampup_ratio)
            ema_beta = 0.5 ** (batch_size / max(ema_halflife_nimg, 1e-8))
        for p_ema, p_net in zip(ema.parameters(), net.parameters()):
            p_ema.copy_(p_net.detach().lerp(p_ema, ema_beta))

        # Perform maintenance tasks once per tick.
        cur_nimg += batch_size
        done = (cur_nimg >= total_kimg * 1000)
        if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
            continue

        # Print status line, accumulating the same information in training_stats.
        tick_end_time = time.time()
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<9.1f}"]
        fields += [f"loss {training_stats.default_collector['Loss/loss']:<9.5f}"]
        fields += [f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
        fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
        fields += [f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"]
        fields += [f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"]
        fields += [f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"]
        fields += [f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}"]
        fields += [f"reserved {training_stats.report0('Resources/peak_gpu_mem_reserved_gb', torch.cuda.max_memory_reserved(device) / 2**30):<6.2f}"]
        torch.cuda.reset_peak_memory_stats()
        dist.print0(' '.join(fields))

        # Check for abort.
        if (not done) and dist.should_stop():
            done = True
            dist.print0()
            dist.print0('Aborting...')

        # Save network snapshot.
        if (snapshot_ticks is not None) and (done or cur_tick % snapshot_ticks == 0) and cur_tick != 0:
            data = dict(ema=ema, loss_fn=loss_fn, augment_pipe=augment_pipe, dataset_kwargs=dict(dataset_kwargs))
            for key, value in data.items():
                if isinstance(value, torch.nn.Module):
                    value = copy.deepcopy(value).eval().requires_grad_(False)
                    misc.check_ddp_consistency(value)
                    data[key] = value.cpu()
                del value # conserve memory
            if dist.get_rank() == 0:
                with open(os.path.join(run_dir, f'network-snapshot-{cur_tick:06d}.pkl'), 'wb') as f:
                    pickle.dump(data, f)
            del data # conserve memory

        # Save full dump of the training state.
        if (state_dump_ticks is not None) and (done or cur_tick % state_dump_ticks == 0) and cur_tick != 0 and dist.get_rank() == 0:
            torch.save(dict(net=net, optimizer_state=optimizer.state_dict()), os.path.join(run_dir, f'training-state-{cur_tick:06d}.pt'))

        # Save latest checkpoints
        if (ckpt_ticks is not None) and (done or cur_tick % ckpt_ticks == 0) and cur_tick != 0:
            dist.print0(f'Save the latest checkpoint at {cur_tick:06d} img...')
            data = dict(ema=ema, loss_fn=loss_fn, augment_pipe=augment_pipe, dataset_kwargs=dict(dataset_kwargs))
            for key, value in data.items():
                if isinstance(value, torch.nn.Module):
                    value = copy.deepcopy(value).eval().requires_grad_(False)
                    misc.check_ddp_consistency(value)
                    data[key] = value.cpu()
                del value # conserve memory
            if dist.get_rank() == 0:
                with open(os.path.join(run_dir, f'network-snapshot-latest.pkl'), 'wb') as f:
                    pickle.dump(data, f)
            del data # conserve memory

            if dist.get_rank() == 0:
                torch.save(dict(net=net, optimizer_state=optimizer.state_dict()), os.path.join(run_dir, f'training-state-latest.pt'))

        # Sample Img
        if (sample_ticks is not None) and (done or cur_tick % sample_ticks == 0) and dist.get_rank() == 0:
            dist.print0('Exporting sample images...')
            images = [generator_fn(ema, z, c).cpu() for z, c in zip(grid_z, grid_c)]
            images = torch.cat(images).numpy()
            save_image_grid(images, os.path.join(run_dir, f'{cur_tick:06d}.png'), drange=[-1,1], grid_size=grid_size)
            del images
    
        # Evaluation
        if (eval_ticks is not None) and (done or cur_tick % eval_ticks == 0) and cur_tick > 0:
            dist.print0('Evaluating models...')
            result_dict = metric_main.calc_metric(metric='fid50k_full', 
                    generator_fn=generator_fn, G=ema, G_kwargs={},
                    dataset_kwargs=dataset_kwargs, num_gpus=dist.get_world_size(), rank=dist.get_rank(), device=device)
            if dist.get_rank() == 0:
                metric_main.report_metric(result_dict, run_dir=run_dir, snapshot_pkl=f'network-snapshot-{cur_tick:06d}.pkl')                        
            
            few_step_fn = functools.partial(generator_fn, mid_t=mid_t)
            result_dict = metric_main.calc_metric(metric='two_step_fid50k_full', 
                    generator_fn=few_step_fn, G=ema, G_kwargs={},
                    dataset_kwargs=dataset_kwargs, num_gpus=dist.get_world_size(), rank=dist.get_rank(), device=device)
            if dist.get_rank() == 0:
                metric_main.report_metric(result_dict, run_dir=run_dir, snapshot_pkl=f'network-snapshot-{cur_tick:06d}.pkl')                        

        # Update logs.
        training_stats.default_collector.update()
        if dist.get_rank() == 0:
            if stats_jsonl is None:
                stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'at')
            stats_jsonl.write(json.dumps(dict(training_stats.default_collector.as_dict(), timestamp=time.time())) + '\n')
            stats_jsonl.flush()
        dist.update_progress(cur_nimg / 1000, total_kimg)

        # Update state.
        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        if done:
            break
        
        # Update Scheduler
        new_stage = (cur_tick-1) // double_ticks
        if new_stage > stage:
            stage = new_stage
            update_scheduler(loss_fn)
    
    # Few-step Evaluation.
    few_step_fn = functools.partial(generator_fn, mid_t=mid_t)
    
    if dist.get_rank() == 0:
        dist.print0('Exporting final sample images...')
        images = [few_step_fn(ema, z, c).cpu() for z, c in zip(grid_z, grid_c)]
        images = torch.cat(images).numpy()
        save_image_grid(images, os.path.join(run_dir, 'final.png'), drange=[-1,1], grid_size=grid_size)
        del images

    dist.print0('Evaluating few-step generation...')
    for _ in range(3):
        for metric in metrics:
            result_dict = metric_main.calc_metric(metric=metric, 
                generator_fn=few_step_fn, G=ema, G_kwargs={},
                dataset_kwargs=dataset_kwargs, num_gpus=dist.get_world_size(), rank=dist.get_rank(), device=device)
            if dist.get_rank() == 0:
                metric_main.report_metric(result_dict, run_dir=run_dir, snapshot_pkl='network-snapshot-latest.pkl')

    # Done.
    dist.print0()
    dist.print0('Exiting...')

#----------------------------------------------------------------------------
