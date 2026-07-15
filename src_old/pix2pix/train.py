#==============================================================================
import argparse
import json
from pathlib import Path
from setproctitle import setproctitle
import time
from copy import deepcopy

import os
import pandas as pd

import torch
from torch.utils.data import DataLoader, RandomSampler
import matplotlib.pyplot as plt
from tqdm import tqdm

from pipeline import PairedDataset
from networks import Generator, Discriminator, Loss, weights_init
#==============================================================================
def moving_average(model, model_test, beta=0.999):
    source_params = model.parameters()
    target_params = model_test.parameters()

    with torch.no_grad():
        for param, param_test in zip(source_params, target_params):
            param_test.data = param_test.data * beta + param.data * (1.0 - beta)
#==============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_ids', type=str, default='0')
    #----------------------------------------------------------------------
    # dataset
    parser.add_argument('--data_root', type=str, default='/userhome/jeon_mg/prep_data/prep_norm')
    parser.add_argument('--dataset_name', type=str, default='paired_171_193')
    parser.add_argument('--input_cols', type=str, default='aia171')
    parser.add_argument('--output_cols', type=str, default='aia193')
    #----------------------------------------------------------------------
    # network
    parser.add_argument('--input_ch', type=int, default=1)
    parser.add_argument('--target_ch', type=int, default=1)
    parser.add_argument('--n_gf', type=int, default=64)
    parser.add_argument('--n_downsample', type=int, default=4)
    parser.add_argument('--n_bottleneck', type=int, default=9)
    parser.add_argument('--pad_type', type=str, default='ReplicationPad2d')
    parser.add_argument('--norm_type', type=str, default='InstanceNorm2d')
    parser.add_argument('--act_type', type=str, default='Mish')
    parser.add_argument('--mode', type=str, default='nearest')
    parser.add_argument('--n_D', type=int, default=4)
    parser.add_argument('--n_df', type=int, default=64)
    #----------------------------------------------------------------------
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--eps', type=float, default=1e-8)
    #----------------------------------------------------------------------
    parser.add_argument('--train_data_csv', type=str, default='/userhome/jeon_mg/prep_data/dataset_train.csv')
    parser.add_argument('--val_data_csv', type=str, default='/userhome/jeon_mg/prep_data/dataset_val.csv')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--total_iters', type=int, default=1_000_000)
    parser.add_argument('--save_freq', type=int, default=10000)
    parser.add_argument('--print_freq', type=int, default=10000)
    parser.add_argument('--val_freq', type=int, default=10000)
    parser.add_argument('--resume_iter', type=int, default=0)
    #----------------------------------------------------------------------
    # loss
    parser.add_argument('--lambda_LSGAN', type=float, default=2.0)
    parser.add_argument('--lambda_FM', type=float, default=10.0)
    parser.add_argument('--lambda_CC', type=float, default=5.0)
    parser.add_argument('--n_CC', type=int, default=4)
    parser.add_argument('--ccc', action='store_true', default=True)
    #----------------------------------------------------------------------
    # optimizer
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    #----------------------------------------------------------------------
    opt = parser.parse_args()
    opt.checkpoint_dir = Path('./checkpoints') / opt.dataset_name
    opt.model_dir = opt.checkpoint_dir / 'Model'
    opt.image_dir = opt.checkpoint_dir / 'Image' / 'Train'   
    opt.model_dir.mkdir(parents=True, exist_ok=True)
    opt.image_dir.mkdir(parents=True, exist_ok=True)
    #--------------------------------------------------------------------------
    opt_dict = {k: str(v) if isinstance(v, Path) else v for k, v in vars(opt).items()}
    with open(opt.checkpoint_dir / 'train_options.json', 'w') as f:
        json.dump(opt_dict, f, indent=4)
    #--------------------------------------------------------------------------
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_ids
    # torch.backends.cudnn.benchmark = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #--------------------------------------------------------------------------
    train_dataset = PairedDataset(
        data_root=opt.data_root,
        data_csv=opt.train_data_csv,
        input_cols=opt.input_cols,
        target_cols=opt.output_cols,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        num_workers=opt.num_workers,
        sampler=RandomSampler(train_dataset, replacement=True, num_samples=opt.total_iters * opt.batch_size),
        pin_memory=True,
        drop_last=True,
    )

    val_dataset = PairedDataset(
        data_root=opt.data_root,
        data_csv=opt.val_data_csv,
        input_cols=opt.input_cols,
        target_cols=opt.output_cols,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=1,
        num_workers=opt.num_workers,
        shuffle=False,
    )
    #--------------------------------------------------------------------------
    G = Generator(opt).apply(weights_init).to(device)
    D = Discriminator(opt).apply(weights_init).to(device)

    G_ema = deepcopy(G)

    G_optim = torch.optim.AdamW(G.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2), eps=opt.eps)
    D_optim = torch.optim.AdamW(D.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2), eps=opt.eps)

    if opt.resume_iter > 0:
        checkpoint_file = opt.model_dir / f'checkpoint_{opt.resume_iter}.pth'
        checkpoint = torch.load(checkpoint_file, map_location=device)
        G.load_state_dict(checkpoint['G_state_dict'])
        D.load_state_dict(checkpoint['D_state_dict'])
        G_ema.load_state_dict(checkpoint['G_ema_state_dict'])
        G_optim.load_state_dict(checkpoint['G_optim_state_dict'])
        D_optim.load_state_dict(checkpoint['D_optim_state_dict'])
        resume_step = opt.resume_iter
    else:
        resume_step = 0
    
    criterion = Loss(opt, device)
    #--------------------------------------------------------------------------
    loss_csv = opt.checkpoint_dir / 'train_loss.csv'

    if opt.resume_iter > 0 and loss_csv.exists():
        train_loss_dict = pd.read_csv(loss_csv).to_dict(orient='list')
    else:
        train_loss_dict = {
            'step': [],
            'D_loss': [],
            'G_loss': [],
            'G_loss_LSGAN': [],
            'G_loss_FM': [],
            'G_loss_CC': [],
        }
        
    
    G.train()
    D.train()
    start = time.time()
    start_step = resume_step
    train_iter = iter(train_dataloader)
    pbar = tqdm(range(resume_step + 1, opt.total_iters + 1), initial=resume_step, total=opt.total_iters)
    for step in pbar:
        inputs, target = next(train_iter)

        inputs = inputs.to(device)
        target = target.to(device)

        D_loss, G_loss, G_loss_LSGAN, G_loss_FM, G_loss_CC = criterion(D, G, inputs, target)

        D_optim.zero_grad()
        G_optim.zero_grad()

        D_loss.backward()

        for param in D.parameters():
            param.requires_grad_(False)
        G_loss.backward()
        for param in D.parameters():
            param.requires_grad_(True)

        D_optim.step()
        G_optim.step()

        moving_average(G, G_ema, beta=0.999)

        pbar.set_postfix_str(f"D_loss: {D_loss.item():.4g} | G_loss: {G_loss.item():.4g}")
        train_loss_dict['step'].append(step)
        train_loss_dict['D_loss'].append(D_loss.item())
        train_loss_dict['G_loss'].append(G_loss.item())
        train_loss_dict['G_loss_LSGAN'].append(G_loss_LSGAN.item())
        train_loss_dict['G_loss_FM'].append(G_loss_FM.item())
        train_loss_dict['G_loss_CC'].append(G_loss_CC.item())

        elapsed = time.time() - start
        done = step - start_step
        rate = elapsed / max(done, 1)
        remain = (opt.total_iters - step) * rate
        hrs = remain // 3600
        mins = (remain % 3600) // 60
        task_str = f"Train {opt.dataset_name}: {step}/{opt.total_iters}"
        eta_str = f"ETA:{int(hrs):02d}h{int(mins):02d}m"
        setproctitle(f"{task_str} | {eta_str}")

        if step % opt.print_freq == 0:
            pd.DataFrame(train_loss_dict).to_csv(opt.checkpoint_dir / 'train_loss.csv', index=False)

        if step % opt.save_freq == 0:
            torch.save({
                'step': step,
                'G_state_dict': G.state_dict(),
                'D_state_dict': D.state_dict(),
                'G_ema_state_dict': G_ema.state_dict(),
                'G_optim_state_dict': G_optim.state_dict(),
                'D_optim_state_dict': D_optim.state_dict(),
            }, opt.model_dir / f'checkpoint_{step}.pth')

        if step % opt.val_freq == 0:
            G.eval()
            D.eval()
            with torch.inference_mode():
                for val_inputs, val_target in val_dataloader:
                    val_inputs = val_inputs.to(device)
                    val_target = val_target.to(device)
                    val_fake = G(val_inputs)

                    val_target = val_target.cpu().squeeze().numpy()
                    val_fake = val_fake.cpu().squeeze().numpy()

                    real_img = opt.image_dir / 'real.png'
                    if not real_img.exists():
                        plt.imsave(real_img, val_target, cmap='gray', vmin=-1, vmax=1)
                    plt.imsave(opt.image_dir / f'{step}_fake.png', val_fake, cmap='gray', vmin=-1, vmax=1)
                    break
            G.train()
            D.train()
    #--------------------------------------------------------------------------