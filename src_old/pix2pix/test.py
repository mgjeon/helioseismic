#==============================================================================
import argparse
import os
from pathlib import Path
from tqdm import tqdm
from scipy.stats import pearsonr
from skimage.metrics import normalized_root_mse
import pandas as pd
import numpy as np

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from pipeline import PairedDataset, inverse_transform_aia
from networks import Generator
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
    #----------------------------------------------------------------------
    parser.add_argument('--test_data_csv', type=str, default='/userhome/jeon_mg/prep_data/dataset_test.csv')
    parser.add_argument('--test_iter', type=int, default=1_000_000)
    parser.add_argument('--disk_mask', action='store_true', default=False)
    #----------------------------------------------------------------------
    opt = parser.parse_args()
    opt.checkpoint_dir = Path('./checkpoints') / opt.dataset_name
    opt.model_dir = opt.checkpoint_dir / 'Model'
    if opt.disk_mask:
        opt.image_dir = opt.checkpoint_dir / 'Image'  / 'Test_disk_mask' / str(opt.test_iter)
        # circle disk mask with radius=Rsun_pix
        Rsun_pix = 512/1.1
        y, x = torch.meshgrid(torch.arange(1024), torch.arange(1024), indexing='ij')
        center = (512, 512)
        mask = ((x - center[0])**2 + (y - center[1])**2 <= Rsun_pix**2).numpy()
    else:
        opt.image_dir = opt.checkpoint_dir / 'Image'  / 'Test' / str(opt.test_iter)
    opt.image_dir_inp = opt.image_dir / 'Input'; opt.image_dir_inp.mkdir(parents=True, exist_ok=True)
    opt.image_dir_oup = opt.image_dir / 'Output'; opt.image_dir_oup.mkdir(parents=True, exist_ok=True)
    #--------------------------------------------------------------------------
    # torch.backends.cudnn.benchmark = False
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_ids
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #--------------------------------------------------------------------------
    test_dataset = PairedDataset(
        data_root=opt.data_root,
        data_csv=opt.test_data_csv,
        input_cols=opt.input_cols,
        target_cols=opt.output_cols,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=opt.num_workers,
        shuffle=False,
    )
    #--------------------------------------------------------------------------
    G = Generator(opt).to(device)

    checkpoint_file = opt.model_dir / f'checkpoint_{opt.test_iter}.pth'
    checkpoint = torch.load(checkpoint_file, map_location=device)
    G.load_state_dict(checkpoint['G_ema_state_dict'])
    print(f"G: loaded from {checkpoint_file}")
    #--------------------------------------------------------------------------
    G.eval()
    nrmse_list = []
    pcc_list = []
    with torch.inference_mode():
        for step, (inputs, real) in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
            # if step > 5:
            #     break  # for debug
            
            inputs = inputs.to(device)
            real = real.to(device)
            fake = G(inputs)

            real = real.cpu().squeeze().numpy()
            fake = fake.cpu().squeeze().numpy()

            if step < 5:  # Save the first 5 results as images for visual inspection
                plt.imsave(opt.image_dir_inp / f'{step}_inp0.png', inputs[0, 0].cpu().squeeze().numpy(), cmap='gray', vmin=-1, vmax=1)
                plt.imsave(opt.image_dir_oup / f'{step}_real.png', real, cmap='gray', vmin=-1, vmax=1)
                plt.imsave(opt.image_dir_oup / f'{step}_fake.png', fake, cmap='gray', vmin=-1, vmax=1)

            real = inverse_transform_aia(real)
            fake = inverse_transform_aia(fake)

            if opt.disk_mask:
                real = real[mask].flatten()
                fake = fake[mask].flatten()

                rmse = np.sqrt(np.mean((real - fake)**2))
                denom = np.sqrt(np.mean(real**2))
                nrmse = rmse / denom
            else:
                nrmse = normalized_root_mse(real, fake, normalization='euclidean')
                
            pcc = pearsonr(real.flatten(), fake.flatten())[0]
            nrmse_list.append(nrmse)
            pcc_list.append(pcc)
    #--------------------------------------------------------------------------
    # Save results to CSV
    results_df = pd.DataFrame({
        'NRMSE': nrmse_list,
        'PCC': pcc_list
    })
    results_df.to_csv(opt.image_dir / 'test_results.csv', index=False)
    print(f"Average NRMSE: {results_df['NRMSE'].mean():.4f}")
    print(f"Average PCC  : {results_df['PCC'].mean():.4f}")