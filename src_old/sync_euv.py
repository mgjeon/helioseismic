import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from astropy.time import Time
import astropy.units as u
from sunpy.map import Map
from astropy.visualization import ImageNormalize, AsymmetricPercentileInterval, AsinhStretch
import numpy as np
plt.rcParams.update({
    'font.size': 15,
})

from utils import get_mu

cmap_304 = plt.get_cmap('sdoaia304')
cmap_304.set_bad('black')
cmap_171 = plt.get_cmap('sdoaia171')
cmap_171.set_bad('black')
norm_pre = ImageNormalize(vmin=1, vmax=200, stretch=AsinhStretch(0.01))

cmap_mu = plt.get_cmap('viridis')
cmap_mu.set_bad('black')
norm_mu = plt.Normalize(vmin=-1, vmax=1)

parser = argparse.ArgumentParser()
parser.add_argument("--date", type=str, default="2014-03-04T00:00:00")
parser.add_argument("--root", type=str, default="pipeline")
args = parser.parse_args()

ROOT = Path(args.root)
ROOT_RAW = ROOT / 'raw'; ROOT_RAW.mkdir(parents=True, exist_ok=True)
ROOT_PRE = ROOT / 'prep'; ROOT_PRE.mkdir(parents=True, exist_ok=True)
ROOT_FIG = ROOT / 'figures'; ROOT_FIG.mkdir(parents=True, exist_ok=True)

date = args.date
obstime = Time.strptime(date, '%Y-%m-%dT%H:%M:%S')
obstime_str = obstime.strftime('%Y%m%d_%H%M%S')


euv_304_sdo_aia = ROOT_PRE / 'sdo_aia_304' / f'{obstime_str}_hg.fits'
euv_304_sta_euvi = ROOT_PRE / 'stereo_euvi_STEREO-A_304' / f'{obstime_str}_hg.fits'
euv_304_stb_euvi = ROOT_PRE / 'stereo_euvi_STEREO-B_304' / f'{obstime_str}_hg.fits'

smap_euv_304_sdo_aia = Map(euv_304_sdo_aia)
smap_euv_304_sta_euvi = Map(euv_304_sta_euvi)
smap_euv_304_stb_euvi = Map(euv_304_stb_euvi)

data_euv_304_sdo_aia = smap_euv_304_sdo_aia.data
data_euv_304_sta_euvi = smap_euv_304_sta_euvi.data
data_euv_304_stb_euvi = smap_euv_304_stb_euvi.data

assert data_euv_304_sdo_aia.shape == (1440, 3600)
assert data_euv_304_sta_euvi.shape == (1440, 3600)
assert data_euv_304_stb_euvi.shape == (1440, 3600)
lon = np.linspace(0, 2*np.pi, 3600)
sinlat = np.linspace(-1, 1, 1440)

mu_euv_304_sdo_aia = get_mu(smap_euv_304_sdo_aia)
mu_euv_304_sta_euvi = get_mu(smap_euv_304_sta_euvi)
mu_euv_304_stb_euvi = get_mu(smap_euv_304_stb_euvi)


fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111)
ax.pcolormesh(lon, sinlat, data_euv_304_sdo_aia, cmap=cmap_304, norm=norm_pre, shading='auto')
ax.set_xticks(np.deg2rad(np.arange(0, 361, 60)))
ax.set_xticklabels(np.arange(0, 361, 60))
ax.set_yticks(np.sin(np.deg2rad(np.arange(-90, 91, 30))))
ax.set_yticklabels(np.arange(-90, 91, 30))
ax.tick_params(axis='both', length=12, which='major')
fig.tight_layout()
fig.savefig(ROOT_FIG / f'sync_euv_304_nearside_{obstime_str}.png', dpi=300, bbox_inches='tight', pad_inches=0)
plt.close()

data_combined = np.zeros_like(data_euv_304_sta_euvi)
data_combined = np.where(mu_euv_304_sta_euvi > mu_euv_304_stb_euvi, data_euv_304_sta_euvi, data_euv_304_stb_euvi)
data_combined[mu_euv_304_sdo_aia > 0] = np.nan

fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111)
ax.pcolormesh(lon, sinlat, data_combined, cmap=cmap_304, norm=norm_pre, shading='auto')
ax.set_xticks(np.deg2rad(np.arange(0, 361, 60)))
ax.set_xticklabels(np.arange(0, 361, 60))
ax.set_yticks(np.sin(np.deg2rad(np.arange(-90, 91, 30))))
ax.set_yticklabels(np.arange(-90, 91, 30))
ax.tick_params(axis='both', length=12, which='major')
fig.tight_layout()
fig.savefig(ROOT_FIG / f'sync_euv_304_farside_{obstime_str}.png', dpi=300, bbox_inches='tight', pad_inches=0)
plt.close()


data_combined_all = data_euv_304_sdo_aia.copy()
data_combined_all = np.where((mu_euv_304_sta_euvi > mu_euv_304_sdo_aia) & (mu_euv_304_sta_euvi > mu_euv_304_stb_euvi), data_euv_304_sta_euvi, data_combined_all)
data_combined_all = np.where((mu_euv_304_stb_euvi > mu_euv_304_sdo_aia) & (mu_euv_304_stb_euvi > mu_euv_304_sta_euvi), data_euv_304_stb_euvi, data_combined_all)

fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111)
ax.pcolormesh(lon, sinlat, data_combined_all, cmap=cmap_304, norm=norm_pre, shading='auto')
ax.set_xticks(np.deg2rad(np.arange(0, 361, 60)))
ax.set_xticklabels(np.arange(0, 361, 60))
ax.set_yticks(np.sin(np.deg2rad(np.arange(-90, 91, 30))))
ax.set_yticklabels(np.arange(-90, 91, 30))
ax.tick_params(axis='both', length=12, which='major')
fig.tight_layout()
fig.savefig(ROOT_FIG / f'sync_euv_304_{obstime_str}.png', dpi=300, bbox_inches='tight', pad_inches=0)
plt.close()


data_seismic = Map('pipeline/seismic/phase.fits').data
data_seismic_5d = Map('pipeline/seismic/phase_5d.fits').data

fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111)
ax.pcolormesh(lon, sinlat, data_euv_304_sdo_aia, cmap=cmap_304, norm=norm_pre, shading='auto')
ax.pcolormesh(lon, sinlat, np.rad2deg(data_seismic), cmap='gray', vmin=-10, vmax=0, shading='auto')
ax.set_xticks(np.deg2rad(np.arange(0, 361, 60)))
ax.set_xticklabels(np.arange(0, 361, 60))
ax.set_yticks(np.sin(np.deg2rad(np.arange(-90, 91, 30))))
ax.set_yticklabels(np.arange(-90, 91, 30))
ax.tick_params(axis='both', length=12, which='major')
fig.tight_layout()
fig.savefig(ROOT_FIG / f'sync_euv_304_near_seismic_{obstime_str}.png', dpi=300, bbox_inches='tight', pad_inches=0)
plt.close()

fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111)
ax.pcolormesh(lon, sinlat, data_euv_304_sdo_aia, cmap=cmap_304, norm=norm_pre, shading='auto')
ax.pcolormesh(lon, sinlat, np.rad2deg(data_seismic_5d), cmap='gray', vmin=-10, vmax=0, shading='auto')
ax.set_xticks(np.deg2rad(np.arange(0, 361, 60)))
ax.set_xticklabels(np.arange(0, 361, 60))
ax.set_yticks(np.sin(np.deg2rad(np.arange(-90, 91, 30))))
ax.set_yticklabels(np.arange(-90, 91, 30))
ax.tick_params(axis='both', length=12, which='major')
fig.tight_layout()
fig.savefig(ROOT_FIG / f'sync_euv_304_near_seismic_5d_{obstime_str}.png', dpi=300, bbox_inches='tight', pad_inches=0)
plt.close()