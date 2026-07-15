import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from astropy.time import Time
import astropy.units as u
from sunpy.map import Map, make_fitswcs_header, make_heliographic_header
from astropy.visualization import ImageNormalize, AsymmetricPercentileInterval, AsinhStretch
import numpy as np
from sunpy.sun import constants
from reproject import reproject_interp
from reproject.mosaicking import reproject_and_coadd
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord

from utils import get_mu, stereo_cor, get_disk_mask

def load_euv(smap, med_val=10):
    data = smap.data
    data[data < 0] = 0
    med_ratio = med_val/np.nanmedian(data[get_disk_mask(smap)])
    data = data*med_ratio
    return Map(data, smap.meta)

plt.rcParams.update({
    'font.size': 15,
})

parser = argparse.ArgumentParser()
parser.add_argument("--date", type=str, default="2014-03-04T00:00:00")
parser.add_argument("--root", type=str, default="pipeline")
args = parser.parse_args()

ROOT = Path(args.root)
ROOT_RAW = ROOT / 'raw'; ROOT_RAW.mkdir(parents=True, exist_ok=True)

date = args.date
obstime = Time.strptime(date, '%Y-%m-%dT%H:%M:%S')
obstime_str = obstime.strftime('%Y%m%d_%H%M%S')

cmap = plt.get_cmap('sdoaia304')
cmap.set_bad('black')
norm = ImageNormalize(vmin=1, vmax=200, stretch=AsinhStretch(0.01))

file_sdo_304 = ROOT_RAW / 'sdo_aia_304' / f'{obstime_str}.fits'
file_sta_304 = ROOT_RAW / 'stereo_euvi_STEREO-A_304' / f'{obstime_str}.fits'
file_stb_304 = ROOT_RAW / 'stereo_euvi_STEREO-B_304' / f'{obstime_str}.fits'

ROOT_FIG = ROOT / 'sync_euv'; ROOT_FIG.mkdir(parents=True, exist_ok=True)

med_val = 10
rsun_ref = constants.radius.to(u.m).value

sdo_304 = Map(file_sdo_304)
sdo_304.meta['rsun_ref'] = rsun_ref
sdo_304 = load_euv(sdo_304, med_val=med_val)

sta_304 = Map(file_sta_304)
sta_304.meta['rsun_ref'] = rsun_ref
sta_304 = stereo_cor(sta_304)
sta_304 = load_euv(sta_304, med_val=med_val)

stb_304 = Map(file_stb_304)
stb_304.meta['rsun_ref'] = rsun_ref
stb_304 = stereo_cor(stb_304)
stb_304 = load_euv(stb_304, med_val=med_val)

mu_sdo_304 = get_mu(sdo_304)
mu_sta_304 = get_mu(sta_304)
mu_stb_304 = get_mu(stb_304)

mu_sdo_304[np.isnan(mu_sdo_304)] = 0
mu_sta_304[np.isnan(mu_sta_304)] = 0
mu_stb_304[np.isnan(mu_stb_304)] = 0

mu_sdo_304[mu_sdo_304 < 0] = 0
mu_sta_304[mu_sta_304 < 0] = 0
mu_stb_304[mu_stb_304 < 0] = 0

p = 3
weight_sdo_304 = mu_sdo_304**p
weight_sta_304 = mu_sta_304**p
weight_stb_304 = mu_stb_304**p
weights = (weight_sdo_304, weight_sta_304, weight_stb_304)

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection=sdo_304)
sdo_304.plot(axes=ax, cmap=cmap, norm=norm)
ax.axis('off')
ax.set_title('')
fig.savefig(ROOT_FIG / 'sdo_304.png', dpi=300, bbox_inches='tight', pad_inches=0)
plt.close()

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111)
ax.imshow(weight_sdo_304, cmap='viridis', origin='lower', norm=plt.Normalize(vmin=0, vmax=1))
ax.axis('off')
ax.set_title('')
fig.savefig(ROOT_FIG / 'sdo_304_weight.png', dpi=300, bbox_inches='tight', pad_inches=0)
plt.close()

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection=sta_304)
sta_304.plot(axes=ax, cmap=cmap, norm=norm)
ax.axis('off')
ax.set_title('')
fig.savefig(ROOT_FIG / 'sta_304.png', dpi=300, bbox_inches='tight', pad_inches=0)
plt.close()

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111)
ax.imshow(weight_sta_304, cmap='viridis', origin='lower', norm=plt.Normalize(vmin=0, vmax=1))
ax.axis('off')
ax.set_title('')
fig.savefig(ROOT_FIG / 'sta_304_weight.png', dpi=300, bbox_inches='tight', pad_inches=0)
plt.close()

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection=stb_304)
stb_304.plot(axes=ax, cmap=cmap, norm=norm)
ax.axis('off')
ax.set_title('')
fig.savefig(ROOT_FIG / 'stb_304.png', dpi=300, bbox_inches='tight', pad_inches=0)
plt.close()

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111)
ax.imshow(weight_stb_304, cmap='viridis', origin='lower', norm=plt.Normalize(vmin=0, vmax=1))
ax.axis('off')
ax.set_title('')
fig.savefig(ROOT_FIG / 'stb_304_weight.png', dpi=300, bbox_inches='tight', pad_inches=0)
plt.close()

maps = (sdo_304, sta_304, stb_304)

observer = sdo_304.observer_coordinate
shape_out = (1024, 2048)
# frame_out = SkyCoord(
#     180, 0, unit=u.deg,
#     frame="heliographic_carrington",
#     obstime=observer.obstime,
#     observer=observer
# )
# scale = [
#     360 / int(shape_out[1]),
#     (180 / np.pi) / (int(shape_out[0]) / 2)
# ] * u.deg / u.pix
# projection_code = "CEA"
# header = make_fitswcs_header(
#     shape_out,
#     frame_out,
#     scale=scale,
#     projection_code=projection_code
# )
header = make_heliographic_header(
    observer.obstime,
    observer,
    shape_out,
    frame='carrington',
    projection_code='CEA',
    map_center_longitude=180*u.deg
)
out_wcs = WCS(header)

sdo_map = sdo_304.reproject_to(out_wcs)
fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111, projection=sdo_map)
sdo_map.plot(axes=ax, cmap=cmap, norm=norm)
ax.set_title('')
ax.axis('off')
fig.savefig(ROOT_FIG / f'sync_euv_304_sdo_{obstime_str}.png', dpi=300, bbox_inches='tight', pad_inches=0)
plt.close()

array, footprint = reproject_and_coadd(
    maps,
    out_wcs,
    shape_out=shape_out,
    reproject_function=reproject_interp,
    match_background=True,
    background_reference=0,
)

outmap = Map(array, header)

fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111, projection=outmap)
outmap.plot(axes=ax, cmap=cmap, norm=norm)
ax.set_title('')
ax.axis('off')
fig.savefig(ROOT_FIG / f'sync_euv_304_sdo_st_{obstime_str}.png', dpi=300, bbox_inches='tight', pad_inches=0)
plt.close()

array, footprint = reproject_and_coadd(
    maps,
    out_wcs,
    shape_out=shape_out,
    input_weights=weights,
    reproject_function=reproject_interp,
)

outmap = Map(array, header)

fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111, projection=outmap)
outmap.plot(axes=ax, cmap=cmap, norm=norm)
ax.set_title('')
ax.axis('off')
fig.savefig(ROOT_FIG / f'sync_euv_304_sdo_st_weight1_{obstime_str}.png', dpi=300, bbox_inches='tight', pad_inches=0)
plt.close()

array, footprint = reproject_and_coadd(
    maps,
    out_wcs,
    shape_out=shape_out,
    input_weights=weights,
    reproject_function=reproject_interp,
    match_background=True,
    background_reference=0,
)

outmap = Map(array, header)

fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111, projection=outmap)
outmap.plot(axes=ax, cmap=cmap, norm=norm)
ax.set_title('')
ax.axis('off')
fig.savefig(ROOT_FIG / f'sync_euv_304_sdo_st_weight2_{obstime_str}.png', dpi=300, bbox_inches='tight', pad_inches=0)
plt.close()