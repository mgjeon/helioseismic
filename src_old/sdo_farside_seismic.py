from pathlib import Path
from parfive import Downloader
from astropy.time import Time
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from sunpy.map.header_helper import make_heliographic_header
from sunpy.coordinates import get_horizons_coord
import astropy.units as u
from sunpy.map import Map

# http://jsoc.stanford.edu/data/farside/

ROOT = Path('pipeline/seismic'); ROOT.mkdir(parents=True, exist_ok=True)

tobs = Time('2014-03-04T00:00:00')

year = tobs.datetime.year
timestamp = tobs.datetime.strftime('%Y.%m.%d_%H:%M:%S')


file_phase = Path(f'pipeline/raw/farside_phase/PHASE_MAP_{timestamp}.fits')
file_phase_5d = Path(f'pipeline/raw/farside_phase_5d/CUM_PHASE_{timestamp}.fits')

if not file_phase.exists():
    url_phase = f'http://jsoc.stanford.edu/data/farside/Phase_Maps/{year}/PHASE_MAP_{timestamp}.fits'
    dl = Downloader(progress=True, overwrite=True)
    dl.enqueue_file(url_phase, path='pipeline/raw/farside_phase')
    dl.download()

if not file_phase_5d.exists():
    url_phase_5d = f'http://jsoc.stanford.edu/data/farside/Phase_Maps_5Day_Cum/{year}/CUM_PHASE_{timestamp}.fits'
    dl = Downloader(progress=True, overwrite=True)
    dl.enqueue_file(url_phase_5d, path='pipeline/raw/farside_phase_5d')
    dl.download()

cmap = plt.get_cmap('magma_r')
norm = plt.Normalize(vmin=-10, vmax=0)

observer = get_horizons_coord('SDO', tobs)
hg_dim_pix = (181, 361)
hg_header = make_heliographic_header(
    observer.obstime,
    observer,
    hg_dim_pix,
    frame='carrington',
    projection_code='CAR',
    map_center_longitude=180*u.deg
)
shape_out = (1024, 2048)
hg_header_cea = make_heliographic_header(
    observer.obstime,
    observer,
    shape_out,
    frame='carrington',
    projection_code='CEA',
    map_center_longitude=180*u.deg
)
lon = np.linspace(0, 2*np.pi, shape_out[1])
sinlat = np.linspace(-1, 1, shape_out[0])

with fits.open(file_phase) as hdul:
    phase_data = hdul[0].data
    phase_header = hdul[0].header
    
phase_map = Map(phase_data, hg_header)
phase_map_cea = phase_map.reproject_to(hg_header_cea)
phase_map_cea_data = np.rad2deg(phase_map_cea.data)
phase_map_cea_data[np.isnan(phase_map_cea_data)] = 0
phase_map_cea = Map(phase_map_cea_data, phase_map_cea.meta)
phase_map_cea.save(ROOT / 'phase.fits', overwrite=True)

fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111, projection=phase_map_cea)
phase_map_cea.plot(axes=ax, cmap=cmap, norm=norm)
ax.set_title('')
ax.axis('off')
fig.savefig(ROOT / 'phase_map.png', dpi=300, bbox_inches='tight', pad_inches=0)
plt.close()

fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111)
ax.pcolormesh(lon, sinlat, phase_map_cea.data, cmap=cmap, norm=norm, shading='auto')
ax.set_xticks(np.deg2rad(np.arange(0, 361, 60)))
ax.set_xticklabels(np.arange(0, 361, 60))
ax.set_yticks(np.sin(np.deg2rad(np.arange(-90, 91, 30))))
ax.set_yticklabels(np.arange(-90, 91, 30))
ax.tick_params(axis='both', length=12, which='major')
fig.tight_layout()
fig.savefig(ROOT / 'phase.png', dpi=300, bbox_inches='tight', pad_inches=0)
plt.close()

with fits.open(file_phase_5d) as hdul:
    phase_5d_data = hdul[0].data
    phase_5d_header = hdul[0].header

phase_5d_map = Map(phase_5d_data, hg_header)
phase_5d_map_cea = phase_5d_map.reproject_to(hg_header_cea)
phase_5d_map_cea_data = np.rad2deg(phase_5d_map_cea.data)
phase_5d_map_cea_data[np.isnan(phase_5d_map_cea_data)] = 0
phase_5d_map_cea = Map(phase_5d_map_cea_data, phase_5d_map_cea.meta)
phase_5d_map_cea.save(ROOT / 'phase_5d.fits', overwrite=True)

fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111, projection=phase_5d_map_cea)
phase_5d_map_cea.plot(axes=ax, cmap=cmap, norm=norm)
ax.set_title('')
ax.axis('off')
fig.savefig(ROOT / 'phase_5d_map.png', dpi=300, bbox_inches='tight', pad_inches=0)
plt.close()

fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111)
ax.pcolormesh(lon, sinlat, phase_5d_map_cea.data, cmap=cmap, norm=norm, shading='auto')
ax.set_xticks(np.deg2rad(np.arange(0, 361, 60)))
ax.set_xticklabels(np.arange(0, 361, 60))
ax.set_yticks(np.sin(np.deg2rad(np.arange(-90, 91, 30))))
ax.set_yticklabels(np.arange(-90, 91, 30))
ax.tick_params(axis='both', length=12, which='major')
fig.tight_layout()
fig.savefig(ROOT / 'phase_5d.png', dpi=300, bbox_inches='tight', pad_inches=0)
plt.close()
