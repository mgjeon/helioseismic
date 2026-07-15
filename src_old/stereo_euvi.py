import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from astropy.time import Time
import astropy.units as u
from sunpy.map import Map
from astropy.visualization import ImageNormalize, AsymmetricPercentileInterval, AsinhStretch
plt.rcParams.update({
    'font.size': 15,
})

from utils import down_euvi
from utils import prep_euv
from utils import draw_disk_euv, draw_hg_euv

dim_pix = (1024, 1024)
rsun_pix = 500


parser = argparse.ArgumentParser()
parser.add_argument("--date", type=str, default="2014-03-04T00:00:00")
parser.add_argument("--root", type=str, default="pipeline")
parser.add_argument("--window", type=float, default=30)
parser.add_argument("--sc", type=str, default="STEREO-A", choices=["STEREO-A", "STEREO-B"])
args = parser.parse_args()

ROOT = Path(args.root)
ROOT_RAW = ROOT / 'raw'; ROOT_RAW.mkdir(parents=True, exist_ok=True)
ROOT_PRE = ROOT / 'prep'; ROOT_PRE.mkdir(parents=True, exist_ok=True)
ROOT_FIG = ROOT / 'figures'; ROOT_FIG.mkdir(parents=True, exist_ok=True)


date = args.date
obstime = Time.strptime(date, '%Y-%m-%dT%H:%M:%S')
obstime_str = obstime.strftime('%Y%m%d_%H%M%S')
window = args.window * u.minute
sc = args.sc
if sc == "STEREO-A":
    scf = "a"
elif sc == "STEREO-B":
    scf = "b"
else:
    raise ValueError(f"Invalid spacecraft: {sc}. Must be 'STEREO-A' or 'STEREO-B'.")

cmap_304 = plt.get_cmap('sdoaia304')
cmap_304.set_bad('black')
cmap_171 = plt.get_cmap('sdoaia171')
cmap_171.set_bad('black')
norm_raw = ImageNormalize(interval=AsymmetricPercentileInterval(1, 99.5), stretch=AsinhStretch(0.1))
norm_pre = ImageNormalize(vmin=1, vmax=200, stretch=AsinhStretch(0.01))



file_304 = down_euvi(ROOT_RAW, obstime, window=window, wavelength=304, sc=sc)
print(file_304)

plot_304 = ROOT_FIG / f'{obstime_str}_euvi_{scf}_304.png'
fig_304 = draw_disk_euv(file_304, cmap=cmap_304, norm=norm_raw)
fig_304.savefig(plot_304, dpi='figure', bbox_inches=None, pad_inches=0)
plt.close()


file_304_disk_prep = ROOT_PRE / file_304.parent.name / f'{obstime_str}_disk.fits'
file_304_disk_prep.parent.mkdir(parents=True, exist_ok=True)
file_304_hg_prep = ROOT_PRE / file_304.parent.name / f'{obstime_str}_hg.fits'
file_304_hg_prep.parent.mkdir(parents=True, exist_ok=True)

if not file_304_disk_prep.exists() or not file_304_hg_prep.exists():
    smap_disk, smap_hg = prep_euv(
        file_304,
        disk_rsun_pix=rsun_pix,
        disk_dim_pix=dim_pix,
        hg_dim_pix=(1440, 3600),
        sc=sc
    )
    smap_disk.save(file_304_disk_prep, overwrite=True)
    smap_hg.save(file_304_hg_prep, overwrite=True)
else:
    smap_disk = Map(file_304_disk_prep)
    smap_hg = Map(file_304_hg_prep)

plot_304_disk = ROOT_FIG / f'{obstime_str}_euvi_{scf}_304_disk_prep.png'
plot_304_hg = ROOT_FIG / f'{obstime_str}_euvi_{scf}_304_hg_prep.png'
fig_304_disk = draw_disk_euv(file_304_disk_prep, cmap=cmap_304, norm=norm_pre)
fig_304_disk.savefig(plot_304_disk, dpi='figure', bbox_inches=None, pad_inches=0)
fig_304_hg = draw_hg_euv(smap_hg.data, cmap=cmap_304, norm=norm_pre)
fig_304_hg.savefig(plot_304_hg, dpi=300, bbox_inches='tight', pad_inches=0)
plt.close()



file_171 = down_euvi(ROOT_RAW, obstime, window=window, wavelength=171, sc=sc)
print(file_171)

plot_171 = ROOT_FIG / f'{obstime_str}_euvi_{scf}_171.png'
fig_171 = draw_disk_euv(file_171, cmap=cmap_171, norm=norm_raw)
fig_171.savefig(plot_171, dpi='figure', bbox_inches=None, pad_inches=0)
plt.close()


file_171_disk_prep = ROOT_PRE / file_171.parent.name / f'{obstime_str}_disk.fits'
file_171_disk_prep.parent.mkdir(parents=True, exist_ok=True)
file_171_hg_prep = ROOT_PRE / file_171.parent.name / f'{obstime_str}_hg.fits'
file_171_hg_prep.parent.mkdir(parents=True, exist_ok=True)

if not file_171_disk_prep.exists() or not file_171_hg_prep.exists():
    smap_disk, smap_hg = prep_euv(
        file_171,
        disk_rsun_pix=rsun_pix,
        disk_dim_pix=dim_pix,
        hg_dim_pix=(1440, 3600),
        sc=sc
    )
    smap_disk.save(file_171_disk_prep, overwrite=True)
    smap_hg.save(file_171_hg_prep, overwrite=True)
else:
    smap_disk = Map(file_171_disk_prep)
    smap_hg = Map(file_171_hg_prep)

plot_171_disk = ROOT_FIG / f'{obstime_str}_euvi_{scf}_171_disk_prep.png'
plot_171_hg = ROOT_FIG / f'{obstime_str}_euvi_{scf}_171_hg_prep.png'
fig_171_disk = draw_disk_euv(file_171_disk_prep, cmap=cmap_171, norm=norm_pre)
fig_171_disk.savefig(plot_171_disk, dpi='figure', bbox_inches=None, pad_inches=0)
fig_171_hg = draw_hg_euv(smap_hg.data, cmap=cmap_171, norm=norm_pre)
fig_171_hg.savefig(plot_171_hg, dpi=300, bbox_inches='tight', pad_inches=0)
plt.close()