
import drms
import numpy as np
import pandas as pd
import astropy.units as u
from astropy.time import Time
from astropy.io import fits
from urllib import request
from tqdm import tqdm
from urllib.request import urlopen
from urllib.error import HTTPError, URLError
from bs4 import BeautifulSoup
from sunpy.map import Map
import re
from datetime import datetime
from sunpy.net import Fido, attrs as a
from pathlib import Path
import sunpy_soar
import matplotlib.pyplot as plt
from sunpy.sun import constants
from astropy.coordinates import SkyCoord
import astropy.units as u
from sunpy.map.header_helper import make_fitswcs_header
from astropy.io import fits
from sunpy.map import Map
import matplotlib.pyplot as plt
from sunpy.coordinates.sun import carrington_rotation_number
from sunpy.sun import constants
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
from sunpy.map.header_helper import make_fitswcs_header
from sunpy.map.maputils import all_coordinates_from_map, product, solar_angular_radius
from copy import deepcopy
from sunpy.map.header_helper import make_heliographic_header
from sunpy.coordinates import frames, get_earth
from sunpy.coordinates.utils import get_heliocentric_angle
from astropy.time import Time
from matplotlib.colors import ListedColormap, BoundaryNorm
from datetime import timedelta
from skimage.transform import resize
from sunpy.physics.differential_rotation import solar_rotate_coordinate
import pandas as pd
import urllib.request
from pathlib import Path
from parfive import Downloader
import warnings; warnings.filterwarnings("ignore")


def get_tobs(tobs_list, format='%Y-%m-%dT%H:%M:%S', scale='utc'):
    try:
        tobs = Time.strptime(tobs_list, format_string=format, scale=scale)
    except Exception as e:
        print(f"Error parsing time: {e}")
        tobs = []
        for t in tobs_list:
            try:
                tobs.append(Time.strptime(t, format_string=format, scale=scale))
            except Exception as e:
                print(f"Error parsing time '{t}': {e}")
                tobs.append(Time('1970-01-01T00:00:00', format='isot', scale='utc'))
        tobs = Time(tobs)
    return tobs


def get_fits(key, seg, i=1, debug=False):
    if 'DATE__OBS' in key.keys():
        key['DATE_OBS'] = key['DATE__OBS']
    f = fits.open('http://jsoc.stanford.edu' + seg, cache=False, do_not_scale_image_data=True)
    if debug:
        return f.info()
    for k, v in key.items():
        if pd.isna(v):
            continue
        try:
            f[i].header[k] = v
        except Exception as e:
            print(f"Error setting header key '{k}': {e}")
            pass
    return f


def down_m_720s(ROOT, obstime, window=30*u.minute):
    obstime_str = obstime.strftime('%Y%m%d_%H%M%S')

    ROOT_RAW = ROOT / 'hmi_m_720s'
    ROOT_RAW.mkdir(parents=True, exist_ok=True)
    file = ROOT_RAW / f'{obstime_str}.fits'

    if file.exists():
        return file

    try:
        time_st = (obstime - window).strftime('%Y.%m.%d_%H:%M:%S')
        time_ed = (obstime + window).strftime('%Y.%m.%d_%H:%M:%S')
        time_range = time_st + '-' + time_ed

        c = drms.Client()

        keys, segs = c.query(
            f'hmi.m_720s[{time_range}]',
            key = drms.JsocInfoConstants.all,
            seg = 'magnetogram',
        )
        print(keys[['T_REC', 'T_OBS', 'QUALITY']])

        tobs = get_tobs(keys['T_OBS'].to_numpy().astype(str), format='%Y.%m.%d_%H:%M:%S_TAI', scale='tai').utc
        indices = list(keys[keys['QUALITY'] == 0].index)
        if len(indices) == 0:
            return None
        idx = indices[np.abs(tobs[indices] - obstime).argmin()]
        key = keys.iloc[idx].to_dict()
        seg = segs.iloc[idx].to_dict()

        f = get_fits(key, seg['magnetogram'], i=1)
        f.writeto(file, overwrite=True, output_verify='silentfix')
        return file
    
    except Exception as e:
        print(f"Error downloading {obstime_str}: {e}")
        return None


def draw_disk_magnetogram(file):
    smap = Map(file)
    dpi = 300
    figsize = (smap.data.shape[1]/dpi, smap.data.shape[0]/dpi)

    cmap = plt.get_cmap('gray')
    cmap.set_bad('black')
    norm = plt.Normalize(vmin=-500, vmax=500)

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_axes([0,0,1,1], projection=smap)
    smap.plot(axes=ax, title='', cmap=cmap, norm=norm)
    # smap.draw_grid(axes=ax, grid_spacing=15*u.deg, color='white', linewidth=1)
    # smap.draw_limb(axes=ax, color='white', linewidth=1)
    ax.grid(False)
    ax.coords[0].set_axislabel('')
    ax.coords[0].set_ticks_visible(False)
    ax.coords[0].set_ticklabel_visible(False)
    ax.coords[1].set_axislabel('')
    ax.coords[1].set_ticks_visible(False)
    ax.coords[1].set_ticklabel_visible(False)
    fig.tight_layout()
    return fig


def get_disk_header(observer, dim_pix, rsun_pix):
    ref_coord = SkyCoord(0*u.arcsec, 0*u.arcsec, frame="helioprojective", observer=observer)

    sol_radius = constants.radius
    distance = observer.radius
    rsun_obs = np.arcsin(sol_radius / distance).to(u.arcsec)
    disk_scale = [rsun_obs.value / rsun_pix, rsun_obs.value / rsun_pix] * (u.arcsec / u.pixel)

    disk_header = make_fitswcs_header(dim_pix, ref_coord, scale=disk_scale)
    return disk_header


def get_disk_map( 
        in_map, disk_header,
        cutoff = 1, fill_value = 0, remove_offlimb = True,
        med_val = False, min_val = False, max_val = False, 
        nan_val = False, nan_first_val = False,
        algorithm = 'interpolation',
        order = 'bilinear'
     ):
    #------------------------------
    in_map.meta['rsun_ref'] = constants.radius.to(u.m).value
    #------------------------------
    if (nan_first_val is not False):
        in_data = deepcopy(in_map.data)
        in_data[np.isnan(in_data)] = float(nan_first_val)
        in_map = deepcopy(Map(in_data, in_map.meta))
    #------------------------------
    if remove_offlimb:
        disk_mask = get_disk_mask(in_map, cutoff=cutoff)
        in_data = deepcopy(in_map.data)
        in_data[~disk_mask] = fill_value
        in_map = deepcopy(Map(in_data, in_map.meta))
    #------------------------------
    if (med_val is not False) or (min_val is not False) or (max_val is not False):
        disk_mask = get_disk_mask(in_map, cutoff=cutoff)
        in_data = deepcopy(in_map.data)
        if (med_val is not False):
            vmed_ratio = float(med_val)/np.nanmedian(in_map.data[disk_mask])
            in_data = deepcopy(in_data*vmed_ratio)
        if (min_val is not False):
            in_data[in_data<float(min_val)] = float(min_val)
        if (max_val is not False):
            in_data[in_data>float(max_val)] = float(max_val)
        in_map = deepcopy(Map(in_data, in_map.meta))  
    #------------------------------
    out_map = in_map.reproject_to(disk_header, algorithm=algorithm, order=order)
    #------------------------------
    if (nan_val is not False):
        out_data = deepcopy(out_map.data)
        out_data[np.isnan(out_data)] = float(nan_val)
        out_map = deepcopy(Map(out_data, out_map.meta))
    #------------------------------
    return out_map


def get_disk_mask(smap, cutoff=1):
    in_coord = all_coordinates_from_map(smap)
    disk_mask = np.arccos(np.cos(in_coord.Tx) * np.cos(in_coord.Ty)) <= cutoff*solar_angular_radius(in_coord)
    return disk_mask


def set_offlimb(in_map, cutoff=1, val=0):
    disk_mask = get_disk_mask(in_map, cutoff=cutoff)
    in_data = deepcopy(in_map.data)
    in_data[~disk_mask] = val
    out_map = deepcopy(Map(in_data, in_map.meta))
    return out_map


def prep_magnetogram(file, disk_rsun_pix, disk_dim_pix, hg_dim_pix):
    smap = Map(file)
    smap.meta['rsun_ref'] = constants.radius.to(u.m).value
    observer = smap.observer_coordinate

    disk_header = get_disk_header(
        observer, 
        dim_pix=disk_dim_pix,
        rsun_pix=disk_rsun_pix
    )
    smap_disk = get_disk_map(
        smap, 
        disk_header, 
        cutoff=1,
        remove_offlimb=True, 
        fill_value=np.nan
    )

    hg_header = make_heliographic_header(
        observer.obstime,
        observer,
        hg_dim_pix,
        frame='carrington',
        projection_code='CEA',
        map_center_longitude=180*u.deg
    )
    smap_hg = get_heliographic_map(
        smap,
        hg_header,
        cutoff=1,
        remove_offlimb=True,
        fill_value=np.nan,
    )

    return smap_disk, smap_hg


def get_heliographic_map(
        in_map, hg_header,
        cutoff=1, fill_value=0, remove_offlimb=True,
        med_val=False, min_val=False, max_val=False, 
        nan_val=False, nan_first_val = False,
        algorithm='interpolation',
        order='bilinear'
        ): 
    #------------------------------
    in_map.meta['rsun_ref'] = constants.radius.to(u.m).value
    #------------------------------
    if (nan_first_val is not False):
        in_data = deepcopy(in_map.data)
        in_data[np.isnan(in_data)] = float(nan_first_val)
        in_map = deepcopy(Map(in_data, in_map.meta))
    #------------------------------
    if remove_offlimb:
        disk_mask = get_disk_mask(in_map, cutoff=cutoff)
        in_data = deepcopy(in_map.data)
        in_data[~disk_mask] = fill_value
        in_map = deepcopy(Map(in_data, in_map.meta))
    #------------------------------
    if (med_val is not False) or (min_val is not False) or (max_val is not False):
        disk_mask = get_disk_mask(in_map, cutoff=cutoff)
        in_data = deepcopy(in_map.data)
        if (med_val is not False):
            vmed_ratio = float(med_val)/np.nanmedian(in_map.data[disk_mask])
            in_data = deepcopy(in_data*vmed_ratio)
        if (min_val is not False):
            in_data[in_data<float(min_val)] = float(min_val)
        if (max_val is not False):
            in_data[in_data>float(max_val)] = float(max_val)
        in_map = deepcopy(Map(in_data, in_map.meta))  
    #------------------------------
    out_map = in_map.reproject_to(hg_header, algorithm=algorithm, order=order)
    #------------------------------
    if (nan_val is not False):
        out_data = deepcopy(out_map.data)
        out_data[np.isnan(out_data)] = float(nan_val)
        out_map = deepcopy(Map(out_data, out_map.meta))
    #------------------------------
    return out_map


def down_hmi_sync(ROOT, obstime, window=1*u.day):
    obstime_str = obstime.strftime('%Y%m%d_%H%M%S')

    ROOT_RAW = ROOT / 'hmi_sync_map'
    ROOT_RAW.mkdir(parents=True, exist_ok=True)
    file = ROOT_RAW / f'{obstime_str}.fits'

    if file.exists():
        return file
    
    try:
        tst = (obstime - window).strftime('%Y.%m.%d_%H:%M:%S')
        tnd = (obstime + window).strftime('%Y.%m.%d_%H:%M:%S')
        tr  = f'{tst}-{tnd}'

        c = drms.Client()
        keys, segs = c.query(
            f'hmi.mrdailysynframe_polfil_720s[{tr}]',
            key = drms.JsocInfoConstants.all,
            seg = 'Mr_polfil',
        )
        print(keys[['T_REC', 'T_OBS']])

        indices = list(keys.index)
        if len(indices) == 0:
            return None

        tobs_list = keys['T_OBS'].to_numpy().astype(str)
        tobs = get_tobs(tobs_list, format='%Y.%m.%d_%H:%M:%S_TAI', scale='tai').utc
        idx = indices[np.argmin(np.abs(tobs - obstime))]
        key = keys.loc[idx].to_dict()
        seg = segs.loc[idx].to_dict()

        f = get_fits(key, seg['Mr_polfil'], i=1)
        f.writeto(file, overwrite=True)
        return file
    
    except Exception as e:
        print(f"Error downloading HMI synchronic map for time {obstime_str}: {e}")
        return None
    

def down_synoptic_map(ROOT, cr_num):
    ROOT_RAW = ROOT / 'synoptic_map'
    ROOT_RAW.mkdir(parents=True, exist_ok=True)
    file = ROOT_RAW / f'cr_{cr_num}.fits'

    if file.exists():
        return file
    
    try:
        c = drms.Client()

        if cr_num >= 2097:
            # HMI
            keys, segs = c.query(
                f'hmi.synoptic_mr_720s[{cr_num}]',
                key = drms.JsocInfoConstants.all,
                seg = 'synopMr',
            )
            key = keys.iloc[0].to_dict()
            seg = segs.iloc[0].to_dict()
            f = get_fits(key, seg['synopMr'], i=0)
        else:
            # MDI
            keys, segs = c.query(
                f'mdi.synoptic_mr_96m[{cr_num}]',
                key = drms.JsocInfoConstants.all,
                seg = 'data',
            )
            key = keys.iloc[0].to_dict()
            seg = segs.iloc[0].to_dict()
            f = get_fits(key, seg['data'], i=0)

        f.writeto(file, overwrite=True, output_verify='silentfix')
        return file
    except Exception as e:
        print(f"Error downloading synoptic map for Carrington rotation {cr_num}: {e}")
        return None
    

def draw_hg_magnetogram(data):
    ny, nx = data.shape
    lon = np.linspace(0, 2*np.pi, nx)
    sinlat = np.linspace(-1, 1, ny)

    cmap = plt.get_cmap('gray')
    cmap.set_bad(color='black')
    norm = plt.Normalize(vmin=-500, vmax=500)

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    # ax.imshow(data, origin='lower', extent=[lon[0], lon[-1], sinlat[0], sinlat[-1]], cmap=cmap, norm=norm)
    ax.pcolormesh(lon, sinlat, data, cmap=cmap, norm=norm, shading='auto')
    # ax.set_xlabel('Carrington Longitude [deg]')
    ax.set_xticks(np.deg2rad(np.arange(0, 361, 60)))
    ax.set_xticklabels(np.arange(0, 361, 60))
    # ax.set_ylabel('Latitude [deg]')
    ax.set_yticks(np.sin(np.deg2rad(np.arange(-90, 91, 30))))
    ax.set_yticklabels(np.arange(-90, 91, 30))
    ax.tick_params(axis='both', length=12, which='major')
    fig.tight_layout()
    return fig


def draw_hg_magnetogram_car(data):
    ny, nx = data.shape
    lon = np.linspace(0, 2*np.pi, nx)
    lat = np.linspace(-np.pi/2, np.pi/2, ny)

    cmap = plt.get_cmap('gray')
    cmap.set_bad(color='black')
    norm = plt.Normalize(vmin=-100, vmax=100)

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    # ax.imshow(data, origin='lower', extent=[lon[0], lon[-1], sinlat[0], sinlat[-1]], cmap=cmap, norm=norm)
    ax.pcolormesh(lon, lat, data, cmap=cmap, norm=norm, shading='auto')
    # ax.set_xlabel('Carrington Longitude [deg]')
    ax.set_xticks(np.deg2rad(np.arange(0, 361, 60)))
    ax.set_xticklabels(np.arange(0, 361, 60))
    # ax.set_ylabel('Latitude [deg]')
    ax.set_yticks(np.deg2rad(np.arange(-90, 91, 30)))
    ax.set_yticklabels(np.arange(-90, 91, 30))
    ax.tick_params(axis='both', length=12, which='major')
    fig.tight_layout()
    return fig


def prep_sync_map(data, obstime):
    crn = carrington_rotation_number(obstime)
    cr_frac = crn - int(crn)
    cr_lon = (1-cr_frac) * 360

    ny, nx = data.shape
    x_syn = np.linspace(0, 360, nx)
    x_car = (x_syn - 60.0 + cr_lon) % 360.0
    idx = np.argsort(x_car)
    data_car = data[:, idx]
    return data_car


def down_phi(ROOT, obstime, window=30*u.minute, product='phi-fdt-blos', level='L2'):
    obstime_str = obstime.strftime("%Y%m%d_%H%M%S")

    ROOT_RAW = ROOT / 'solo_phi_fdt'
    ROOT_RAW.mkdir(parents=True, exist_ok=True)

    file = ROOT_RAW / f'{obstime_str}.fits'
    if file.exists():
        return file

    try:
        time_st = obstime - window
        time_ed = obstime + window
        time_range = a.Time(time_st, time_ed)

        search = Fido.search(
            time_range,
            a.Instrument('PHI'), 
            a.Level(level),
            a.soar.Product(product),
        )
        print(search)

        tobs = get_tobs(np.array(search['soar']['Start time']).astype(str), format='%Y-%m-%d %H:%M:%S.%f')
        idx = np.abs(tobs - obstime).argmin()
        f = search['soar'][idx]

        res = Path(Fido.fetch(f, path=ROOT_RAW)[0])
        res.replace(file)

        return file
    except Exception as e:
        print(f"Error downloading PHI {product} data for time {obstime_str}: {e}")
        return None
    

def down_aia_euv(ROOT_RAW, obstime, window=30*u.minute, wavelength=304):
    obstime_str = obstime.strftime('%Y%m%d_%H%M%S')

    file = ROOT_RAW / f'{obstime_str}.fits'

    if file.exists():
        return file
    
    try:
        time_st = (obstime - window).strftime('%Y.%m.%d_%H:%M:%S')
        time_ed = (obstime + window).strftime('%Y.%m.%d_%H:%M:%S')
        time_range = time_st + '-' + time_ed

        c = drms.Client()

        keys, segs = c.query(
            f'aia.lev1_euv_12s[{time_range}][{wavelength}]',
            key = drms.JsocInfoConstants.all,
            seg = 'image',
        )
        print(keys[['T_REC', 'T_OBS', 'QUALITY']])
        
        tobs = get_tobs(keys['T_OBS'].to_numpy().astype(str), format='%Y-%m-%dT%H:%M:%S.%fZ', scale='utc').utc
        indices = list(keys[keys['QUALITY'] == 0].index)
        if len(indices) == 0:
            return None
        idx = indices[np.abs(tobs[indices] - obstime).argmin()]
        key = keys.iloc[idx].to_dict()
        seg = segs.iloc[idx].to_dict()

        f = get_fits(key, seg['image'], i=1)
        f.writeto(file, overwrite=True, output_verify='silentfix')

        return file
    except Exception as e:
        print(f"Error downloading AIA {wavelength} data for time {obstime_str}: {e}")
        return None
    

def draw_disk_euv(file, cmap, norm):
    smap = Map(file)
    dpi = 300
    figsize = (smap.data.shape[1]/dpi, smap.data.shape[0]/dpi)

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_axes([0,0,1,1], projection=smap)
    smap.plot(axes=ax, title='', cmap=cmap, norm=norm)
    # smap.draw_grid(axes=ax, grid_spacing=15*u.deg, color='white', linewidth=1)
    # smap.draw_limb(axes=ax, color='white', linewidth=1)
    ax.grid(False)
    ax.coords[0].set_axislabel('')
    ax.coords[0].set_ticks_visible(False)
    ax.coords[0].set_ticklabel_visible(False)
    ax.coords[1].set_axislabel('')
    ax.coords[1].set_ticks_visible(False)
    ax.coords[1].set_ticklabel_visible(False)
    fig.tight_layout()
    return fig


def draw_hg_euv(data, cmap, norm):
    ny, nx = data.shape
    lon = np.linspace(0, 2*np.pi, nx)
    sinlat = np.linspace(-1, 1, ny)

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    ax.pcolormesh(lon, sinlat, data, cmap=cmap, norm=norm, shading='auto')
    ax.set_xticks(np.deg2rad(np.arange(0, 361, 60)))
    ax.set_xticklabels(np.arange(0, 361, 60))
    ax.set_yticks(np.sin(np.deg2rad(np.arange(-90, 91, 30))))
    ax.set_yticklabels(np.arange(-90, 91, 30))
    ax.tick_params(axis='both', length=12, which='major')
    fig.tight_layout()
    return fig


def prep_euv(file, disk_rsun_pix, disk_dim_pix, hg_dim_pix, sc=None):
    smap = Map(file)
    smap.meta['rsun_ref'] = constants.radius.to(u.m).value
    observer = smap.observer_coordinate
    if sc == 'STEREO-A' or sc == 'STEREO-B':
        smap = stereo_cor(smap)

    disk_header = get_disk_header(
        observer, 
        dim_pix=disk_dim_pix,
        rsun_pix=disk_rsun_pix
    )
    smap_disk = get_disk_map(
        smap, 
        disk_header, 
        cutoff=1,
        med_val=10,
        min_val=0,
        nan_val=0,
        remove_offlimb=False
    )

    hg_header = make_heliographic_header(
        observer.obstime,
        observer,
        hg_dim_pix,
        frame='carrington',
        projection_code='CEA',
        map_center_longitude=180*u.deg
    )
    smap_hg = get_heliographic_map(
        smap,
        hg_header,
        cutoff=1,
        med_val=10,
        min_val=0,
        fill_value=np.nan,
        remove_offlimb=True,
    )

    return smap_disk, smap_hg







def down_eui(ROOT, obstime, window=30*u.minute, wavelength=304, level='L2'):
    obstime_str = obstime.strftime("%Y%m%d_%H%M%S")

    ROOT_RAW = ROOT / f'solo_eui_fsi_{wavelength}'
    ROOT_RAW.mkdir(parents=True, exist_ok=True)

    file = ROOT_RAW / f'{obstime_str}.fits'
    if file.exists():
        return file

    try:
        time_st = obstime - window
        time_ed = obstime + window
        time_range = a.Time(time_st, time_ed)

        search = Fido.search(
            time_range,
            a.Instrument('EUI'), 
            a.Level(level),
            a.soar.Product(f'eui-fsi{wavelength}-image')
        )
        print(search)

        tobs = get_tobs(np.array(search['soar']['Start time']).astype(str), format='%Y-%m-%d %H:%M:%S.%f')
        idx = np.abs(tobs - obstime).argmin()
        f = search['soar'][idx]

        res = Path(Fido.fetch(f, path=ROOT_RAW)[0])
        res.replace(file)

        return file
    except Exception as e:
        print(f"Error downloading EUI {wavelength} data for time {obstime_str}: {e}")
        return None
    

def down_euvi(ROOT, obstime, window=30*u.minute, wavelength=304, sc='A'):
    obstime_str = obstime.strftime("%Y%m%d_%H%M%S")

    ROOT_RAW = ROOT / f'stereo_euvi_{sc}_{wavelength}'
    ROOT_RAW.mkdir(parents=True, exist_ok=True)

    file = ROOT_RAW / f'{obstime_str}.fits'
    if file.exists():
        return file

    try:
        time_st = (obstime - window).datetime
        time_ed = (obstime + window).datetime

        df = _get_summary_df(ROOT_RAW, sc, obstime)
        df = df[
            (df['wavelnth'] == wavelength) &
            (df['datetime'] >= time_st) &
            (df['datetime'] <= time_ed)
        ].copy()
        print(df)

        if df.empty:
            return None
        
        df['diff'] = (df['datetime'] - obstime.datetime).abs().dt.total_seconds()
        row = df.loc[df['diff'].idxmin()]

        url = _fits_url(sc, row['datetime'], row['filename'])
        print(url)
        _download(url, file)
        return file
    except Exception as e:
        print(f"Error downloading STEREO {sc} EUVI {wavelength} data for time {obstime_str}: {e}")
        return None
    

def _get_summary_df(ROOT_RAW, sc, obstime):
    yyyymm = obstime.strftime('%Y%m')
    if sc == 'STEREO-A':
        sc_letter = 'A'
    elif sc == 'STEREO-B':
        sc_letter = 'B'
    else:
        raise ValueError(f"Invalid spacecraft identifier: {sc}. Must be 'STEREO-A' or 'STEREO-B'.")

    cache_path = ROOT_RAW / 'summary' / f'scc{sc_letter}{yyyymm}.img.eu'
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if not cache_path.exists():
        _download(_summary_url(sc, yyyymm), cache_path)

    # Format: filename | date time | Tel | Exptime | Xsize | Ysize | Filter | wavelnth | ...
    # col indices with | as whitespace tokens:
    # 0=filename 1=| 2=date 3=time 4=| 5=Tel 6=Exptime 7=| 8=Xsize 9=| 10=Ysize 11=| 12=Filter 13=| 14=wavelnth
    df = pd.read_csv(
        cache_path, sep=r'\s+', header=None,
        usecols=[0, 2, 3, 14],
        on_bad_lines='skip',
    )
    df.columns = ['filename', 'date', 'time', 'wavelnth']
    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], format='%Y/%m/%d %H:%M:%S', errors='coerce')
    df['wavelnth'] = pd.to_numeric(df['wavelnth'], errors='coerce')
    df = df.dropna(subset=['datetime', 'wavelnth'])
    df['wavelnth'] = df['wavelnth'].astype(int)
    return df


class TqdmUpTo(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        return self.update(b * bsize - self.n)

def _download(url, path, desc=None):
    desc = desc or url.split('/')[-1]
    with TqdmUpTo(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc=desc) as t:
        request.urlretrieve(url, filename=path, reporthook=t.update_to)
        t.total = t.n

def _summary_url(sc, yyyymm):
    sc_letter = 'A' if sc == 'STEREO-A' else 'B'
    sc_path = 'a' if sc == 'STEREO-A' else 'b'
    return (
        f"https://stereo-ssc.nascom.nasa.gov/data/ins_data/secchi/L0_YMD/"
        f"{sc_path}/summary/scc{sc_letter}{yyyymm}.img.eu"
    )

def _fits_url(sc, in_time, filename):
    sc_path = 'a' if sc == 'STEREO-A' else 'b'
    return (
        f"https://stereo-ssc.nascom.nasa.gov/data/ins_data/secchi/L0_YMD/"
        f"{sc_path}/img/euvi/{in_time.strftime('%Y/%m/%d')}/{filename}"
    )

def stereo_cor(inmap):
    bias = inmap.meta['biasmean']
    ipsum = inmap.meta['ipsum']
    if ipsum > 1:
        bias = bias*float((2**(ipsum-1))**2)
    PreMap = Map(np.array(inmap.data, dtype = float)-float(bias), inmap.meta)
    return PreMap


def down_gong_adapt(ROOT, obstime):
    obstime_str = obstime.strftime("%Y%m%d_%H%M%S")

    ROOT_RAW = ROOT / 'gong_adapt'
    ROOT_RAW.mkdir(parents=True, exist_ok=True)

    file = ROOT_RAW / f'{obstime_str}.fits'
    if file.exists():
        return file
    
    try:
        URL = 'https://gong.nso.edu/adapt/maps/gong/'
        year = obstime.strftime('%Y')
        date_key = obstime.strftime('%Y%m%d%H%M')  # e.g. 202606251600

        dir_url = URL + year + '/'
        with urllib.request.urlopen(dir_url) as resp:
            html = resp.read().decode()

        pattern = rf'(adapt403\d+_\d+_{re.escape(date_key)}_\S+?\.fts\.gz)'
        matches = list(dict.fromkeys(re.findall(pattern, html)))
        if not matches:
            raise FileNotFoundError(f"No ADAPT file found for {date_key} in {dir_url}")
        
        print(matches)

        matched = dir_url + matches[0]

        dl = Downloader(progress=True)
        dl.enqueue_file(matched, path=ROOT_RAW)
        files = dl.download()
        assert len(files) == 1, f"Expected 1 file to be downloaded, but got {len(files)}"
        downloaded_file = files[0]
        Path(downloaded_file).rename(file)
        return file
    except Exception as e:
        print(f"Error downloading GONG ADAPT data for time {obstime_str}: {e}")
        return None
    

def get_mu(smap):
    coords = all_coordinates_from_map(smap)
    observer = smap.observer_coordinate
    coords = coords.transform_to(frames.Helioprojective(observer=observer, obstime=observer.obstime))
    mu = np.cos(get_heliocentric_angle(coords).to(u.rad).value)

    # mu[mu < 0] = np.nan
    # mu[mu > 1] = 1

    return mu
