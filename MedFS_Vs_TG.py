import pandas as pd
import sys
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib as mpl
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy.fft as F
import scipy as sp
from scipy import signal
import matplotlib.colors as mcolors
mpl.use('Agg')

# ------------------------------
# VARIABILI GLOBALI
# ------------------------------
# Event name
event_name=''

# File of tide-gauges location
coo_file = 'TGs_'+event_name+'.coo'

# MedFS time-series path:
medfs_dir = '/work/cmcc/ag15419/surge/'+event_name+'/mod_extr/'

# Output directory:
output_dir = '/work/cmcc/ag15419/surge/'+event_name+'/plots/'

# ---------------------------
# Dates based on the event name
# ---------------------------
# Importo il database degli eventi 
from events_db import events

if event_name not in events:
    sys.exit(f"UNKNOWN Event '{event_name}'! Define it in the event dictionary.")

event_start_str, event_end_str, box = events[event_name]

event_start = pd.Timestamp(event_start_str + " 00:00:00", tz="UTC")
event_end   = pd.Timestamp(event_end_str   + " 23:59:59", tz="UTC")

print(f"\nEvent: {event_name}")
print(f"Start: {event_start}")
print(f"End  : {event_end}")

# -------------------------------   
# PERIOD of the analysis defined as -30 / +10 wrt the event
# SURGE reference period (mean on all avalaible days before the event )
# BULLTETIN period (forecast to be included in the analysis)
# FORECAST plot limits (-3 to +3 days wrt the event range)
# TIDAL diurnal and semi-diurnal bands (for detiding)
# -------------------------------

# Period of the analysis (defined as -30 / +10 wrt the event)
xlim_start = event_start - pd.Timedelta(days=30)
xlim_end   = event_end + pd.Timedelta(days=10)

# Surge reference
base_start  = xlim_start
base_end    = event_start - pd.Timedelta(days=1)

# Bulletins covering the event period
b_start_date = event_start - pd.Timedelta(days=10)  # 10 days before the event
b_end_date   = event_end - pd.Timedelta(days=1)     # 1 day before the event
# bulletin range
bdates = pd.date_range(start=b_start_date, end=b_end_date, freq='D')

# Forecast plots range
xlim_fc_start = event_start - pd.Timedelta(days=3)
xlim_fc_end   = event_end + pd.Timedelta(days=3) - pd.Timedelta(seconds=1) 

# Tidal diurnal and semi-diurnal bands
semid_tides_band = [11.2, 17.0]
diurnal_tides_band = [21.0, 32.0]

# ------------------------------
# FUNZIONI GENERICHE
# ------------------------------

def read_coo_file(coo_file):
    df = pd.read_csv(
        coo_file,
        sep=';',
        comment='#',
        header=None,
        usecols=[0,1,2,3],
        names=['lat', 'lon', 'name', 'obs_path']
    )
    return df


def read_obs_csv(obs_folder, hourly_mean=True, interpolate_gaps=True, verbose=True):

    obs_folder = Path(obs_folder)
    obs_files = [obs_folder] if obs_folder.is_file() else list(obs_folder.glob('*.csv'))
    obs_dict = {}

    for file_path in obs_files:
        try:
            # -----------------------------
            # Lettura dati
            # -----------------------------
            times, values = [], []
            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(' ', 1)
                    if len(parts) < 2:
                        continue
                    times.append(parts[0])
                    values.append(parts[1])

            df = pd.DataFrame({'time': times, 'ssh_obs': values})
            df['time'] = pd.to_datetime(
                df['time'],
                format='%Y-%m-%dT%H:%M:%SZ',
                utc=True,
                errors='coerce'
            )
            df['ssh_obs'] = pd.to_numeric(df['ssh_obs'], errors='coerce')
            df = df.dropna(subset=['time', 'ssh_obs'])
            df = df.set_index('time')

            # -----------------------------
            # Diagnostica duplicati
            # -----------------------------
            n_dup = df.index.duplicated().sum()
            if verbose and n_dup > 0:
                print(f"[OBS WARNING] {file_path.name}: {n_dup} duplicated timestamps found")

            df = df[~df.index.duplicated(keep='last')]

            # -----------------------------
            # Ordina tempo
            # -----------------------------
            df = df.sort_index()

            # -----------------------------
            # Info step temporale
            # -----------------------------
            if verbose:
                dt_counts = df.index.to_series().diff().value_counts().head(3)
                #print(f"[OBS INFO] {file_path.name} time-step (top):")
                #print(dt_counts)

            # -----------------------------
            # Resample orario
            # -----------------------------
            if hourly_mean:
                df = df.resample('1H', label='right', closed='right').mean()
                df.index = df.index - pd.Timedelta(minutes=30)

                # -----------------------------
                # Interpolazione dei gap
                # -----------------------------
                if interpolate_gaps:
                    df['ssh_obs'] = df['ssh_obs'].interpolate(method='time', limit_direction='both')

            obs_dict[file_path.stem] = df

        except Exception as e:
            print(f"[OBS ERROR] {file_path.name}: {e}")

    if len(obs_dict) == 1:
        return list(obs_dict.values())[0]

    return obs_dict

def read_obs_netcdf(obs_folder,hourly_mean=True,interpolate_gaps=True,verbose=True):

    obs_folder = Path(obs_folder)
    obs_files = [obs_folder] if obs_folder.is_file() else sorted(obs_folder.glob("*.nc"))
    obs_dict = {}

    for file_path in obs_files:
        try:
            # -----------------------------
            # Apertura NetCDF
            # -----------------------------
            ds = xr.open_dataset(file_path, decode_times=True)

            # -----------------------------
            # Controlli minimi
            # -----------------------------
            if "TIME" not in ds or "SLEV" not in ds:
                raise ValueError("Required variables TIME and/or SLEV not found")

            # -----------------------------
            # Estrazione tempo e valori
            # -----------------------------
            time = pd.to_datetime(ds["TIME"].values)
            
            ssh_data = ds["SLEV"].values
            if ssh_data.ndim == 2:
                ssh = ssh_data[:, 0]  # prendi solo il primo livello di DEPTH
            elif ssh_data.ndim == 1:
                ssh = ssh_data
            else:
                raise ValueError(f"SLEV has unexpected shape {ssh_data.shape}")
            
            df = pd.DataFrame({"ssh_obs": ssh}, index=time)
            
            # rende indice timezone-aware UTC
            df.index = df.index.tz_localize("UTC") if df.index.tz is None else df.index
            
            # rimuove NaN solo sui dati
            df = df.dropna(subset=["ssh_obs"])

            # -----------------------------
            # FILTRO INTERVALLO TEMPO
            # -----------------------------
            df = df[(df.index >= xlim_start) & (df.index <= xlim_end)]

            # -----------------------------
            # Diagnostica duplicati
            # -----------------------------
            n_dup = df.index.duplicated().sum()
            if verbose and n_dup > 0:
                print(f"[OBS WARNING] {file_path.name}: {n_dup} duplicated timestamps found")

            df = df[~df.index.duplicated(keep="last")]

            # -----------------------------
            # Ordina tempo
            # -----------------------------
            df = df.sort_index()

            # -----------------------------
            # Info step temporale
            # -----------------------------
            if verbose and len(df) > 1:
                dt_counts = df.index.to_series().diff().value_counts().head(3)
                print(f"OBS FILE: {file_path.name}")
                #print(dt_counts)

            # -----------------------------
            # Resample orario
            # -----------------------------
            if hourly_mean:
                df = df.resample("1H", label="right", closed="right").mean()
                df.index = df.index - pd.Timedelta(minutes=30)

                # -----------------------------
                # Interpolazione gap
                # -----------------------------
                if interpolate_gaps:
                    df["ssh_obs"] = df["ssh_obs"].interpolate(
                        method="time",
                        limit_direction="both"
                    )

            obs_dict[file_path.stem] = df
            ds.close()

            print(df.index)
            print ("#####")

        except Exception as e:
            print(f"[OBS ERROR] {file_path.name}: {e}")

    if len(obs_dict) == 1:
        return list(obs_dict.values())[0]

    return obs_dict

def read_model_nc(nc_file, obs_index=None):
    print(f"MOD FILE: {nc_file}")
    ds = xr.open_dataset(nc_file)
    ssh = ds['sossheig']  # (time, lat, lon)
    time = pd.to_datetime(ds['time_counter'].values).tz_localize('UTC')
    ssh_vals = ssh[:,0,0].values
    df_mod = pd.DataFrame({'ssh_mod': ssh_vals}, index=time)
    if obs_index is not None:
        start, end = obs_index.min(), obs_index.max()
        df_mod = df_mod.loc[start:end]
    print(df_mod.index)
    print ("#####")
    return df_mod

# ------------------------------
# FUNZIONI PLOT
# ------------------------------

def plot_tg(name, obs, mod, outdir):
    # Usa solo dati resample orario
    df = obs.join(mod, how='inner')
    if df.empty:
        print(f'Nessuna sovrapposizione temporale per {name}')
        return None

    # Allinea media MODEL a OBS
    offset = df['ssh_obs'].mean() - df['ssh_mod'].mean()
    df['ssh_mod_offset'] = df['ssh_mod'] + offset

    # Seleziona periodo evento per annotazioni
    df_event = df.loc[event_start:event_end]

    # Check valori validi nell'evento
    if df_event['ssh_obs'].notna().any():
        idx_max_obs = df_event['ssh_obs'].idxmax()
        max_obs = df_event['ssh_obs'].max()
    else:
        idx_max_obs = None
        max_obs = None

    if df_event['ssh_mod_offset'].notna().any():
        idx_max_mod = df_event['ssh_mod_offset'].idxmax()
        max_mod = df_event['ssh_mod_offset'].max()
    else:
        idx_max_mod = None
        max_mod = None

    # --- PLOT ---
    fontsize, linewidth, legend_fontsize = 20, 3, 18
    plt.figure(figsize=(12,6))
    plt.axvspan(event_start, event_end, color='lightgrey', alpha=0.3)
    
    # Plotta solo dati resample orario
    plt.plot(df.index, df['ssh_obs'], color='tab:orange',
             lw=linewidth,
             label=f'OBS (max {max_obs:.3f} m @ {idx_max_obs:%d/%m %H:%M})' if max_obs is not None else 'OBS (no data)')
    plt.plot(df.index, df['ssh_mod_offset'], color='blue',
             lw=linewidth,
             label=f'ANALYSIS (max {max_mod:.3f} m @ {idx_max_mod:%d/%m %H:%M})' if max_mod is not None else 'ANALYSIS (no data)')
    
    plt.title(name, fontsize=fontsize)
    plt.ylabel('Sea level [m]', fontsize=fontsize)
    plt.xlabel('Time', fontsize=fontsize)
    plt.legend(fontsize=legend_fontsize, loc='upper left', framealpha=0.7)
    plt.xticks(fontsize=fontsize-2, rotation=30)
    plt.yticks(fontsize=fontsize-2)
    #plt.xlim(xlim_start, xlim_end)
    plt.grid()
    plt.tight_layout()
    outdir.mkdir(exist_ok=True, parents=True)
    plt.savefig(outdir / f'{name}_obs_vs_mod.png', dpi=150)
    plt.close()
    return df


def plot_tg_map(df_coo, outdir, figsize=(20,7)):
    fig = plt.figure(figsize=figsize)
    plt.rcParams['font.size'] = 20
    ax = plt.axes(projection=ccrs.PlateCarree())

    # Estensione per tutto il Mediterraneo
    min_lon, max_lon = -6, 37   # Ovest: Marocco/Spagna, Est: Levante/Cipro
    min_lat, max_lat = 30, 46   # Sud: Nord Africa, Nord: coste francesi e Balcani
    ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.COASTLINE, linewidth=1)
    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)

    for _, row in df_coo.iterrows():
        lat, lon, name = row['lat'], row['lon'], row['name']
        ax.plot(lon, lat, 'ro', markersize=8, transform=ccrs.PlateCarree())
        ax.text(lon+0.2, lat, name, fontsize=16, transform=ccrs.PlateCarree())
    
    plt.title('Tide-Gauges – Mediterranean', fontsize=20)
    outdir.mkdir(exist_ok=True, parents=True)
    plt.tight_layout()
    plt.savefig(outdir / 'tg_map_mediterranean.png', dpi=150)
    plt.close()
    
def plot_tg_map_single(df_coo, tg_name, outdir, figsize=(10,3)):
        row = df_coo[df_coo['name'] == tg_name]
        if row.empty:
            print(f"[MAP WARNING] TG {tg_name} non trovata")
            return
    
        lat, lon = row.iloc[0]['lat'], row.iloc[0]['lon']
    
        fig = plt.figure(figsize=figsize)
        plt.rcParams['font.size'] = 14
        ax = plt.axes(projection=ccrs.PlateCarree())
    
        # Stesso dominio del Mediterraneo
        min_lon, max_lon = -6, 37
        min_lat, max_lat = 30, 46
        ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())
    
        ax.add_feature(cfeature.LAND, facecolor='lightgray')
        ax.add_feature(cfeature.COASTLINE, linewidth=1)
        ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    
        # Evidenzia solo la TG
        ax.plot(lon, lat, 'ro', markersize=10, transform=ccrs.PlateCarree())
        ax.text(lon+0.2, lat, tg_name, fontsize=16, transform=ccrs.PlateCarree())
    
        plt.title(f'Tide-Gauge: {tg_name}', fontsize=20)
        outdir.mkdir(exist_ok=True, parents=True)
        plt.tight_layout()
        plt.savefig(outdir / f'{tg_name}_map.png', dpi=150)
        plt.close()
    
def compute_stats(df, name):
        from sklearn.metrics import mean_squared_error
        obs = df['ssh_obs']
        mod = df['ssh_mod_offset']
        rmse = np.sqrt(mean_squared_error(obs, mod))
        corr = obs.corr(mod)
        offset = mod.mean() - df['ssh_mod'].mean()
        n_points = len(df)
        return {
            'name': name,
            'n_points': n_points,
            'obs_mean': obs.mean(),
            'mod_mean': df['ssh_mod'].mean(),
            'offset_applied': offset,
            'rmse': rmse,
            'corr': corr
        }
    
    # ------------------------------
    # DETIDING / SPETTRI
    # ------------------------------
    
def fft2bands(nelevation, low_bound=1./20.0, high_bound=1./30.0,
                  low_bound_1=1/11.2, high_bound_1=1/13.5, alpha=0.4, invert='False'):
        if len(nelevation) % 2:
            result = F.rfft(nelevation, len(nelevation))
        else:
            result = F.rfft(nelevation)
        freq = F.fftfreq(len(nelevation))[:result.shape[0]]
        factor = np.ones_like(result)
        sl = np.logical_and(high_bound < freq, freq < low_bound)
        sl_2 = np.logical_and(high_bound_1 < freq, freq < low_bound_1)
        a = factor[sl]
        b = factor[sl_2]
        lena = a.shape[0]
        lenb = b.shape[0]
        a = 1 - sp.signal.tukey(lena, alpha)
        b = 1 - sp.signal.tukey(lenb, alpha)
        factor[sl] = a[:lena]
        factor[sl_2] = b[:lenb]
        if invert=='False':
            result = result * factor
        else:
            result = result * (-(factor-1))
        relevation = F.irfft(result, len(nelevation))
        return relevation, np.abs(factor)
    
def plot_detided_ts(name, df, outdir):
        fontsize, linewidth, legend_fontsize = 20, 3, 18
    
        # Allinea media MODEL detided a OBS detided
        offset_detided = df['ssh_obs_detided'].mean() - df['ssh_mod_offset_detided'].mean()
        df['ssh_mod_offset_detided_aligned'] = df['ssh_mod_offset_detided'] + offset_detided
    
        # Seleziona periodo evento
        df_event = df.loc[event_start:event_end]
    
        # OBS detided
        if df_event['ssh_obs_detided'].notna().any():
            idx_max_obs = df_event['ssh_obs_detided'].idxmax()
            max_obs = df_event['ssh_obs_detided'].max()
            obs_label = f'OBS detided (max {max_obs:.3f} m @ {idx_max_obs:%d/%m %H:%M})'
        else:
            idx_max_obs = None
            max_obs = None
            obs_label = 'OBS detided (no valid data)'
    
        # MODEL detided
        if df_event['ssh_mod_offset_detided_aligned'].notna().any():
            idx_max_mod = df_event['ssh_mod_offset_detided_aligned'].idxmax()
            max_mod = df_event['ssh_mod_offset_detided_aligned'].max()
            mod_label = f'ANALYSIS detided (max {max_mod:.3f} m @ {idx_max_mod:%d/%m %H:%M})'
        else:
            idx_max_mod = None
            max_mod = None
            mod_label = 'ANALYSIS detided (no valid data)'
    
        # --- PLOT ---
        plt.figure(figsize=(12,6))
        plt.axvspan(event_start, event_end, color='lightgrey', alpha=0.3)
    
        # Plotta solo dati resample orario
        plt.plot(df.index, df['ssh_obs_detided'], color='tab:orange', lw=linewidth, label=obs_label)
        plt.plot(df.index, df['ssh_mod_offset_detided_aligned'], color='blue', lw=linewidth, label=mod_label)
    
        plt.title(f'{name} – Detided sea level', fontsize=fontsize)
        plt.ylabel('Sea level [m]', fontsize=fontsize)
        plt.xlabel('Time', fontsize=fontsize)
        plt.legend(loc='upper left', fontsize=legend_fontsize, framealpha=0.7)
        plt.xticks(fontsize=fontsize-2, rotation=30)
        plt.yticks(fontsize=fontsize-2)
        plt.grid()
        plt.tight_layout()
        outdir.mkdir(exist_ok=True, parents=True)
        plt.savefig(outdir / f'{name}_detided_ts.png', dpi=150)
        plt.close()
    
def compute_spectrum_amp(ts, dt_hours):
        ts = ts - np.nanmean(ts)
        n = len(ts)
        fft = np.fft.rfft(ts)
        amp = np.abs(fft)/n*2
        freq = np.fft.rfftfreq(n, d=dt_hours*3600)
        period = 1/freq/3600
        return period[1:], amp[1:]
    
def plot_spectra(name, df, outdir, dt_hours=1.0):
        fontsize = 20
        per_obs, amp_obs = compute_spectrum_amp(df['ssh_obs'].values, dt_hours)
        per_mod, amp_mod = compute_spectrum_amp(df['ssh_mod_offset'].values, dt_hours)
        per_obs_d, amp_obs_d = compute_spectrum_amp(df['ssh_obs_detided'].values, dt_hours)
        per_mod_d, amp_mod_d = compute_spectrum_amp(df['ssh_mod_offset_detided'].values, dt_hours)
        plt.figure(figsize=(12,6))
        plt.semilogx(per_obs, amp_obs, label='OBS')
        plt.semilogx(per_mod, amp_mod, label='MODEL')
        plt.semilogx(per_obs_d, amp_obs_d, '--', label='OBS detided')
        plt.semilogx(per_mod_d, amp_mod_d, '--', label='MODEL detided')
        plt.xlabel('Period [h]', fontsize=fontsize)
        plt.ylabel('Amplitude [m]', fontsize=fontsize)
        plt.title(f'{name} – Amplitude spectra', fontsize=fontsize)
        plt.legend(fontsize=fontsize-2, framealpha=0.7)
        plt.grid(which='both')
        plt.xlim(2, 100)
        plt.axvline(12, color='gray', linestyle='--', lw=2)
        plt.axvline(24, color='gray', linestyle='--', lw=2)
        plt.tight_layout()
        outdir.mkdir(exist_ok=True, parents=True)
        plt.savefig(outdir / f'{name}_spectra_amp.png', dpi=150)
        plt.close()
    
    
def combine_all_plots(df_coo, outdir):
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        
        for _, row in df_coo.iterrows():
            name = row['name']
            
            # Percorsi delle figure già prodotte
            map_file = outdir / f"{name}_map.png"
            obs_vs_mod_file = outdir / f"{name}_obs_vs_mod.png"
            detided_file = outdir / f"{name}_detided_ts.png"
            forecast_file = outdir / f"{name}_forecast_all_original.png"
            forecast_detided_file = outdir / f"{name}_forecast_all_detided.png"
            
            # Nuova figura
            fig = plt.figure(figsize=(12,10))
            gs = GridSpec(3, 2, figure=fig, height_ratios=[1,1,1], width_ratios=[1,1])
            
            # --- Mappa in alto (centro) ---
            ax_map = fig.add_subplot(gs[0, :])
            map_img = plt.imread(map_file)
            ax_map.imshow(map_img)
            ax_map.axis('off')  # togli assi
            
            # --- 2x2 plot sotto ---
            ax1 = fig.add_subplot(gs[1,0])
            img1 = plt.imread(obs_vs_mod_file)
            ax1.imshow(img1)
            ax1.axis('off')
            
            ax2 = fig.add_subplot(gs[1,1])
            img2 = plt.imread(detided_file)
            ax2.imshow(img2)
            ax2.axis('off')
            
            ax3 = fig.add_subplot(gs[2,0])
            img3 = plt.imread(forecast_file)
            ax3.imshow(img3)
            ax3.axis('off')
            
            ax4 = fig.add_subplot(gs[2,1])
            img4 = plt.imread(forecast_detided_file)
            ax4.imshow(img4)
            ax4.axis('off')
            
            # Ridurre gli spazi bianchi tra i subplot
            plt.subplots_adjust(
                left=0.03,
                right=0.97,
                top=0.97,
                bottom=0.03,
                hspace=0.05,
                wspace=0.05
            )
            
            # Salva figura combinata
            fig_file = outdir / f"{name}_combined.png"
            plt.savefig(fig_file, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Combined plot saved: {fig_file}")
    
    
def compute_detided_event_anomaly(df,event_start, event_end,base_start, base_end,var='ssh_obs_detided'):

        df_event = df.loc[event_start:event_end]
        df_base  = df.loc[base_start:base_end]
    
        if not df_event[var].notna().any():
            return np.nan
        if not df_base[var].notna().any():
            return np.nan
    
        max_event = df_event[var].max()
        mean_base = df_base[var].mean()
    
        return max_event - mean_base

def plot_detided_event_anomaly_map(df_tg_summary,outdir,value_col,title,outfile,bounds=None,cmap_name='plasma_r'):

    if bounds is None:
        bounds = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7] #, 0.85, 0.9, 0.95, 1.0]

    plt.figure(figsize=(20, 7))
    plt.rcParams['font.size'] = 20
    ax = plt.axes(projection=ccrs.PlateCarree())

    # Estensione Mediterraneo
    ax.set_extent([-6, 37, 30, 46], crs=ccrs.PlateCarree())

    # Background
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.COASTLINE, linewidth=1)

    # Gridlines
    gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    gl.top_labels = False
    gl.right_labels = False

    # Colorbar discreta
    cmap = plt.cm.get_cmap(cmap_name)
    norm = mcolors.BoundaryNorm(bounds, ncolors=cmap.N, clip=False)

    # Scatter TG
    sc = ax.scatter(
        df_tg_summary['lon'],
        df_tg_summary['lat'],
        c=df_tg_summary[value_col],
        cmap=cmap,
        norm=norm,
        s=80,
        edgecolor='k',
        transform=ccrs.PlateCarree(),
        zorder=10
    )

    # Colorbar
    cbar = plt.colorbar(
        sc,
        ax=ax,
        orientation='horizontal',
        pad=0.08,
        ticks=bounds,
        extend='max'
    )
    # surge reference label
    if base_start.month == base_end.month:
       surge_ref_label = f"{base_start.day}–{base_end.day} {base_start.strftime('%b %Y')}"
    else:
       surge_ref_label = (f"{base_start.day} {base_start.strftime('%b')} – "
               f"{base_end.day} {base_end.strftime('%b %Y')}")

    cbar.set_label(rf'$\max(\eta^{{det}}_{{event}}) - \langle \eta^{{det}} \rangle_{{{surge_ref_label}}}$ [m]',fontsize=20)

    plt.title(title, fontsize=20)
    plt.tight_layout()

    outdir.mkdir(exist_ok=True, parents=True)
    plt.savefig(outdir / outfile, dpi=150)
    plt.close()

    
    # ------------------------------
    # MAIN
    # ------------------------------
    
def main():
        mod_dir = Path(medfs_dir)
        outdir = Path(output_dir)

        stats_list = []
        tg_summary = []

        df_coo = read_coo_file(coo_file)
    
        for _, row in df_coo.iterrows():
            name, obs_path = row['name'], row['obs_path']
            print(f'Processing {name}')
    
            try:
                #obs = read_obs_csv(obs_path)
                obs = read_obs_netcdf(obs_path)
    
            except Exception as e:
                print(f"Obs error: {e}")
                continue
    
            # MODEL
            nc_file = mod_dir / f'{name}_mod_MedFS_analysis.nc'
            if not nc_file.exists(): 
                print(f"Missing model file for {name}")
                continue
            mod = read_model_nc(nc_file, obs.index)
    
            df_plot = plot_tg(name, obs, mod, outdir)
    
            # Mappa generale
            plot_tg_map(df_coo, outdir)
    
            # Mappe singole per slide
            for _, row in df_coo.iterrows():
              plot_tg_map_single(df_coo, row['name'], outdir)
    
            if df_plot is None: continue
    
            # -------------------------
            # DETIDING OBS/MODEL
            # -------------------------
            low_bound_d = 1.0 / diurnal_tides_band[1]
            high_bound_d = 1.0 / diurnal_tides_band[0]
            low_bound_sd = 1.0 / semid_tides_band[1]
            high_bound_sd = 1.0 / semid_tides_band[0]
    
            obs_detided, _ = fft2bands(
                df_plot['ssh_obs'].values,
                low_bound=high_bound_d, high_bound=low_bound_d,
                low_bound_1=high_bound_sd, high_bound_1=low_bound_sd,
                alpha=0.4, invert='False'
            )
            df_plot['ssh_obs_detided'] = obs_detided
    
            mod_detided, _ = fft2bands(
                df_plot['ssh_mod_offset'].values,
                low_bound=high_bound_d, high_bound=low_bound_d,
                low_bound_1=high_bound_sd, high_bound_1=low_bound_sd,
                alpha=0.4, invert='False'
            )
            df_plot['ssh_mod_offset_detided'] = mod_detided
    
            plot_detided_ts(name, df_plot, outdir)
            plot_spectra(name, df_plot, outdir, dt_hours=1.0)
    
            # -------------------------
            # FORECASTS
            # -------------------------
            
            # --- PLOT 1: forecast ORIGINAL ---
            plt.figure(figsize=(12,6))
            plt.axvspan(event_start, event_end, color='lightgrey', alpha=0.3)
            
            # calcolo max su ANALYSIS e OBS
            df_event = df_plot.loc[event_start:event_end]
            try:
               idx_max_obs = df_event['ssh_obs'].idxmax()
            except:
               idx_max_obs = None
            try:
               max_obs = df_event['ssh_obs'].max()
            except:
               max_obs = None
            try:
               idx_max_mod = df_event['ssh_mod_offset'].idxmax()
            except:
               idx_max_mod = None
            try:
               max_mod = df_event['ssh_mod_offset'].max()
            except:
               max_mod = None
            
            plt.plot(
                df_plot.index,
                df_plot['ssh_obs'],
                color='tab:orange',
                lw=3,
                label=f'OBS (max {max_obs:.3f} m @ {idx_max_obs:%d/%m %H:%M})'
            )
            plt.plot(
                df_plot.index,
                df_plot['ssh_mod_offset'],
                color='blue',
                lw=3,
                label=f'ANALYSIS (max {max_mod:.3f} m @ {idx_max_mod:%d/%m %H:%M})'
            )
            
            # forecast
            cmap = plt.cm.viridis
            norm = plt.Normalize(vmin=0, vmax=len(bdates)-1)
            
            for i, bdate in enumerate(bdates):
                fc_file = mod_dir / f'{name}_mod_MedFS_forecast_b{bdate:%Y%m%d}.nc'
                if not fc_file.exists():
                    print(f"Missing forecast file: {fc_file}")
                    continue
                df_fc = read_model_nc(fc_file)
                # offset rispetto alle OBS presenti nel periodo forecast
                mask_obs = (df_plot.index >= df_fc.index.min()) & (df_plot.index <= df_fc.index.max())
                offset_fc = df_plot['ssh_obs'].loc[mask_obs].mean() - df_fc['ssh_mod'].mean() if mask_obs.any() else 0.0
                df_fc['ssh_mod_offset'] = df_fc['ssh_mod'] + offset_fc
            
                # calcolo massimo per questa forecast
                df_event_fc = df_fc.loc[event_start:event_end]
                if not df_event_fc.empty:
                    idx_max_fc = df_event_fc['ssh_mod_offset'].idxmax()
                    max_fc = df_event_fc['ssh_mod_offset'].max()
                else:
                    idx_max_fc = df_fc.index[0]
                    max_fc = df_fc['ssh_mod_offset'].max()
            
                color = cmap(norm(i))
                plt.plot(
                    df_fc.index,
                    df_fc['ssh_mod_offset'],
                    lw=2,
                    color=color,
                    label=f'FC {bdate:%d %b} (max {max_fc:.3f} m @ {idx_max_fc:%d/%m %H:%M})'
                )
            
            plt.title(f'{name}', fontsize=20)
            plt.ylabel('Sea level [m]', fontsize=20)
            plt.xlabel('Time', fontsize=20)
            plt.legend(loc='upper left', fontsize=12, framealpha=0.7)
            plt.xticks(fontsize=16, rotation=30)
            plt.yticks(fontsize=16)
            plt.xlim(xlim_start, xlim_end)
            plt.grid()
            plt.tight_layout()
            outdir.mkdir(exist_ok=True, parents=True)
            plt.savefig(outdir / f'{name}_forecast_all_original.png', dpi=150)
            plt.close()
            
            # --- PLOT 2: forecast DETIDED ---
            plt.figure(figsize=(12,6))
            plt.axvspan(event_start, event_end, color='lightgrey', alpha=0.3)
            
            # calcolo max su ANALYSIS detided e OBS detided
            offset_detided = df_plot['ssh_obs_detided'].mean() - df_plot['ssh_mod_offset_detided'].mean()
            df_plot['ssh_mod_offset_detided_aligned'] = df_plot['ssh_mod_offset_detided'] + offset_detided
            
            df_event = df_plot.loc[event_start:event_end]
            idx_max_obs = df_event['ssh_obs_detided'].idxmax()
            max_obs = df_event['ssh_obs_detided'].max()
            idx_max_mod = df_event['ssh_mod_offset_detided_aligned'].idxmax()
            max_mod = df_event['ssh_mod_offset_detided_aligned'].max()
            
            plt.plot(
                df_plot.index,
                df_plot['ssh_obs_detided'],
                color='tab:orange',
                lw=3,
                label=f'OBS detided (max {max_obs:.3f} m @ {idx_max_obs:%d/%m %H:%M})'
            )
            plt.plot(
                df_plot.index,
                df_plot['ssh_mod_offset_detided_aligned'],
                color='blue',
                lw=3,
                label=f'ANALYSIS detided (max {max_mod:.3f} m @ {idx_max_mod:%d/%m %H:%M})'
            )
            
            # forecast detided
            for i, bdate in enumerate(bdates):
                fc_file = mod_dir / f'{name}_mod_MedFS_forecast_b{bdate:%Y%m%d}.nc'
                if not fc_file.exists():
                    continue
                df_fc = read_model_nc(fc_file)
                mask_obs = (df_plot.index >= df_fc.index.min()) & (df_plot.index <= df_fc.index.max())
                offset_fc = df_plot['ssh_obs'].loc[mask_obs].mean() - df_fc['ssh_mod'].mean() if mask_obs.any() else 0.0
                df_fc['ssh_mod_offset'] = df_fc['ssh_mod'] + offset_fc
            
                df_fc_detided, _ = fft2bands(
                    df_fc['ssh_mod_offset'].values,
                    low_bound=high_bound_d, high_bound=low_bound_d,
                    low_bound_1=high_bound_sd, high_bound_1=low_bound_sd,
                    alpha=0.4, invert='False'
                )
                df_fc_detided_series = pd.Series(df_fc_detided, index=df_fc.index)
            
                df_event_fc = df_fc_detided_series.loc[event_start:event_end]
                if not df_event_fc.empty:
                    idx_max_fc = df_event_fc.idxmax()
                    max_fc = df_event_fc.max()
                else:
                    idx_max_fc = df_fc.index[0]
                    max_fc = df_fc_detided_series.max()
            
                color = cmap(norm(i))
                plt.plot(
                    df_fc.index,
                    df_fc_detided_series,
                    lw=2,
                    color=color,
                    label=f'FC {bdate:%d %b} detided (max {max_fc:.3f} m @ {idx_max_fc:%d/%m %H:%M})'
                )
            
            plt.title(f'{name} – Detided sea level', fontsize=20)
            plt.ylabel('Sea level [m]', fontsize=20)
            plt.xlabel('Time', fontsize=20)
            plt.legend(loc='upper left', fontsize=12, framealpha=0.7)
            plt.xticks(fontsize=16, rotation=30)
            plt.yticks(fontsize=16)
            plt.xlim(xlim_start, xlim_end)
            plt.grid()
            plt.tight_layout()
            outdir.mkdir(exist_ok=True, parents=True)
            plt.savefig(outdir / f'{name}_forecast_all_detided.png', dpi=150)
            plt.close()
    
            # -------------------------
            # STATISTICHE
            # -------------------------
            stats = compute_stats(df_plot, name)
            stats_list.append(stats)
    
            # -------------------------
            # CALCOLO SURGE
            # -------------------------
    
            delta_detided = compute_detided_event_anomaly(df_plot,event_start, event_end,base_start, base_end,var='ssh_obs_detided')
            delta_detided_mod = compute_detided_event_anomaly(df_plot,event_start,event_end,base_start,base_end,var='ssh_mod_offset_detided')
    
            row = df_coo[df_coo['name'] == name]
            lat, lon = row.iloc[0]['lat'], row.iloc[0]['lon']
    
            tg_summary.append({
               'name': name,
               'lon': lon,
               'lat': lat,
               'delta_detided': delta_detided,
               'delta_detided_mod': delta_detided_mod})
    
        # -------------------------
        # PLOT COMBINATI
        # -------------------------
        combine_all_plots(df_coo, outdir)
    
        # -------------------------
        # PLOT SURGE
        # -------------------------
        df_tg_summary = pd.DataFrame(tg_summary)

        plot_detided_event_anomaly_map(
           df_tg_summary=df_tg_summary,
           outdir=outdir,
           value_col='delta_detided',
           title='SURGE – Detided sea-level anomaly during event (OBS)',
           outfile='TG_detided_event_anomaly_map_OBS.png')    

        plot_detided_event_anomaly_map(
           df_tg_summary=df_tg_summary,
           outdir=outdir,
           value_col='delta_detided_mod',
           title='SURGE – Detided sea-level anomaly during event (ANALYSIS)',
           outfile='TG_detided_event_anomaly_map_ANALYSIS.png')

    
        if stats_list:
            df_stats = pd.DataFrame(stats_list)
            df_stats.to_csv(outdir / 'tg_obs_vs_mod_stats.csv', index=False)
            print("Statistiche salvate in tg_obs_vs_mod_stats.csv")
        else:
            print("Nessuna statistica calcolata")

if __name__ == '__main__':
    main()
