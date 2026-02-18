import xarray as xr
import numpy as np
from pathlib import Path
import pandas as pd

# ==========================
# INPUT UTENTE
# ==========================
# Old (all dates in a single file) or new (1 file per day) archive
archive = "old"         
# Name of the event 
event_name = "vaia"    

# --- OLD archive ---
data_dir = Path("/data/inputs/METOCEAN/historical/obs/ocean/in_situ/CMS/MedSea/history_EIS_202511/tidegauge/")
# --- NEW archive (giornaliero) ---
new_archive_dir = Path("/data/inputs/METOCEAN/historical/obs/ocean/in_situ/CMS/MedSea/latest_evalid_EIS_202311/")
new_output_dir  = Path(f"/work/cmcc/ag15419/surge/{event_name}/obs/")

output_csv = Path(f"TGs_{event_name}.coo")

# ---------------------------
# Dates based on the event name
# ---------------------------

# Importo il database degli eventi
from events_db import events

# Funzione poer la lettura delle coordinate
if event_name not in events:
    sys.exit(f"UNKNOWN Event '{event_name}'! Define it in the event dictionary.")

event_start_str, event_end_str, box = events[event_name]

event_start = pd.Timestamp(event_start_str + " 00:00:00", tz="UTC")
event_end   = pd.Timestamp(event_end_str   + " 23:59:59", tz="UTC")

# Period of the analysis (defined as -30 / +10 wrt the event)
interval_start = event_start - pd.Timedelta(days=30)
interval_end   = event_end + pd.Timedelta(days=10)

t_start_input = pd.to_datetime(interval_start)
t_end_input   = pd.to_datetime(interval_end)

lat_min, lat_max, lon_min, lon_max = box

print(f"\nEvento: {event_name}")
print(f"Intervallo analisi: {t_start_input} → {t_end_input}")
print(f"Box: lat[{lat_min},{lat_max}] lon[{lon_min},{lon_max}]")

def get_lat_lon(ds):
    # LAT
    if 'LATITUDE' in ds.coords:
        lat = float(ds['LATITUDE'].values)
    elif 'LATITUDE' in ds.data_vars:
        val = ds['LATITUDE']
        if hasattr(val, 'values'):
            lat = float(val.values[0]) if val.size > 0 else None
        else:
            lat = float(val)
    else:
        lat = None

    # LON
    if 'LONGITUDE' in ds.coords:
        lon = float(ds['LONGITUDE'].values)
    elif 'LONGITUDE' in ds.data_vars:
        val = ds['LONGITUDE']
        if hasattr(val, 'values'):
            lon = float(val.values[0]) if val.size > 0 else None
        else:
            lon = float(val)
    else:
        lon = None

    return lat, lon

# ==========================================================
# ======= CASO NEW → CONCATENAZIONE FILE GIORNALIERI ======
# ==========================================================

if archive == "new":

    print("\nModalità NEW: concatenazione file giornalieri per stazione...")

    # Creazione cartella di output se non esiste
    new_output_dir.mkdir(parents=True, exist_ok=True)

    # Giorni richiesti
    bdates = pd.date_range(
        t_start_input.normalize(),
        t_end_input.normalize(),
        freq="D"
    )

    # Dizionario: nome_stazione → lista file giornalieri
    station_files = {}

    for d in bdates:
        day_dir = new_archive_dir / f"{d:%Y%m%d}"
        if not day_dir.exists():
            continue

        for f in day_dir.glob("*.nc"):
            parts = f.name.replace(".nc", "").split("_")
            name = parts[3] if len(parts) >= 4 else None
            if name is None:
                continue

            station_files.setdefault(name, []).append(f)

    if not station_files:
        raise FileNotFoundError("Nessun file trovato nell'intervallo richiesto.")

    print(f"Trovate {len(station_files)} stazioni da concatenare")

    # Creazione file concatenati per ogni stazione
    for name, files in station_files.items():
        files = sorted(files)
        datasets = []

        for f in files:
            ds = xr.open_dataset(f, decode_cf=False)

            # Taglio primo livello verticale se presente
            if 'DEPTH' in ds.dims:
                ds = ds.isel(DEPTH=0)

            # Accetta il file solo se contiene SLEV
            if 'SLEV' in ds.data_vars:
                # Seleziona SLEV + LATITUDE e LONGITUDE
                vars_to_keep = ['SLEV']
                for coord in ['LATITUDE', 'LONGITUDE']:
                    if coord in ds.coords or coord in ds.data_vars:
                        vars_to_keep.append(coord)
                ds = ds[vars_to_keep]
                datasets.append(ds)
            else:
                ds.close()  # chiudi subito se non c'è SLEV

        # Concatenazione lungo TIME, se c'è almeno un dataset valido
        if datasets:
            ds_concat = xr.concat(datasets, dim='TIME')

            # File di output
            out_file = new_output_dir / f"{name}_{t_start_input:%Y%m%d}_{t_end_input:%Y%m%d}.nc"

            if not out_file.exists():
                # Pulizia attributi globali e delle variabili
                ds_concat.attrs = {}
                for v in ds_concat.data_vars:
                    ds_concat[v].attrs = {}

                ds_concat.to_netcdf(out_file)
                print(f"Creato {out_file.name}")
            else:
                print(f"{out_file.name} già esistente")

            ds_concat.close()

            # Chiudi anche i dataset originali
            for ds in datasets:
                ds.close()
        else:
            print(f"Nessun file valido con SLEV per la stazione {name}. File saltato.")

    # Dopo la concatenazione, lavoriamo sui nuovi file
    data_dir = new_output_dir

# ==========================================================
# ================== LOOP SUI FILE =========================
# ==========================================================

nc_files = sorted(data_dir.glob("*.nc"))
selected = []

for nc_file in nc_files:
    try:
        ds = xr.open_dataset(nc_file, decode_times=True)

        # --- LAT/LON ---
        lat, lon = get_lat_lon(ds)
        if lat is None or lon is None:
           print ('issues with tg coordinates!')
           ds.close()
           continue

        print ('PROVA COO',lat,lon)

        if not (lat_min <= lat <= lat_max and lon_min <= lon <= lon_max):
            ds.close()
            continue

        if "TIME" not in ds.coords:
            print ('issues with tg time!')
            ds.close()
            continue

        time_vals = ds["TIME"].values
        print ('PROVA time',time_vals)
        start_time = pd.to_datetime(time_vals[0], utc=True)
        end_time   = pd.to_datetime(time_vals[-1], utc=True)

        # Accetta file se almeno una parte dell'intervallo rientra
        if end_time < t_start_input or start_time > t_end_input:
           ds.close()
           continue

        fname = nc_file.name
        parts = fname.replace(".nc", "").split("_")
        if archive == "old":
           name = parts[3] if len(parts) >= 4 else ""
        else:  # archive == "new"
           name = parts[0]
        print ('PROVA name',name)

        selected.append({
            "lat": lat,
            "lon": lon,
            "name": name,
            "path": str(nc_file.resolve())
        })

        ds.close()

    except Exception as e:
        print(f"Errore su {nc_file.name}: {e}")


# ==========================================================
# ================= SALVA CSV FINALE =======================
# ==========================================================

df = pd.DataFrame(selected, columns=["lat", "lon", "name", "path"])
df.to_csv(output_csv, index=False, sep=";")

print(f"\nCreato {output_csv} con {len(df)} file selezionati")
