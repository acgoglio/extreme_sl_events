import xarray as xr
from pathlib import Path
import pandas as pd

# --------------------------
# Imposta la cartella con i netCDF
# --------------------------
data_dir = Path("/data/inputs/METOCEAN/historical/obs/ocean/in_situ/CMS/MedSea/history_EIS_202511/tidegauge/")
output_csv = Path("TG_obs_inventory.csv")

# Lista di tutti i file netCDF nella cartella
nc_files = sorted(data_dir.glob("*.nc"))

# Lista per salvare info
inventory = []

for nc_file in nc_files:
    try:
        ds = xr.open_dataset(nc_file, decode_times=True)  # decodifica le date CF
        
        # Estrazione nome variabili
        var_names = list(ds.data_vars.keys())
        
        # Coordinate
        lat = ds.coords.get('LATITUDE', None)
        lon = ds.coords.get('LONGITUDE', None)
        lat = float(lat.values) if lat is not None else None
        lon = float(lon.values) if lon is not None else None
        
        # Range temporale usando TIME
        if 'TIME' in ds.coords:
            time_vals = ds['TIME'].values
            try:
                start_time = pd.to_datetime(time_vals[0])
                end_time = pd.to_datetime(time_vals[-1])
            except Exception:
                start_time = str(time_vals[0])
                end_time = str(time_vals[-1])
        else:
            start_time = None
            end_time = None
        
        # Salva info
        inventory.append({
            "file": nc_file.name,
            "lat": lat,
            "lon": lon,
            "variables": ", ".join(var_names),
            "start_time": start_time,
            "end_time": end_time
        })
        
        ds.close()
        
    except Exception as e:
        print(f"Errore su {nc_file.name}: {e}")

# --------------------------
# Salva il risultato in CSV
# --------------------------
df = pd.DataFrame(inventory)
df.to_csv(output_csv, index=False)
print(f"Inventory salvato in {output_csv}")
