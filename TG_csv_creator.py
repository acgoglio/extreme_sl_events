import xarray as xr
from pathlib import Path
import pandas as pd

# ==========================
# INPUT UTENTE
# ==========================
data_dir = Path("/data/inputs/METOCEAN/historical/obs/ocean/in_situ/CMS/MedSea/history_EIS_202511/tidegauge/")
output_csv = Path("TGs_daniel.coo")

# Intervallo temporale
t_start_input = pd.to_datetime("2023-08-15")
t_end_input   = pd.to_datetime("2023-09-20")

# Lon-Lat Box:
lat_min, lat_max = 30.0, 40.0
lon_min, lon_max = 12.0, 36.0

# Gloria 2020
# t_start_input = pd.to_datetime("2020-01-01")
# t_end_input   = pd.to_datetime("2020-02-01")
# lat_min, lat_max = 34.6, 44.6
# lon_min, lon_max = -4.0, 8.7
# 
# Ianos 2020
# t_start_input = pd.to_datetime("2020-09-01")
# t_end_input   = pd.to_datetime("2020-10-01")
# lat_min, lat_max = 30.0, 41.0
# lon_min, lon_max = 9.0, 36.0
#
# Apollo 2021
# t_start_input = pd.to_datetime("2021-10-05")
# t_end_input   = pd.to_datetime("2021-11-05")
# lat_min, lat_max = 30.0, 38.0
# lon_min, lon_max = 9.0, 36.0
#
# Blas 2021
# t_start_input = pd.to_datetime("2021-10-25")
# t_end_input   = pd.to_datetime("2021-11-30")
# lat_min, lat_max = 35.0, 43.0
# lon_min, lon_max = 0.0, 16.0
#
# Venice Acqua Alta 2022
#t_start_input = pd.to_datetime("2022-11-01")
#t_end_input   = pd.to_datetime("2022-12-01")
#lat_min, lat_max = 40.0, 46.0
#lon_min, lon_max = 12.0, 22.0
#
# Daniel 2023
#t_start_input = pd.to_datetime("2023-08-15")
#t_end_input   = pd.to_datetime("2023-19-20")
#lat_min, lat_max = 30.0, 40.0
#lon_min, lon_max = 12.0, 36.0
#
# Harry 2026
#t_start_input = pd.to_datetime("2025-12-23")
#t_end_input   = pd.to_datetime("2026-01-23")
#lat_min, lat_max = 30.0, 42.0
#lon_min, lon_max = -2.0, 30.0

# ==========================
# LOOP SUI FILE
# ==========================
nc_files = sorted(data_dir.glob("*.nc"))
selected = []

for nc_file in nc_files:
    try:
        ds = xr.open_dataset(nc_file, decode_times=True)

        # --- Coordinate ---
        lat = ds.coords.get("LATITUDE", None)
        lon = ds.coords.get("LONGITUDE", None)

        if lat is None or lon is None:
            ds.close()
            continue

        lat = float(lat.values)
        lon = float(lon.values)

        # filtro spaziale
        if not (lat_min <= lat <= lat_max and lon_min <= lon <= lon_max):
            ds.close()
            continue

        # --- Tempo ---
        if "TIME" not in ds.coords:
            ds.close()
            continue

        time_vals = ds["TIME"].values
        start_time = pd.to_datetime(time_vals[0])
        end_time   = pd.to_datetime(time_vals[-1])

        # filtro temporale: intervallo incluso
        if not (start_time <= t_start_input and end_time >= t_end_input):
            ds.close()
            continue

        # --- Estrazione name dal filename ---
        fname = nc_file.name
        parts = fname.replace(".nc", "").split("_")

        if len(parts) >= 4:
           name = parts[3]
        else:
           name = ""

        # --- Salva info ---
        selected.append({
            "lat": lat,
            "lon": lon,
            "name": name,
            "path": str(nc_file.resolve())
        })

        ds.close()

    except Exception as e:
        print(f"Errore su {nc_file.name}: {e}")

# ==========================
# SALVA CSV FINALE
# ==========================
df = pd.DataFrame(selected, columns=["lat", "lon", "name", "path"])
df.to_csv(output_csv, index=False, sep=";")

print(f"Creato {output_csv} con {len(df)} file selezionati")
