import xarray as xr
import numpy as np
import pandas as pd
import geopandas as gpd
import cdsapi
import os
import zarr
import warnings
import matplotlib.pyplot as plt
import contextily as ctx
from datetime import timedelta
from tqdm import tqdm

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# =============================================================================
# 1. CONFIGURATION (EDIT THIS SECTION)
# =============================================================================



REGION_NAME = os.environ.get("CONUS_REGION", "region_06")  # override via CONUS_REGION env var
# ERA5 comes from the Copernicus Climate Data Store (CDS). Use your own free CDS key:
# set the CDSAPI_KEY environment variable, or put it in ~/.cdsapirc (see the CDS API how-to page).
CDS_KEY = os.environ.get("CDSAPI_KEY")  # None -> cdsapi reads ~/.cdsapirc


CONFIG = {
    # Time Range (Example: Short duration for testing)
    "start_time": "1980-01-01 00:00",
    "end_time": "2025-12-31 00:00",

    # Paths
    "base_path": f"data/atmospheric/{REGION_NAME}/",
    "poly_lr": f"GIS/regions/{REGION_NAME}/{REGION_NAME}_lr_boundary.shp",
    "grid_lr": f"GIS/regions/{REGION_NAME}/{REGION_NAME}_lr_latlon.npz",

    # API Keys for downloading ERA5 from CDS
    "cds_key": CDS_KEY
}

# =============================================================================
# 2. VARIABLE METADATA
# =============================================================================

REQUIRED_VARS = ['U10', 'V10', 'PSFC', 'MSLP']

# Our short name -> ERA5 CDS API name
ERA5_VAR_MAP = {
    'U10':  '10m_u_component_of_wind',
    'V10':  '10m_v_component_of_wind',
    'PSFC': 'surface_pressure',
    'MSLP': 'mean_sea_level_pressure',
}

# ERA5 short name in returned netCDF -> our standardized name
ERA5_RENAME = {'u10': 'U10', 'v10': 'V10', 'sp': 'PSFC', 'msl': 'MSLP'}

VAR_ATTRS = {
    'U10':  {'units': 'm s-1', 'long_name': '10-meter U wind component'},
    'V10':  {'units': 'm s-1', 'long_name': '10-meter V wind component'},
    'PSFC': {'units': 'hPa',   'long_name': 'Surface Pressure'},
    'MSLP': {'units': 'hPa',   'long_name': 'Mean Sea Level Pressure'},
}

PRESSURE_VARS = ('PSFC', 'MSLP')  # need Pa -> hPa conversion

def get_missing_vars(zarr_path, required=REQUIRED_VARS):
    """Return REQUIRED_VARS not present in an existing zarr (all of them if the
    zarr does not exist; empty list if the store cannot be read)."""
    if not os.path.exists(zarr_path):
        return list(required)
    try:
        ds = xr.open_zarr(zarr_path, consolidated=True)
        present = set(ds.data_vars)
        ds.close()
        return [v for v in required if v not in present]
    except Exception as e:
        print(f"Warning: cannot inspect {zarr_path}: {e}")
        return []

# =============================================================================
# 3. CORE PROCESSING FUNCTIONS
# =============================================================================

def download_era5_month(client, bbox, time_range, region_name, variables):
    """Downloads one month of the requested ERA5 variables, converts Pa->hPa
    for pressure fields, and sets attributes."""
    date_str = f"{time_range[0].strftime('%Y-%m-%d')}/{time_range[-1].strftime('%Y-%m-%d')}"
    min_lon, min_lat, max_lon, max_lat = bbox
    area = [max_lat, min_lon, min_lat, max_lon]  # N, W, S, E

    cds_vars = [ERA5_VAR_MAP[v] for v in variables]

    pid = os.getpid()
    temp_file = f"temp_era5_{region_name}_{pid}_{time_range[0].strftime('%Y%m%d%H')}.nc"

    try:
        client.retrieve(
            'reanalysis-era5-single-levels',
            {
                'product_type': 'reanalysis', 'format': 'netcdf',
                'variable': cds_vars,
                'date': date_str,
                'time': [f"{h:02d}:00" for h in range(24)],
                'area': area,
            },
            temp_file
        )
        ds = xr.open_dataset(temp_file)

        # 1. Standardize names (only rename what's present)
        rename_dict = {}
        for old, new in (('longitude', 'lon'), ('latitude', 'lat'), ('valid_time', 'time')):
            if old in ds.coords or old in ds.dims:
                rename_dict[old] = new
        for short, std in ERA5_RENAME.items():
            if short in ds.data_vars:
                rename_dict[short] = std
        ds = ds.rename(rename_dict)

        # 2. Pa -> hPa for pressure fields
        for var in PRESSURE_VARS:
            if var in ds.data_vars:
                ds[var] = ds[var] / 100.0

        # 3. Attributes
        for var in ds.data_vars:
            if var in VAR_ATTRS:
                ds[var].attrs = VAR_ATTRS[var]

        ds = ds.load()
        ds.close()
        os.remove(temp_file)
        return ds

    except Exception as e:
        if os.path.exists(temp_file): os.remove(temp_file)
        raise RuntimeError(f"ERA5 Download failed: {e}")

def interpolate_to_target_grid(ds, target_lats, target_lons):
    """Bilinear interpolation of a rectilinear (ERA5) dataset onto the target
    (y, x) lat/lon grid."""
    return ds.interp(
        lat=xr.DataArray(target_lats, dims=("y", "x")),
        lon=xr.DataArray(target_lons, dims=("y", "x")),
        method="linear",
    )

def save_optimized_zarr(ds, save_path, full_time_range, variables=None):
    """
    Fill temporal gaps, cast data to float16, and write to a Zstd-compressed
    Zarr store. Creates a new store if `save_path` does not exist; otherwise
    appends the requested variables to the existing store (mode='a'),
    overwriting any same-named variables already present.

    Args:
        ds (xarray.Dataset): Interpolated atmospheric dataset.
        save_path (str): Path of the Zarr store.
        full_time_range (pandas.DatetimeIndex): Contiguous hourly time index to
            reindex against.
        variables (list[str] | None): Subset of variables in `ds` to write.
            Defaults to every data_var in `ds`.

    Returns:
        bool: True if anything was written; False if `ds` is None or no
        requested variables are actually in `ds`.

    Zarr store layout (full):
        Dimensions:  time (hourly), y, x
        Coordinates: time (time), lat (y, x), lon (y, x)
        Data vars (each (time, y, x), float16, Blosc/Zstd-compressed):
            - U10  (10-m U wind, m s-1)
            - V10  (10-m V wind, m s-1)
            - PSFC (surface pressure, hPa)
            - MSLP (mean sea-level pressure, hPa)
    """
    if ds is None:
        return False

    if variables is None:
        variables = list(ds.data_vars)
    variables = [v for v in variables if v in ds.data_vars]
    if not variables:
        return False

    # Continuity & filling
    ds_cont = ds.reindex(time=full_time_range, fill_value=np.nan)

    # Subset to requested vars and cast to float16
    ds_final = ds_cont[variables].copy()
    for var in variables:
        ds_final[var] = ds_final[var].astype(np.float16)

    compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=2)
    encoding = {var: {'compressor': compressor} for var in variables}

    if os.path.exists(save_path):
        # Append new variables; drop non-dim coords (lat/lon) so we don't
        # rewrite them on the existing store.
        non_dim_coords = [c for c in ds_final.coords if c not in ds_final.dims]
        ds_to_append = ds_final.reset_coords(non_dim_coords, drop=True) if non_dim_coords else ds_final
        ds_to_append.to_zarr(save_path, mode='a', encoding=encoding, consolidated=True)
    else:
        ds_final.to_zarr(save_path, mode='w', encoding=encoding, consolidated=True)
    return True

def visualize_atmospheric_state(
    ds: xr.Dataset, 
    target_time: str, 
    title: str = "Atmospheric State",
    vector_stride: int = 1,
    extent: list = None
):
    """
    Visualizes Pressure (hPa) and Wind in EPSG:4326 (Lat/Lon).
    """
    print(f"Visualizing {title} for {target_time}...")

    # 1. Select Time Slice
    try:
        t_sel = pd.to_datetime(target_time)
        if 'time' in ds.dims and ds.sizes['time'] > 1:
            data = ds.sel(time=t_sel, method='nearest').squeeze()
        else:
            data = ds.squeeze()
        data = data.compute()
    except KeyError:
        print(f"Time {target_time} not found.")
        return

    # 2. Coordinates
    lons = data['lon'].values
    lats = data['lat'].values
    if lons.ndim == 1 and lats.ndim == 1:
        lons, lats = np.meshgrid(lons, lats)
        
    # 3. Setup Plot
    fig, ax = plt.subplots(figsize=(12, 10))

    if extent is not None:
        ax.set_xlim(extent[0], extent[2])
        ax.set_ylim(extent[1], extent[3])
    else:
        ax.set_xlim(lons.min(), lons.max())
        ax.set_ylim(lats.min(), lats.max())

    # 4. Base Map
    try:
        ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, crs="EPSG:4326", alpha=1.0, zorder=0)
    except Exception as e:
        print(f"Basemap warning: {e}")

    # 5. Plot Data (Pressure is ALREADY hPa)
    pressure = data['PSFC'].values
    
    mesh = ax.pcolormesh(
        lons, lats, pressure, 
        cmap='coolwarm', 
        alpha=0.5, 
        shading='auto', 
        zorder=1 
    )
    cb = plt.colorbar(mesh, ax=ax, label="Surface Pressure (hPa)", pad=0.02)

    # 6. Plot Vectors
    u = data['U10'].values
    v = data['V10'].values
    sl = (slice(None, None, vector_stride), slice(None, None, vector_stride))
    
    q = ax.quiver(
        lons[sl], lats[sl], u[sl], v[sl],
        scale=300, width=0.002, color='black', alpha=0.8, zorder=2, label="10m Wind"
    )
    ax.quiverkey(q, 0.9, 1.05, 20, r'$20 \frac{m}{s}$', labelpos='E', coordinates='axes')

    ax.set_title(f"{title}\n{target_time}", fontsize=14)
    ax.set_xlabel("Longitude (°)", fontsize=12)
    ax.set_ylabel("Latitude (°)", fontsize=12)
    ax.ticklabel_format(useOffset=False, style='plain')

    plt.tight_layout()
    plt.show()
# =============================================================================
# 4. MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    print(f"--- Initializing (region={REGION_NAME}) ---")
    
    # 1. Load Geometries
    lr_poly = gpd.read_file(CONFIG['poly_lr']).to_crs("EPSG:4326")
    lr_grid = np.load(CONFIG['grid_lr'])
    
    # 2. Setup API
    cds = cdsapi.Client(url="https://cds.climate.copernicus.eu/api", key=CONFIG['cds_key'])

    # 3. Create Directory
    path_lr = os.path.join(CONFIG['base_path'], "lr_grid")
    os.makedirs(path_lr, exist_ok=True)

    # --- Time Loop (Handles both long periods and short 2-day tests) ---
    
    # A. Generate complete hourly range required
    full_range = pd.date_range(start=CONFIG['start_time'], end=CONFIG['end_time'], freq='h')
    
    # B. Identify unique Year-Month combinations in this range
    #    This ensures we process the correct monthly files even if range < 1 month
    unique_periods = full_range.to_period('M').unique()
    
    print(f"Processing {len(unique_periods)} monthly chunks for range: {CONFIG['start_time']} to {CONFIG['end_time']}")

    for period in tqdm(unique_periods, desc="Processing Months"):
        
        # Determine the start/end of this specific month
        # We process the whole month OR just the slice requested if it's a partial month?
        # Architecture decision: File is named YYYYMM.zarr. It should ideally contain the full month
        # or at least the continuous chunk requested.
        # For simplicity in this script: We stick to the global start/end constraints.
        
        # Calculate intersection of (This Month) and (Global Request)
        month_start = period.to_timestamp()
        month_end = (period + 1).to_timestamp() - timedelta(hours=1)
        
        # Intersect with global request
        chunk_start = max(pd.to_datetime(CONFIG['start_time']), month_start)
        chunk_end = min(pd.to_datetime(CONFIG['end_time']), month_end)
        
        # Hourly range for this chunk (used for Zarr index)
        chunk_hourly = pd.date_range(chunk_start, chunk_end, freq='h')

        # File name is YYYYMM.zarr
        fname = f"{period.year}{period.month:02d}.zarr"

        # --- ERA5 ---
        out_path = os.path.join(path_lr, fname)
        missing_lr = get_missing_vars(out_path)
        if missing_lr:
            try:
                bounds = lr_poly.total_bounds + [-0.5, -0.5, 0.5, 0.5] # Buffer
                ds = download_era5_month(cds, bounds, chunk_hourly, REGION_NAME, variables=missing_lr)
                ds_interp = interpolate_to_target_grid(ds, lr_grid['lats'], lr_grid['lons'])
                save_optimized_zarr(ds_interp, out_path, chunk_hourly, variables=missing_lr)
            except Exception as e:
                tqdm.write(f"ERA5 Fail {fname}: {e}")

    print("\nProcessing Complete.")


    # =========================================================================
    # 5. EXAMPLE USAGE: VISUALIZATION
    # =========================================================================
    # Skip the matplotlib viz when running headless / in batch (env var set).
    if os.environ.get("CONUS_REGION"):
        import sys; sys.exit(0)
    print("\n--- Generating Visualization ---")
    
    # Pick a time to plot (e.g., the middle of our requested range)
    plot_time = full_range[len(full_range)//2].strftime("%Y-%m-%d %H:%M")
    plot_time = '2011-08-27 23:00'
    plot_year_month = full_range[len(full_range)//2].strftime("%Y%m")
    common_extent = lr_poly.total_bounds
    
    # Load the specific Zarr file we just created
    try:
        path_lr_zarr = os.path.join(path_lr, f"{plot_year_month}.zarr")
        if os.path.exists(path_lr_zarr):
            ds_lr = xr.open_zarr(path_lr_zarr, consolidated=True)
            visualize_atmospheric_state(ds_lr, plot_time, title="ERA5 (Low Res)", vector_stride=1, extent=common_extent)
        else:
            print(f"File not found: {path_lr_zarr}")
    except Exception as e:
        print(f"Visualization Error: {e}")

