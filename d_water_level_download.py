import os
import pandas as pd
import requests
import numpy as np
from pathlib import Path
import time
import json
import random
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# --- Configuration ---
INPUT_CSV = Path('data/stations/noaa_stations_conus.csv')
OUTPUT_DIR = Path('data/water_levels')

START_DATE = "1980-01-01"
END_DATE = "2025-12-31"

# NOAA API Config
BASE_URL = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"

# --- Setup Robust Session ---
def get_session():
    """
    Creates and configures a robust requests.Session instance.

    Features:
    - User-Agent spoofing to mimic a standard browser (Chrome on Windows).
    - Automatic Retry logic:
        - Retries up to 5 times.
        - Backoff factor of 2 (wait times: 2s, 4s, 8s, etc.).
        - Retries on specific HTTP status codes (429, 500, 502, 503, 504).

    Returns:
        requests.Session: A configured session object ready for API requests.
    """
    session = requests.Session()
    
    # Header makes us look like Chrome on Windows
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    session.headers.update(headers)

    retry = Retry(
        total=5,
        backoff_factor=2,  # Wait 2s, 4s, 8s...
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    return session

session = get_session()

def fetch_year_data(station_id, year, product, datum, interval=None):
    """
    Fetches one year of water level data from the NOAA CO-OPS API.

    Args:
        station_id (str): The 7-digit station identifier.
        year (int): The year to fetch.
        product (str): Data product type (e.g., 'hourly_height', 'predictions').
        datum (str): The vertical reference datum (e.g., 'MSL', 'STND').
        interval (str, optional): Data interval (e.g., 'h' for hourly). Defaults to None.

    Returns:
        pd.DataFrame: A DataFrame containing the fetched data. 
                      Returns an empty DataFrame if the request fails, 
                      no data is found, or JSON decoding errors occur.
    
    Raises:
        PermissionError: If the API returns a 403 Forbidden status.
    """
    params = {
        "station": station_id,
        "product": product,
        "datum": datum,
        "units": "metric",
        "time_zone": "gmt",
        "format": "json",
        "begin_date": f"{year}-01-01",
        "end_date": f"{year}-12-31",
        "application": "DataScript_SafeMode"
    }
    if interval:
        params["interval"] = interval

    try:
        # TIMEOUT: 30 seconds to be generous
        response = session.get(BASE_URL, params=params, timeout=30)
        
        # If we are banned (403), raise error immediately to stop script
        if response.status_code == 403:
            raise PermissionError("403 Forbidden - IP is blocked.")
            
        if response.status_code != 200:
            return pd.DataFrame()

        try:
            data = response.json()
        except json.JSONDecodeError:
            return pd.DataFrame()

        if 'error' in data:
            return pd.DataFrame()

        key = 'predictions' if product == 'predictions' else 'data'
        if key in data:
            return pd.DataFrame(data[key])

    except PermissionError as e:
        raise e # Re-raise to stop the main loop
    except Exception:
        pass # Ignore minor connection blips
        
    return pd.DataFrame()

def process_station(station_row):
    """
    Downloads observed and predicted water level data for a specific station,
    combines them, and saves the result to a compressed HDF5 file with metadata.

    HDF5 File Structure ({station_id}.h5):
    --------------------------------------
    1. Group/Key: '/water_levels'
       - Type: pandas.DataFrame
       - Index: DatetimeIndex (GMT/UTC)
       - Columns:
           - 'observed' (float16): Observed water levels.
           - 'predicted' (float16): Predicted water levels.
    
    2. Attributes (accessible via store.get_storer('water_levels').attrs.metadata):
       - 'station_id' (str): The NOAA station ID.
       - 'units' (str): 'metric'.
       - 'time_zone' (str): 'gmt'.
       - 'datums_found' (list): List of datums successfully fetched (e.g., ['MSL', 'STND']).
       - 'source' (str): 'NOAA CO-OPS API'.

    Args:
        station_row (pd.Series): A row from the input CSV containing 'station_id'.
    """
    st_id = str(int(float(station_row['station_id'])))
    file_path = OUTPUT_DIR / f"{st_id}.h5"
    
    if file_path.exists():
        print(f"Skipping {st_id} (Already Downloaded)")
        return

    print(f"--> Processing {st_id}...", end="", flush=True)

    obs_frames = []
    pred_frames = []
    
    # We will track which datums we actually found data for
    detected_datums = set()

    start_year = int(START_DATE.split('-')[0])
    end_year = int(END_DATE.split('-')[0])

    for year in range(start_year, end_year + 1):
        # --- 1. Fetch Observed ---
        # Try MSL first
        datum_used = "MSL"
        obs = fetch_year_data(st_id, year, "hourly_height", datum_used)
        
        # If MSL failed, try STND
        if obs.empty:
            datum_used = "STND"
            obs = fetch_year_data(st_id, year, "hourly_height", datum_used)
        
        if not obs.empty:
            obs_frames.append(obs)
            detected_datums.add(datum_used)

        # --- 2. Fetch Predictions ---
        # Predictions usually default to station datum or specific tidal datums, 
        # but we request MSL to match.
        pred = fetch_year_data(st_id, year, "predictions", "MSL", interval="h")
        if not pred.empty:
            pred_frames.append(pred)

        time.sleep(random.uniform(0.2, 0.5))

    # --- SAVE ---
    if not obs_frames and not pred_frames:
        print(" No Data Found.")
        return

    try:
        df_obs = pd.concat(obs_frames) if obs_frames else pd.DataFrame()
        df_pred = pd.concat(pred_frames) if pred_frames else pd.DataFrame()

        if not df_obs.empty:
            df_obs['t'] = pd.to_datetime(df_obs['t'])
            df_obs.set_index('t', inplace=True)
            df_obs = df_obs[~df_obs.index.duplicated()]
            df_obs = df_obs[['v']].rename(columns={'v': 'observed'}).apply(pd.to_numeric, errors='coerce')

        if not df_pred.empty:
            df_pred['t'] = pd.to_datetime(df_pred['t'])
            df_pred.set_index('t', inplace=True)
            df_pred = df_pred[~df_pred.index.duplicated()]
            df_pred = df_pred[['v']].rename(columns={'v': 'predicted'}).apply(pd.to_numeric, errors='coerce')

        final_df = df_obs.join(df_pred, how='outer').astype(np.float16)
        
        # --- NEW: Save with Metadata ---
        with pd.HDFStore(file_path, mode='w', complevel=9, complib='blosc') as store:
            # Save the data
            store.put('water_levels', final_df)
            
            # Save the attributes (Metadata)
            metadata = {
                'station_id': st_id,
                'units': 'metric',
                'time_zone': 'gmt',
                'datums_found': list(detected_datums), # e.g., ['MSL'] or ['STND'] or both
                'source': 'NOAA CO-OPS API'
            }
            store.get_storer('water_levels').attrs.metadata = metadata
            
        print(f" Saved ({len(final_df)} rows) | Datum: {list(detected_datums)}")

    except Exception as e:
        print(f" Error saving: {e}")

# --- Main Block ---
if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not INPUT_CSV.exists():
        print(f"Error: {INPUT_CSV} not found.")
        exit(1)

    df_stations = pd.read_csv(INPUT_CSV)
    df_stations = df_stations[df_stations['status'] == 'Active']
    
    total = len(df_stations)
    print(f"Starting SEQUENTIAL download for {total} stations.")
    print("Press Ctrl+C to pause/stop safely.\n")

    try:
        for index, row in df_stations.iterrows():
            process_station(row)
            
            # Extra pause between stations to be super safe
            time.sleep(0.1)
            
    except PermissionError:
        print("\n\nCRITICAL: API denied access (403). You are still blocked. Try again later.")
    except KeyboardInterrupt:
        print("\n\nStopped by user.")
    
    print("\nProcessing Complete.")