import os
import requests
import pandas as pd
import geopandas as gpd
from pathlib import Path

# --- Configuration ---
OUTPUT_DIR = Path('data/stations')
REGIONS_DIR = Path('GIS/regions')
BULK_URL = "https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations.json"

def fetch_noaa_stations():
    """Fetches station metadata, filters to CONUS, then fetches datums/floods for Active only."""
    print("1. Fetching bulk station list...")
    
    # Get active and historic stations (basic metadata only)
    try:
        active_data = requests.get(BULK_URL, params={'type': 'waterlevels'}).json().get('stations', [])
        active_ids = {st['id'] for st in active_data}
        
    except Exception as e:
        print(f"   ! Error fetching active stations: {e}")
        return pd.DataFrame()

    try:
        historic_data = requests.get(BULK_URL, params={'type': 'historicwl'}).json().get('stations', [])
    except Exception as e:
        print(f"   ! Error fetching historic stations: {e}")
        return pd.DataFrame()

    # Merge safely
    all_stations_dict = {st['id']: st for st in historic_data}
    for st in active_data:
        all_stations_dict[st['id']] = st
        
    all_data = list(all_stations_dict.values())
    filtered_stations = []

    # Filter down to CONUS FIRST to save API calls
    for st in all_data:
        try:
            lat = float(st['lat'])
            lon = float(st['lng'])
        except (ValueError, KeyError, TypeError):
            continue

        if not (22 <= lat <= 51 and -126 <= lon <= -66):
            continue
        if st.get('greatlakes'):
            continue

        filtered_stations.append({
            'station_id': st['id'],
            'name': st.get('name', 'Unknown'),
            'state': st.get('state', ''),
            'lat': lat,
            'lon': lon,
            'status': 'Active' if st['id'] in active_ids else 'Historic',
            'affil': st.get('affiliations') or "", 
            'is_gl': 0
        })

    print(f"   -> Filtered down to {len(filtered_stations)} CONUS stations.")
    print("2. Fetching Datums & Flood Levels (Active Stations Only)...")

    session = requests.Session()
    processed_data = []

    def calculate_offset(val, datum_val):
        try:
            return round(float(val) - float(datum_val), 3)
        except (TypeError, ValueError):
            return None

    active_count = sum(1 for st in filtered_stations if st['status'] == 'Active')
    processed_active = 0

    # Loop through filtered stations, but only query API if Active
    for st in filtered_stations:
        # Initialize empty variables so Historic stations get standard None values
        d_mllw, d_stnd, d_msl, d_mhhw, d_navd88 = None, None, None, None, None
        nws_minor, nws_mod, nws_major = None, None, None

        if st['status'] == 'Active':
            processed_active += 1
            if processed_active % 50 == 0:
                print(f"   -> Queried {processed_active}/{active_count} active stations...")

            st_id = st['station_id']
            detail_url = f"https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations/{st_id}.json"
            
            try:
                # Query API in metric units
                resp = session.get(detail_url, params={'expand': 'floodlevels,datums', 'units': 'metric'}, timeout=10)
                if resp.status_code == 200:
                    st_detail = resp.json().get('stations', [{}])[0]
                    
                    # Extract Flood Levels
                    flood_levels = st_detail.get('floodlevels', {})
                    if isinstance(flood_levels, list):  
                        flood_levels = flood_levels[0] if flood_levels else {}
                    elif not isinstance(flood_levels, dict):
                        flood_levels = {}

                    nws_minor = flood_levels.get('nws_minor') or flood_levels.get('nos_minor')
                    nws_mod = flood_levels.get('nws_moderate') or flood_levels.get('nos_moderate')
                    nws_major = flood_levels.get('nws_major') or flood_levels.get('nos_major')

                    # Extract Datums
                    datums_data = st_detail.get('datums', {})
                    datum_list = datums_data.get('datums', []) if isinstance(datums_data, dict) else datums_data
                    if not isinstance(datum_list, list):
                        datum_list = []
                        
                    d_dict = {d.get('name', '').lower(): d.get('value') for d in datum_list if isinstance(d, dict)}

                    d_mllw = d_dict.get('mllw')
                    d_stnd = d_dict.get('stnd')
                    d_msl = d_dict.get('msl')
                    d_mhhw = d_dict.get('mhhw')
                    d_navd88 = d_dict.get('navd88')
            except Exception:
                pass # If it fails, variables remain None

        # Append data 
        st.update({
            # Raw Datums (in meters, referenced to STND)
            'd_mllw_m': d_mllw,
            'd_stnd_m': d_stnd,
            'd_msl_m': d_msl,
            'd_mhhw_m': d_mhhw,
            'd_navd_m': d_navd88,
            
            # Calculated Flood Levels (in meters, referenced to MSL)
            'f_min_ms_m': calculate_offset(nws_minor, d_msl),
            'f_mod_ms_m': calculate_offset(nws_mod, d_msl),
            'f_maj_ms_m': calculate_offset(nws_major, d_msl)
        })
        processed_data.append(st)

    return pd.DataFrame(processed_data)

def add_region_data(gdf, regions_root):
    """Performs spatial join with region shapefiles."""
    print("3. Assigning Regions...")
    
    if not regions_root.exists():
        print(f"   ! Regions directory '{regions_root}' not found. Skipping.")
        return gdf

    region_files = sorted(regions_root.rglob("*_tidegauge_boundary.shp"))
    
    if not region_files:
        print("   ! No shapefiles matching '*_tidegauge_boundary.shp' found. Skipping.")
        return gdf

    gdf['region'] = -1
    stations_assigned = 0

    for shp_path in region_files:
        try:
            parts = shp_path.stem.split('_')
            if len(parts) > 1 and parts[1].isdigit():
                region_id = int(parts[1])
            else:
                continue

            poly_gdf = gpd.read_file(shp_path)
            
            if poly_gdf.crs != gdf.crs:
                poly_gdf = poly_gdf.to_crs(gdf.crs)

            stations_in_region = gpd.sjoin(gdf, poly_gdf, how="inner", predicate="within")
            
            if not stations_in_region.empty:
                gdf.loc[stations_in_region.index, 'region'] = region_id
                stations_assigned += len(stations_in_region)
                
        except Exception as e:
            print(f"   ! Error processing {shp_path.name}: {e}")
            continue

    print(f"   -> Assigned regions to {stations_assigned} stations.")
    return gdf

# --- Main Block ---
if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = fetch_noaa_stations()
    
    if df.empty:
        print("No stations found or error occurred. Exiting.")
    else:
        gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df.lon, df.lat),
            crs="EPSG:4326"
        )

        gdf = add_region_data(gdf, REGIONS_DIR)

        if 'region' in gdf.columns:
            print("\n   --- Station Counts by Region ---")
            unique_regions = sorted(gdf['region'].unique())
            for reg_id in unique_regions:
                region_df = gdf[gdf['region'] == reg_id]
                total_count = len(region_df)
                active_count = len(region_df[region_df['status'] == 'Active'])
                label = f"Region {int(reg_id)}" if reg_id != -1 else "Unassigned"
                print(f"   {label}: {total_count}, {active_count} active")
            print("   --------------------------------\n")

        # Rename to stay under Shapefile limits
        shp_gdf = gdf.rename(columns={'station_id': 'st_id', 'name': 'st_name'})
        
        shp_path = OUTPUT_DIR / 'noaa_stations_conus.shp'
        try:
            shp_gdf.to_file(shp_path)
            print(f"4. Saved Shapefile: {shp_path}")
        except Exception as e:
            print(f"   ! Error saving Shapefile: {e}")

        csv_path = OUTPUT_DIR / 'noaa_stations_conus.csv'
        pd.DataFrame(gdf.drop(columns='geometry')).to_csv(csv_path, index=False)
        print(f"5. Saved CSV: {csv_path}")