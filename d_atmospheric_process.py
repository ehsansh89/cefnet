import xarray as xr
import numpy as np
import os
import glob
import plotly.graph_objects as go
import pandas as pd
from pathlib import Path

# =============================================================================
# CONFIGURATION & THRESHOLDS
# =============================================================================
REGION_NAME = os.environ.get("CONUS_REGION", "region_06")  # override via CONUS_REGION env var
BASE_DIR = Path(f"data/atmospheric/{REGION_NAME}")
QA_DIR = BASE_DIR / "qa"
QA_DIR.mkdir(parents=True, exist_ok=True)

GRIDS = ["lr_grid"]
VARIABLES = ['U10', 'V10', 'PSFC', 'MSLP']

LIMITS = {
    'U10':  (-100.0, 100.0),
    'V10':  (-100.0, 100.0),
    'PSFC': (700.0, 1300.0),  # surface pressure varies with elevation
    'MSLP': (850.0, 1100.0),  # tighter; record TC minimum ~870 hPa
}

def check_time_continuity(times: pd.DatetimeIndex, grid_name: str):
    if times.empty: return ""
    expected_range = pd.date_range(start=times.min(), end=times.max(), freq='h')
    missing = expected_range.difference(times)
    duplicates = times[times.duplicated()]
    
    report = f"\n--- Time Continuity Report: {grid_name} ---\n"
    if len(missing) == 0 and len(duplicates) == 0:
        report += "✅ Continuous: No missing or duplicate hours found.\n"
    else:
        if len(missing) > 0: report += f"❌ MISSING: {len(missing)} timestamps.\n"
        if len(duplicates) > 0: report += f"❌ DUPLICATES: {len(duplicates)} timestamps.\n"
    print(report)
    return report

# =============================================================================
# DATA PROCESSING
# =============================================================================
all_results = {grid: [] for grid in GRIDS}

for grid in GRIDS:
    search_path = BASE_DIR / grid / "*.zarr"
    files = sorted(glob.glob(str(search_path)))
    if not files: continue

    total_time_indices = []
    print(f"\nProcessing Grid: {grid.upper()}")
    header_cols = ['FILE'.ljust(15)]
    for var in VARIABLES:
        header_cols.append(f'{var}_MIN'.ljust(9))
        header_cols.append(f'{var}_MAX'.ljust(9))
    header_cols.append('STATUS')
    header = ' | '.join(header_cols)
    print(header)
    print("-" * len(header))

    for file_path in files:
        file_name = os.path.basename(file_path)
        row_data = {'file': file_name, 'grid': grid}
        file_issues = []

        try:
            ds = xr.open_zarr(file_path, consolidated=True)
            
            # 1. Strict Time Check
            if 'time' not in ds.coords:
                raise KeyError(f"Critical Error: 'time' coordinate missing in {file_name}")
            
            f_times = pd.to_datetime(ds.time.values)
            total_time_indices.extend(f_times)
            row_data['time'] = f_times[0]

            # 2. Variable Extraction & Physical Check
            for var in VARIABLES:
                if var in ds:
                    data = ds[var].values
                    v_min, v_max = np.nanmin(data), np.nanmax(data)
                    
                    has_nan_inf = np.isnan(data).any() or np.isinf(data).any()
                    lower, upper = LIMITS[var]
                    out_of_bounds = (v_min < lower) or (v_max > upper)

                    if has_nan_inf or out_of_bounds:
                        row_data[f'{var}_min'], row_data[f'{var}_max'] = np.nan, np.nan
                        row_data[f'{var}_issue'] = True
                        flag = "BAD_VAL" if has_nan_inf else "OUT_OF_RANGE"
                        file_issues.append(f"{var}({flag})")
                    else:
                        row_data[f'{var}_min'], row_data[f'{var}_max'] = v_min, v_max
                        row_data[f'{var}_issue'] = False
                else:
                    row_data[f'{var}_min'], row_data[f'{var}_max'] = np.nan, np.nan
                    file_issues.append(f"MISSING_{var}")

            status_str = "✅ OK" if not file_issues else f"❌ {','.join(file_issues)}"
            row_cols = [file_name.ljust(15)]
            for var in VARIABLES:
                vmin = row_data.get(f'{var}_min', np.nan)
                vmax = row_data.get(f'{var}_max', np.nan)
                row_cols.append(f'{vmin:<9.2f}' if not np.isnan(vmin) else 'NaN'.ljust(9))
                row_cols.append(f'{vmax:<9.2f}' if not np.isnan(vmax) else 'NaN'.ljust(9))
            row_cols.append(status_str)
            print(' | '.join(row_cols))

            all_results[grid].append(row_data)

        except Exception as e:
            print(f"{file_name:<15} | CRITICAL ERROR: {str(e)}")

    check_time_continuity(pd.DatetimeIndex(total_time_indices), grid)

# =============================================================================
# SAVE RESULTS & PLOT
# =============================================================================
# Save CSV Report
full_df = pd.concat([pd.DataFrame(all_results[g]) for g in GRIDS])
csv_path = QA_DIR / "diagnostic_report.csv"
full_df.to_csv(csv_path, index=False)
print(f"\nCSV Report saved to: {csv_path}")

# Plotting with auto-save to HTML
for var in VARIABLES:
    fig = go.Figure()
    for grid in GRIDS:
        df = full_df[full_df['grid'] == grid].sort_values('time')
        if df.empty: continue

        fig.add_trace(go.Scatter(
            x=df['time'], y=df[f'{var}_max'],
            name=f"{grid.upper()} Max", mode='lines+markers',
            line=dict(color='#1f77b4', width=1.5),
            marker=dict(size=3),
            legendgroup=grid,
            connectgaps=False
        ))

        fig.add_trace(go.Scatter(
            x=df['time'], y=df[f'{var}_min'],
            name=f"{grid.upper()} Min", mode='lines',
            line=dict(color='#1f77b4', width=1, dash='dot'),
            legendgroup=grid,
            connectgaps=False
        ))

    fig.update_layout(
        title=f"QA Diagnostic: {var}",
        xaxis_title="Time", yaxis_title=f"{var} Units",
        template="plotly_white", hovermode="x unified"
    )
    
    plot_path = QA_DIR / f"diagnostic_{var}.html"
    fig.write_html(str(plot_path))
    print(f" Plot saved to: {plot_path}")
    if not os.environ.get("CONUS_REGION"):  # skip auto-open in batch mode
        fig.show()