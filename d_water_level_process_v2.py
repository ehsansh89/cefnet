"""
d_water_level_process_v2.py — revised water-level detrending.

Change vs v1 (d_water_level_process.py):
  * NO neighbor borrowing. Each station is detrended from its OWN record.
  * Detrend removes ONLY the secular sea-level trend (SLR + local subsidence),
    estimated by OLS on ANNUAL MEANS of the non-tidal residual (robust to
    seasonal/storm/end effects). Seasonal + interannual variability is KEPT in
    the surge target on purpose: the model has a lookback but no time input, so
    it learns the current elevated baseline from recent obs, while the secular
    monotonic drift (which it cannot anchor without a date) is what we remove.
  * Trend line is NOT forced through the 1992 NTDE point (that 30-yr lever arm
    amplified slope error). The full OLS line (slope + intercept) is removed, so
    the surge target is zero-mean AND zero-trend for every station; this also
    cancels the STND-vs-MSL datum inconsistency in the model target.
  * Stations kept only if they have >= MIN_SPAN_YEARS of their own data reaching
    a recent year (no short-record patching).

Pipeline per station:
    NTR(t)   = observed - predicted                         (non-tidal residual)
    slope    = OLS slope of annual means of NTR             (m/day, secular only)
    line(t)  = slope*ord(t) + c,  c set so surge is zero-mean
    surge(t) = NTR(t) - line(t)                             (model target = residual)
    observed_detrended = observed - line(t)  ( = predicted + surge )

Reconstruction at inference:
    WL_abs(t) = predicted(t) + surge_model(t) + line(t)
    line(t)   = slope_m_per_day*(ord(t) - ord(2025-01-01)) + msl_2025_m

Output schema matches v1 (drop-in for s_sample_batch / s_scalers / s_build_catalog):
    observed, observed_detrended, predicted, residual(=surge), slr_component(=line)
"""

import os
import gc
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from pyproj import Transformer
from tqdm import tqdm

# --- Configuration ---
DATA_DIR = Path('data/water_levels')
STATION_METADATA_FILE = Path('data/stations/noaa_stations_conus.csv')
VALID_RANGES_FILE = Path('data/stations/valid_ranges.csv')
PROCESSED_DIR = Path('data/water_levels_processed_v2')
PROCESSED_STATIONS_DIR = Path('data/stations_processed_v2')
PLOT_DIR = PROCESSED_DIR / 'plots'

RIVERINE_IDS = ['8518962', '8539094', '8545240', '8594900', '8760922',
                '8546252', '8546252', '9440083', '9439201', '9440422']
TRANSFORMER = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

NTDE_START, NTDE_END = 1983, 2001
TREND_REF_DATE = datetime(2025, 1, 1)          # anchor for the interpretable msl_2025
FULL_START, FULL_END = '1980-01-01', '2025-12-31 23:00:00'

# Station-selection thresholds (the "exclude short records" step)
MIN_SPAN_YEARS = 15.0                          # first->last valid obs span (>=~10yr training before 2020 val/test)
REQUIRE_LAST_YEAR = 2024                        # must reach a recent year (for 2024-25 test)
MIN_ANNUAL_BINS = 5                            # need this many usable yearly means to fit a slope
MIN_HOURS_PER_YEAR = 2000                      # a yearly mean must rest on >= this many hours
# QC flags (do not auto-drop)
SLOPE_FLAG_MM_YR = 30.0                        # plausible relative rate ceiling (Gulf subsidence ~20)
TIDE_TREND_FLAG_MM_YR = 1.0                    # harmonic tide should be ~0 trend


def secular_slope_from_annual(ntr):
    """OLS slope (m/day) of the annual means of NTR. Robust to seasonal/storms.
    Returns (slope, annual_means_series) or (None, None) if too few yearly bins."""
    ann = ntr.resample('YS').mean()
    cnt = ntr.resample('YS').count()
    ann = ann[cnt >= MIN_HOURS_PER_YEAR].dropna()
    if len(ann) < MIN_ANNUAL_BINS:
        return None, None
    x = np.array([datetime(ts.year, 7, 1).toordinal() for ts in ann.index], dtype=float)
    y = ann.values.astype(float)
    slope = np.polyfit(x - x.mean(), y, 1)[0]   # center x for conditioning; slope unchanged
    return slope, ann


def linear_trend_mm_yr(series):
    """Verification: OLS trend of a series in mm/yr (for QA labels)."""
    s = series.dropna()
    if len(s) < 100:
        return None
    x = s.index.map(datetime.toordinal).values.astype(float)
    slope = np.polyfit(x - x.mean(), s.values.astype(float), 1)[0]
    return slope * 365.25 * 1000.0


def plot_qa_station_v2(df, sid, meta, info, global_map, map_extents):
    """6-panel QA: observed+secular trend, detrended obs, tide, surge, + map."""
    colors = ['#8db4ad', '#a7bed3', '#ffcaaf', '#dabfff']
    fig = plt.figure(figsize=(16, 28))
    gs = fig.add_gridspec(6, 1, height_ratios=[1, 1, 1, 1, 0.2, 5.0])
    time_lims = [datetime(1980, 1, 1), datetime(2025, 12, 31)]
    x_days_full = df.index.map(datetime.toordinal).values.astype(float)

    titles = ['Observed + Secular Trend (own OLS on annual means)',
              'Detrended Observed (verification trend ~0)',
              'Tide Prediction (verification trend ~0)',
              'Residual = Surge  [seasonal kept] (verification trend ~0)']
    cols = ['observed', 'observed_detrended', 'predicted', 'residual']

    for i in range(4):
        ax = fig.add_subplot(gs[i])
        col = cols[i]
        if df[col].notna().any():
            ax.plot(df.index, df[col], color=colors[i], alpha=0.8, linewidth=0.7)
            if i == 0:
                slope = info['trend_slope_m_per_day']
                intercept = info['trend_intercept_m']
                flag = '  [FLAGGED]' if info['slope_flag'] else ''
                label = f"Secular trend: {info['trend_slope_mm_yr']:.2f} mm/yr{flag}"
                trend_line = slope * x_days_full + intercept
                ax.plot(df.index, trend_line, color='#d62828', linestyle='--',
                        linewidth=1.5, label=label)
                ann = info.get('annual')
                if ann is not None:
                    ax.scatter(ann.index, ann.values,
                               color='black', s=28, zorder=5, label='annual means (fit pts)')
                # mark the 2025 reconstruction level
                ax.plot(TREND_REF_DATE, info['msl_2025_m'], 'o', color='black',
                        markersize=10, zorder=6,
                        label=f"msl_2025 = {info['msl_2025_m']:.3f} m (datum)")
            else:
                tr = linear_trend_mm_yr(df[col])
                label = f"Verification trend: {tr:.4f} mm/yr" if tr is not None else "No trend"
                if tr is not None:
                    xm = x_days_full.mean()
                    ymean = df[col].mean()
                    ax.plot(df.index, (tr / (365.25 * 1000.0)) * (x_days_full - xm) + ymean,
                            color='#d62828', linestyle='--', linewidth=1.5, label=label)
        ax.set_title(titles[i], loc='left', fontsize=14, fontweight='bold')
        ax.set_xlim(time_lims)
        ax.set_ylabel('Level (m)')
        ax.legend(loc='upper left', frameon=True)

    ax_map = fig.add_subplot(gs[5])
    sx, sy = TRANSFORMER.transform(meta['lon'], meta['lat'])
    if global_map is not None:
        ax_map.imshow(global_map, extent=map_extents)
    margin = 350000
    ax_map.set_xlim(sx - margin, sx + margin)
    ax_map.set_ylim(sy - margin, sy + margin)
    ax_map.scatter([sx], [sy], color='#d62828', s=1000, alpha=0.3, zorder=5)
    ax_map.scatter([sx], [sy], color='#d62828', s=200, edgecolor='white', linewidth=2, zorder=6)
    ax_map.set_axis_off()
    ax_map.set_title(f"Station {sid}: {meta['name']}\n"
                     f"{info['n_years']:.1f} yr | datum {info['datum']} | type local_fit",
                     fontsize=18, fontweight='bold', pad=25)

    plt.subplots_adjust(hspace=0.7)
    plt.savefig(PLOT_DIR / f"{sid}_qa.png", dpi=100, bbox_inches='tight')
    fig.clf()
    plt.close(fig)
    plt.close('all')
    gc.collect()


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_STATIONS_DIR.mkdir(parents=True, exist_ok=True)

    # Optional global basemap (skip gracefully if offline)
    global_map, map_extents = None, None
    try:
        import contextily as ctx
        print("Downloading global context map...")
        west, south, east, north = -128, 24, -65, 50
        w, s = TRANSFORMER.transform(west, south)
        e, n = TRANSFORMER.transform(east, north)
        global_map, map_extents = ctx.bounds2img(w, s, e, n,
                                                  source=ctx.providers.CartoDB.Positron, zoom=5)
    except Exception as exc:
        print(f"  basemap unavailable ({exc}); plotting without it.")

    stations_df = pd.read_csv(STATION_METADATA_FILE, dtype={'station_id': str})
    stations_df = stations_df[stations_df['status'] == 'Active'].reset_index(drop=True)
    valid_ranges = pd.read_csv(VALID_RANGES_FILE, dtype={'station_number': str})
    stations_df = stations_df.merge(valid_ranges, left_on='station_id',
                                    right_on='station_number', how='left')

    full_index = pd.date_range(start=FULL_START, end=FULL_END, freq='h', tz='UTC')
    ord_ref = TREND_REF_DATE.toordinal()

    final_info, exclusion_log = {}, []
    store = pd.HDFStore(PROCESSED_DIR / 'processed_water_levels.h5', mode='w')

    print("Processing stations (own-record secular detrend, no borrowing)...")
    for _, row in tqdm(stations_df.iterrows(), total=len(stations_df)):
        sid = row['station_id']
        h5_path = DATA_DIR / f"{sid}.h5"

        if not h5_path.exists():
            exclusion_log.append({'sid': sid, 'reason': 'File missing'}); continue
        if sid in RIVERINE_IDS:
            exclusion_log.append({'sid': sid, 'reason': 'Riverine station'}); continue
        reg = row.get('region')
        if pd.isna(reg) or int(reg) == -1:
            exclusion_log.append({'sid': sid, 'reason': 'No region (outside region boxes)'}); continue

        with pd.HDFStore(h5_path, mode='r') as src:
            df = src['/water_levels'].astype('float32')
            try:
                datums = src.get_storer('water_levels').attrs.metadata.get('datums_found', [])
            except Exception:
                datums = []
        datum_str = '+'.join(datums) if datums else '?'
        if 'observed' not in df.columns or 'predicted' not in df.columns:
            exclusion_log.append({'sid': sid, 'reason': 'Missing columns'}); continue

        df.index = df.index.tz_localize('UTC') if df.index.tz is None else df.index.tz_convert('UTC')
        if row['start_year'] > 0:
            df = df[df.index.year >= row['start_year']]
        if row['end_year'] > 0:
            df = df[df.index.year <= row['end_year']]
        # only trust observed where the tide also exists (NTR must be defined)
        df.loc[df['observed'].notna() & df['predicted'].isna(), 'observed'] = np.nan

        obs = df['observed'].reindex(full_index)
        pred = df['predicted'].reindex(full_index)
        valid = obs.dropna()
        if len(valid) == 0:
            exclusion_log.append({'sid': sid, 'reason': 'No valid observed'}); continue

        first_v, last_v = valid.index.min(), valid.index.max()
        n_years = (last_v - first_v).days / 365.25
        if n_years < MIN_SPAN_YEARS:
            exclusion_log.append({'sid': sid, 'reason': f'Span {n_years:.1f}<{MIN_SPAN_YEARS}yr'}); continue
        if last_v.year < REQUIRE_LAST_YEAR:
            exclusion_log.append({'sid': sid, 'reason': f'Ends {last_v.year}<{REQUIRE_LAST_YEAR}'}); continue

        ntr = obs - pred
        slope, ann = secular_slope_from_annual(ntr)
        if slope is None:
            exclusion_log.append({'sid': sid, 'reason': 'Too few yearly bins for slope'}); continue

        # zero-mean recenter on hourly NTR -> surge is zero-mean & zero-trend
        nv = ntr.dropna()
        ord_h = nv.index.map(datetime.toordinal).values.astype(float)
        intercept = float(nv.values.astype(float).mean() - slope * ord_h.mean())

        ord_full = full_index.map(datetime.toordinal).values.astype(float)
        line = slope * ord_full + intercept                       # secular trend, float64
        detrended = obs.values - line                             # observed_detrended
        surge = ntr.values - line                                 # residual / surge target

        msl_2025 = float(slope * ord_ref + intercept)
        slope_mm_yr = slope * 365.25 * 1000.0
        ntde_mask = (full_index.year >= NTDE_START) & (full_index.year <= NTDE_END)
        ntde_cov = obs[ntde_mask].notna().sum() / (24 * 365 * (NTDE_END - NTDE_START + 1))
        ntde_msl = float(obs[ntde_mask].mean()) if ntde_cov >= 0.8 else np.nan
        tide_trend = linear_trend_mm_yr(pred)
        detr_trend = linear_trend_mm_yr(pd.Series(detrended, index=full_index))
        resid_trend = linear_trend_mm_yr(pd.Series(surge, index=full_index))

        info = {
            'station_id': sid, 'name': row['name'], 'lat': row['lat'], 'lon': row['lon'],
            'type': 'local_fit', 'n_years': round(n_years, 2),
            'record_start': str(first_v.date()), 'record_end': str(last_v.date()),
            'datum': datum_str,  # which datum the raw observed actually came in (MSL, STND fallback, or both)
            'trend_slope_m_per_day': float(slope),
            'trend_intercept_m': intercept,
            'trend_slope_mm_yr': round(slope_mm_yr, 3),
            'detr_slope_mm_yr': round(detr_trend, 4) if detr_trend is not None else np.nan,
            'resid_slope_mm_yr': round(resid_trend, 4) if resid_trend is not None else np.nan,
            'msl_2025_m': round(msl_2025, 4),
            'ntde_msl_m': round(ntde_msl, 4) if ntde_msl == ntde_msl else np.nan,
            'tide_trend_mm_yr': round(tide_trend, 4) if tide_trend is not None else np.nan,
            'slope_flag': bool(abs(slope_mm_yr) > SLOPE_FLAG_MM_YR),
            'tide_flag': bool(tide_trend is not None and abs(tide_trend) > TIDE_TREND_FLAG_MM_YR),
            'annual': ann,
        }

        df_final = pd.DataFrame({
            'observed': obs.values,
            'observed_detrended': detrended,
            'predicted': pred.values,
            'residual': surge,
            'slr_component': line,
        }, index=full_index).astype('float32')
        df_final.index.name = 'timestamp'

        store.put(f'station_{sid}', df_final, format='table')
        attrs = {k: v for k, v in info.items() if k != 'annual'}
        store.get_storer(f'station_{sid}').attrs.metadata = {**row.to_dict(), **attrs}

        plot_qa_station_v2(df_final[::6].copy(), sid, row, info, global_map, map_extents)
        final_info[sid] = info
        del df, obs, pred, ntr, df_final
        gc.collect()

    store.close()

    # ---- Summary table ----
    print("\n" + "=" * 92)
    print(f"{'STATION':<10} | {'YEARS':<6} | {'SLOPE mm/yr':<12} | {'msl_2025':<9} | "
          f"{'tide_tr':<8} | FLAGS")
    print("-" * 92)
    for sid, info in final_info.items():
        flags = ('SLOPE ' if info['slope_flag'] else '') + ('TIDE' if info['tide_flag'] else '')
        print(f"{sid:<10} | {info['n_years']:<6.1f} | {info['trend_slope_mm_yr']:<12.2f} | "
              f"{info['msl_2025_m']:<9.3f} | {str(info['tide_trend_mm_yr']):<8} | {flags}")

    print(f"\nKept {len(final_info)} stations | excluded {len(exclusion_log)}")

    # QC: confirm slopes are normal across all kept stations
    detr = [i['detr_slope_mm_yr'] for i in final_info.values()]
    resd = [i['resid_slope_mm_yr'] for i in final_info.values()]
    negs = [(s, i['trend_slope_mm_yr'], i['n_years']) for s, i in final_info.items()
            if i['trend_slope_mm_yr'] < 0]
    print(f"QC slopes (mm/yr): max|detrended|={max(abs(x) for x in detr):.3f}  "
          f"max|residual|={max(abs(x) for x in resd):.3f}  (both should be ~0)")
    print(f"QC negative SLR slopes ({len(negs)}): "
          + ", ".join(f"{s}({sl:.1f},{ny:.0f}yr)" for s, sl, ny in sorted(negs, key=lambda t: t[1])))
    excl = pd.DataFrame(exclusion_log)
    if not excl.empty:
        print("Exclusion reasons:")
        print(excl['reason'].value_counts().to_string())
        excl.to_csv(PROCESSED_STATIONS_DIR / 'exclusions.csv', index=False)

    # ---- Station metadata CSV + shapefile ----
    keep_cols = ['station_id', 'name', 'state', 'lat', 'lon', 'is_gl', 'region',
                 'd_msl_m', 'd_navd_m', 'd_mhhw_m', 'f_min_ms_m', 'f_mod_ms_m', 'f_maj_ms_m']
    keep_cols = [c for c in keep_cols if c in stations_df.columns]
    meta_df = stations_df[stations_df['station_id'].isin(final_info)][keep_cols].copy()
    for fld in ['n_years', 'trend_slope_mm_yr', 'detr_slope_mm_yr', 'tide_trend_mm_yr',
                'resid_slope_mm_yr', 'msl_2025_m', 'ntde_msl_m',
                'trend_slope_m_per_day', 'trend_intercept_m',
                'slope_flag', 'tide_flag', 'record_start', 'record_end']:
        meta_df[fld] = meta_df['station_id'].map(lambda x: final_info[x][fld])
    meta_df.to_csv(PROCESSED_STATIONS_DIR / 'stations_processed.csv', index=False)

    gdf = gpd.GeoDataFrame(meta_df,
                           geometry=gpd.points_from_xy(meta_df['lon'], meta_df['lat']),
                           crs="EPSG:4326")
    gdf.to_file(PROCESSED_STATIONS_DIR / 'stations_processed.shp')
    print(f"\nWrote {PROCESSED_DIR/'processed_water_levels.h5'}, "
          f"{PROCESSED_STATIONS_DIR/'stations_processed.csv'}, and {len(final_info)} QA plots.")
