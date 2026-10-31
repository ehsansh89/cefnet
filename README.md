# CEFNet: Coastal Ensemble Forecast Network

CEFNet is a suite of deep learning models that turn public weather forecasts into 72-hour water-level forecasts at 122 NOAA tide gauges along the coasts of the contiguous United States. One model is trained for each of ten regional domains, with three random seeds per domain. Driving the 30 models with the 31-member GEFS ensemble gives 93 forecasts per gauge and cycle.

This repository holds the code used for the paper:

> Shahabi, A. and Tahvildari, N. *Learned ensembles improve coastal flood guidance within the historical experience envelope.* Under review.

Live forecasts: https://cefnet.com

## What is in this repository

- **Code:** every script used to download the data, train the models, run the forecasts, and compute the numbers and figures in the paper, with the exceptions listed under [What is not included](#what-is-not-included). This is research code, kept as it was run for the paper apart from the small release edits listed under [Release edits](#release-edits).
- **Small reference files the code needs:** gauge lists (`data/stations*`), input and target scalers (`samples/scalers_v2/`), regional grid and boundary files (`GIS/regions/`), and the forecast-cycle list for Fig. 1 (`outputs/paper/inference/nn_24lb_72fh_v2/needed_cycles.json`). `data/stations_processed/` is the full 163-gauge inventory used to build the baseline crosswalks, and `data/stations_processed_v2/` is the 122-gauge set the models use.
- **Trained weights:** attached to the GitHub Release as `cefnet_weights_v2.zip`, not stored in the repository. Unzip it in the repository root so that the files land in `models/nn_24lb_72fh_v2/region_XX/seed_Y/best_phase1.pt`.

The input data are not included. All inputs are public, and the `d_*` scripts download them (see [Data sources](#data-sources)). The full set is several hundred gigabytes.

## What is not included

- The script that computes the attention maps behind Supplementary Fig. 2. `paper_si_assets.py` plots its cached output. The script needs a config update before release and will be added.
- The scripts behind the live-season results in Supplementary Figs. 6 and 7. They are part of the forecast dashboard code and will be released with it.
- The downloaded inputs, the training samples, and the forecast archives.

## Setup

```
conda env create -f environment.yml
conda activate cefnet
```

Install the PyTorch build that matches your machine (CPU or CUDA) if the pip line in `environment.yml` does not suit it. `ffmpeg` is only needed for the optional animation written by `inference_v2.py`.

On Windows, set `KMP_DUPLICATE_LIB_OK=TRUE` before running scripts that import PyTorch together with the geospatial libraries (the pip PyTorch wheel and conda ship separate OpenMP runtimes), and set `PYTHONUTF8=1` if you redirect script output to a file.

## How the scripts are run

- Run each script from the repository root, for example `python train.py`. Scripts move to their own folder first, so all paths are relative to the root.
- Scripts take no command-line arguments, except `d_ofs_download.py` (see step 4). Each one reads a JSON config with a matching name, or the settings at the top of the file.
- Some scripts read environment variables:
  - `REGION_ID` (1 to 10) and `SEED` (1 to 3) select the region and seed for `train.py` and `inference_v2.py`. `START_TIME` and `END_TIME` override the hindcast span in `inference_v2.py`.
  - `CONUS_REGION` (for example `region_06`) selects the region for `d_atmospheric_download.py` and `d_atmospheric_process.py`. `REGIONS` (comma-separated) selects the regions for `s_generate_samples_v2.py`.
  - `X_*_CONFIG` variables point a script to a different config file (`GEFS_DOWNLOAD_CONFIG_PATH` for the GEFS downloader). The defaults are the configs used for the paper.
- The scripts follow a naming pattern: `d_` downloads or processes data, `s_` builds training samples and scalers, `x_` holds helper steps, and `o_` and `paper_` make scores and figures.

## Pipeline, in run order

| Step | Scripts | What it does |
|---|---|---|
| 1. Gauges and regions | `d_create_region_boxes.py`, `d_gauge_stations.py`, `d_water_level_download.py`, `d_water_level_process_v2.py` | Defines the ten regional grids and boundaries, gets gauge metadata and flood thresholds from NOAA CO-OPS and assigns each gauge to a region, downloads observed water levels and tide predictions, and removes the tide and the secular trend to make the training target (the residual). Selects the 122 gauges. |
| 2. Training atmosphere | `d_atmospheric_download.py`, `d_atmospheric_process.py` | Downloads ERA5 10 m wind and surface pressure on each regional grid. |
| 3. Forecast atmosphere | `x_gfs_inventory.py` → `d_atmospheric_forecast_download.py`, `x_gefs_inventory.py` → `d_atmospheric_forecast_download_gefs.py` | Downloads archived GFS and GEFS (31 members) forecasts for 2024 and 2025. |
| 4. Physics baselines | `x_stofs_inventory.py` → `x_stofs_crosswalk.py` → `d_stofs_download.py`, `x_ofs_inventory_ncei.py` → `x_ofs_crosswalk.py` → `d_ofs_download.py` → `x_ofs_datum_bridge.py` | Downloads STOFS-2D-Global and regional OFS forecasts at the gauges, and estimates the datum offset for each OFS model at each gauge. `d_ofs_download.py` takes flags: CBOFS, TBOFS and NGOFS2 were downloaded from the NCEI archive (`python d_ofs_download.py --model cbofs`, the default source). DBOFS, GoMOFS, SFBOFS and WCOFS were downloaded from the NODD mirror (`--source s3`), and WCOFS with `--exclude-covered-by sfbofs` so that San Francisco Bay gauges are scored against SFBOFS only. |
| 5. Training samples | `s_scalers_v2.py` → `s_build_catalog_v2.py` → `s_generate_samples_v2.py` | Fits the scalers, builds the sampling catalog, and writes the scaled training samples. `s_sample_batch_v2.py` assembles each sample. |
| 6. Training | `train.py` (+ `nn.py`, `train_config_v2.json`, `nn_config.json`) | Trains one region and seed. Run it 30 times (10 regions × 3 seeds). Training years are up to 2021, validation 2022 to 2023, and test 2024 to 2025. |
| 7. Forecasts | `paper_fig1_inference_v2.py`, `inference_ensemble_v2.py`, `inference_v2.py` | Runs the deterministic GFS-driven forecasts, the 93-member GEFS-driven ensemble, and the ERA5-driven hindcast (`inference_v2.py` once per region with `REGION_ID` 1 to 10 and `SEED=1`). |
| 8. Ensemble post-processing | `x_ensemble_verify.py` → `x_ensemble_calibrate_v2.py` → `x_dressed_baseline.py` → `o_ensemble_decision.py`, `o_ensemble_uq.py` | Aligns forecasts, observations, flood thresholds and baselines on one grid. Fits the spread calibration on one year and tests it on the other year. Builds the dressed STOFS baseline and the flood-warning events. |
| 9. Paper numbers and figures | `paper_*.py` (see below) | Computes the scores, tables and figures in the paper. |

`nn.py` defines the model. `s_sample_batch.py`, `s_scalers.py`, `s_gfs_archive.py`, `s_gefs_archive.py`, `inference_ensemble.py`, `x_stofs_pair.py`, `x_ofs_pair.py`, `x_stofs_datum_bridge.py`, `o_style.py`, `paper_common.py` and the other `paper_*` files not listed below are helper modules that other scripts import. The run blocks of `x_stofs_pair.py`, `x_ofs_pair.py` and `x_stofs_datum_bridge.py` belong to an earlier version of the comparison and are not used for the paper; only their functions are.

## Where each paper figure comes from

Some script names keep the numbering of earlier drafts.

| Paper item | Processing | Plotting |
|---|---|---|
| Fig. 1 (accuracy against STOFS-2D and OFS) | `paper_fig1_process.py`, `paper_fig1_lead.py` | `paper_fig1_plot.py` |
| Fig. 2 (accuracy by flood severity) | `paper_fig5_extremes_process.py` | `paper_fig5_extremes_plot.py` |
| Fig. 3 (training support of high water levels) | `paper_fig5_extrap_draft.py` | same script |
| Fig. 4 (ERA5-driven storm cases) | `paper_fig6_ladder_draft.py` (runs its own inference) | same script |
| Fig. 5 (reliability of calibrated flood probabilities) | `paper_fig2_process.py` | `paper_fig3_rework_draft.py` |
| Fig. 6 (flood warnings) | `paper_fig2_warn_process.py`, `paper_fig4_table_process.py` | `paper_fig4_assembly_draft.py` |
| Fig. 7 (architecture schematic) | none, drawn by hand | none |
| Methods, ERA5 attribution numbers | `paper_fig5_attribution.py` | none |
| Major-episode counts | `paper_fig3_process.py` | none |
| Supplementary figures and tables | the scripts above, plus `paper_fig1_sensitivity.py` | `paper_si_assets.py` |

## Data sources

| Data | Source | Script |
|---|---|---|
| Water levels, tide predictions, datums, flood thresholds | NOAA CO-OPS API | `d_gauge_stations.py`, `d_water_level_download.py` |
| ERA5 reanalysis | Copernicus Climate Data Store. Needs your own free CDS API key in `~/.cdsapirc` or the `CDSAPI_KEY` environment variable. | `d_atmospheric_download.py` |
| GFS and GEFS forecasts | NOAA Open Data Dissemination (NODD) | `d_atmospheric_forecast_download*.py` |
| STOFS-2D-Global | NODD (`noaa-gestofs-pds`) | `d_stofs_download.py` |
| Regional OFS | NOAA NCEI archive, and NODD for DBOFS, GoMOFS, SFBOFS and WCOFS | `d_ofs_download.py` |
| Hurricane best tracks | IBTrACS version 4r01, North Atlantic file, saved as `data/storm_tracks/ibtracs_NA_full.csv` | used by `paper_fig6_ladder_draft.py` |

## Compute

- Training used one GPU per region and seed on an HPC cluster.
- The two-year ensemble archive (10 regions, 31 GEFS members × 3 seeds, four cycles a day) took about 86 hours on a desktop CPU.
- One forecast cycle with all 93 members takes about 10 seconds per region on a CPU.

## Release edits

The code is as it was run for the paper, except for these edits made for release. None changes a computation.

- `d_atmospheric_download.py`: the authors' CDS keys were removed. The script now reads `CDSAPI_KEY` or `~/.cdsapirc`.
- The default config files of `train.py`, `paper_fig1_process.py`, `x_ensemble_verify.py`, `o_ensemble_uq.py` and `x_ofs_datum_bridge.py` were switched from the earlier v1 files to the v2 files used for the paper. The `X_*_CONFIG` variables still override them.
- `d_create_region_boxes.py` and `d_gauge_stations.py` read and write `GIS/regions/` instead of an older `regions/` folder.
- The GFS and GEFS download configs were reset from the last partial run to the full paper period.
- Non-ASCII characters were removed from console messages, two stale output folders were renamed, and comments were tidied.

## License

A license will be added when the paper is published.

## Contact

Ali Shahabi (ashah127@fiu.edu) and Navid Tahvildari (ntahvild@fiu.edu), Department of Civil and Environmental Engineering, Florida International University.
