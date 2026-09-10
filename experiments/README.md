# Experiments

This directory holds the data collected by the orchestrator runs (`orchestrator.py`). Each subdirectory is named after the run start date (format `YYYYMMDD`) and contains the cluster workload time series and the decision events.

## Run layout

```
experiments/<YYYYMMDD>/
├── data/
│   ├── cluster_workload_{baseline,default,lstm}_<timestamp>.csv   # cluster load every 10 s
│   ├── events_{baseline,default,lstm}_<timestamp>.json            # wake, shutdown, sla_violation
│   ├── experiment_results_<timestamp>.json                        # per-mode run results
│   ├── scoreboard.json                                            # LSTM model scoreboard (lstm mode)
│   ├── crashed_hosts.json                                         # crashed host registry
│   ├── analysis_<ts1>__vs__<ts2>__vs__<ts3>.json                  # comparative analysis
│   └── experiment_run_<timestamp>.log                             # full run log
```

## How to reproduce

### 1. Run the three modes in sequence (baseline, default, lstm)

```bash
# 14 hours per mode, 27 VMs per cycle, 70%/50% thresholds
DUR_HOURS=14 MODELS="baseline default lstm" LIM_MAX=70 LIM_MED=50 bash run_experiments.sh
```

Each mode writes its workload CSVs and event files to the working directory, named with the mode start timestamp.

### 2. Generate the comparative analysis

```bash
python3 analyze_metrics.py \
  --baseline-events events_baseline_<ts>.json --baseline-csv cluster_workload_baseline_<ts>.csv \
  --reactive-events events_default_<ts>.json  --reactive-csv  cluster_workload_default_<ts>.csv \
  --lstm-events     events_lstm_<ts>.json     --lstm-csv      cluster_workload_lstm_<ts>.csv
```

The script crosses each event file with its workload series and produces the `analysis_*.json`
file (active hours, energy, SLA episodes and shutdown delay).

## Interpreting the analysis output

The analysis compares the three modes over an **aligned window**, measured in hours and anchored
at the baseline start. The window length is limited by the shortest series, so all modes are
compared over exactly the same period (typically ~14 h).

Key naming: `baseline` = the unmanaged mode; `reactive` = the **Default** mode (CES2 reactive
verifier); `lstm` = the **LSTM** mode (CES2 with the predictive branch).

| JSON field | Meaning | Unit |
|---|---|---|
| `window.start` / `window.duration_hours` | Anchor timestamp and length of the aligned window | — / h |
| `baseline_hours`, `reactive_hours`, `lstm_active_hours` | Time hosts stayed powered on, summed over the 3 hosts | host·h |
| `energy_kwh_baseline`, `energy_kwh_default`, `energy_kwh_lstm` | Energy consumed in the window | kWh |
| `default_saved_pct`, `lstm_saved_pct` | Energy saved by each managed mode vs. the baseline | % |
| `sla_episodes_baseline`, `sla_episodes_reactive`, `sla_episodes_lstm` | Number of SLA violation episodes in the window | count |
| `late_shutdown_reactive_min`, `late_shutdown_lstm_min` | Mean delay between load dropping below `lim_med` and the actual host shutdown | min |

How to read the results:

- **Energy** is derived from the active hours via a constant per-host power (`P_AVG`, from
  `config.py`), so it is a relative comparison between modes, not an absolute measurement.
- **SLA episodes** are transitions in which a host with VMs exceeds the SLA threshold
  (`lim_max` plus the margin in `SLA_RAM_MARGIN_PCT`), or a host with VMs goes down unexpectedly.
  The baseline has no verifier, so its `sla_episodes` is always 0 — a **measurement artifact**,
  not a sign of better availability.
- The **energy × availability trade-off** is the central result: the reactive mode saves more
  energy but concentrates load and incurs more SLA episodes, whereas the LSTM mode keeps more
  hosts on (less energy saved) while reducing SLA episodes and their severity.
- All statistics are computed **only over the aligned window**, so values may differ slightly
  from those in a full single-mode run.

## Available runs

| Folder | Period | Modes | Order |
|--------|--------|-------|-------|
| `20260903/` | Sep 1–3, 2026 | 3 × 14 h | lstm → default → baseline |

The 20260903 data is the dataset used in the thesis (aligned 14 h window per mode).
