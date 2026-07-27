# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Cloud Energy Saver 2 is a power management system for OpenStack cloud computing environments. It monitors compute host resource usage (RAM) and automatically powers hosts on/off based on workload to save energy.

## Architecture

The system uses a modular architecture with these main components:

- **orchestrator.py**: Primary entry point for experiments. Manages the full pipeline end-to-end: wakes hosts, waits for readiness, registers hosts, starts verification (with workload collection + LSTM training), runs the VM instantiator loop, monitors progress, and saves final status. Supports `--wake-only`, `--verify-only`, and `--instantiator-only` modes.
- **ces.py**: Lightweight manual-utilities CLI for debugging and one-off operations (register, status, turn VMs on/off). Verification and instantiation logic lives in the orchestrator; ces.py now only prints a redirect message for those flags.
- **registrator.py**: Discovers and registers compute hosts from the OpenStack environment
- **status.py**: Retrieves current status of compute hosts (state, VMs running, RAM usage)
- **verifier.py**: Main control loop that monitors hosts and triggers power management actions
- **changestate.py**: Handles Wake-on-LAN, VBoxManage (local + SSH), and SSH-based shutdown. Wake fires all methods in parallel and confirms success via ping (2 min timeout).
- **instances.py**: Manages VM instantiation and deletion for load testing
- **predict.py**: Provides predictive models (naive, ARIMA, LSTM) for forecasting RAM usage
- **workload.py**: Saves and retrieves historical RAM usage data for prediction models
- **event_logger.py**: Logs power management events to `events_<model>_<timestamp>.json`
- **lstm_multivariate.py**: Standalone script for LSTM hyperparameter random search experiments

### OpenStack Integration

The system communicates with OpenStack via HTTP API calls to the controller node (http://controller:8774 for Nova, http://controller:5000 for Keystone). Authentication uses X-Auth-Token headers with tokens cached in `token.txt`.

### Host Registration

Compute hosts must be registered before management operations. The registration process stores hostnames in `registered.txt` as a Python list literal string. Only registered hosts are monitored and managed.

### Prediction Models

The verifier supports multiple prediction models for forecasting RAM usage:

- **default**: Uses current RAM readings from `ram_usage.get()`
- **naive**: Uses last known workload value (if recent) or current RAM
- **arima**: ARIMA(1,0,1) model fitted to workload history
- **lstm**: LSTM neural network with continuous background training

LSTM models are saved in `models/{hostname}/` with checkpoints. The system trains models in background threads and selects the best performing model based on loss values. Training runs whenever total data exceeds 100 samples (per-host). Predictions that deviate more than 15% (and >15 percentage points) from the last reading are flagged as incoherent and the model is penalized.

## Installation and Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Manual utilities (ces.py) — debugging / one-off operations

```bash
# Register compute hosts
python ces.py --registrator

# Show host status (continuous 10s refresh)
python ces.py --status

# Turn VMs on/off manually
python ces.py --on 5
python ces.py --off 5
```

### Experiments (orchestrator.py) — primary entry point

```bash
# Full experiment with LSTM
python orchestrator.py --model lstm --lim-max 70 --lim-med 50 --num-vms 27

# Partial stages
python orchestrator.py --wake-only
python orchestrator.py --verify-only --model lstm --lim-max 70 --lim-med 50
python orchestrator.py --instantiator-only --num-vms 27
```

Note: `ces.py --verifier` / `--instantiator` print a redirect message to the orchestrator rather than running the legacy code path.

## Power Management Logic

The verifier classifies hosts as:
- **running**: hosts with VMs running
- **idle**: hosts running but no VMs
- **offline**: hosts registered but powered off

Actions taken based on average RAM usage:
- **Above MAX**: Wake offline hosts if available, or keep minimal idle hosts
- **Between MED and MAX**: Keep 1 idle host, power off others
- **Below MED**: Power off all idle hosts (keep 1 if no running hosts exist)

## File Structure Notes

- `registered.txt`: Stores list of registered hostnames as Python list literal
- `token.txt`: Caches OpenStack authentication token
- `{hostname}.csv`: Historical RAM usage data collected by workload.py
- `models/{hostname}/`: Trained LSTM models and checkpoints
- `results.csv`: LSTM hyperparameter search results from lstm_multivariate.py

## Development Notes

- The system uses SSH commands for both VM management (via `openstack` CLI) and host power operations
- MAC addresses are stored per-host for Wake-on-LAN functionality; VirtualBox VMs are detected by MAC prefix `08:00:27`
- The orchestrator is the primary runner: it starts workload collection, LSTM training (when `--model lstm`), verification, and instantiation as coordinated background threads
- Emergency wakes set a 5-minute cooldown (`EMERGENCY_COOLDOWN_SECONDS`) during which the host cannot be shut down, preventing rapid on/off churn
- The status command runs in a continuous loop with 10s refresh interval
- LSTM models train in background threads; prediction uses best available model or falls back to current RAM usage
