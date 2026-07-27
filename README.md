# Cloud Energy Saver 2

Power management system for OpenStack cloud computing environments. Monitors compute host resource usage (RAM) and automatically powers hosts on/off based on workload to save energy using predictive models (naive, ARIMA, LSTM).

---

## Windows VirtualBox VM Setup (REQUIRED)

**Complete this setup before running any commands below.**

The Cloud Energy Saver (CES) controls VirtualBox VMs (controller, compute1, compute2, compute3) running on Windows host via SSH.

![Topology](https://raw.githubusercontent.com/dssantos/Cloud-Energy-Saver/refs/heads/master/topologia.png)

### Download and Import Pre-configured VMs

Download pre-configured VirtualBox images and documentation from:
[OpenStack VMs](https://mega.nz/#F!TbBmSA4b!YHuaruKoxMUFtyM6OXNsWQ)

The link opens a cloud folder with the following files:
- **Controller.vdi** - VM disk with Ubuntu 16.04 server and OpenStack configured
- **ComputeVM.vdi** - VM disk for compute nodes
- **ComputePen.raw** - Hard drive image to clone to USB drive
- **README.txt** - Detailed OpenStack configuration instructions

After accessing:
1. Download the .vdi files from the cloud folder
2. In VirtualBox on Windows: Machine → New → Create new VM → Use existing disk (.vdi files)
3. Clone the compute VM to create compute2 and compute3 (right-click → Clone)
4. Ensure VMs are named: controller, compute1, compute2, compute3

---

## Ubuntu/Cloud Energy Saver Setup (REQUIRED)

**Complete this setup on the Linux machine running Cloud Energy Saver.**

### 1. Configure Network
```bash
sudo ip addr add 10.0.0.100/24 dev <network_interface>
```

### 2. Setup SSH to Windows
```bash
# Generate key
ssh-keygen -t ed25519

# Copy to Windows
cat ~/.ssh/id_ed25519.pub | ssh <windows_user>@<windows_host> "mkdir -p ~/.ssh && chmod 700 ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys"

# Test
ssh <windows_user>@<windows_host> "VBoxManage list vms"
```

### Test VM Start
```bash
ssh <windows_user>@<windows_host> "VBoxManage startvm <vm_name> --type=headless"
```

---

## Prerequisites

Before installing Cloud Energy Saver 2, ensure you have:

- **Python 3.x** installed on Ubuntu machine
- **VirtualBox** with Extension Pack installed on Windows host
- **SSH access** configured between Ubuntu and Windows hosts
- **OpenStack** configured and running on VMs
- **OpenStack VMs**: controller, compute1, compute2, compute3

---

## Installation
```bash
git clone https://github.com/dssantos/dssantos-Cloud-Energy-Saver2.git ces2
cd ces2
python3 -m venv .ces2
source .ces2/bin/activate
pip install -U pip
pip install -r requirements.txt

# Configure .env file (copy from example and update values)
cp .env.example .env
# Edit .env with your Windows host IP and username:
# WINDOWS_HOST=<windows_host_ip>
# WINDOWS_USER=<windows_username>
```

---

## Usage

The project has two entry points with distinct roles:

- **`ces.py`** — lightweight manual utilities for inspecting/operating the environment (status, register, turn VMs on/off). Use for debugging and one-off operations.
- **`orchestrator.py`** — full experiment runner (wake hosts + verification + VM load generation, predictive models, event logging, SLA tracking).

### Manual utilities (`ces.py`)

#### Find and register compute hosts
```bash
python ces.py --registrator
```

#### Show current status of Compute nodes (refreshes every 10s)
```bash
python ces.py --status
```

#### Turn VMs on/off (manual load control)
```bash
# Start 5 VMs
python ces.py --on 5

# Shut down 5 VMs
python ces.py --off 5
```

---

## Experiment Orchestrator

The orchestrator (`orchestrator.py`) is the main entry point for running experiments. It manages a complete experiment end-to-end: wakes all hosts, waits for them to be ready, starts verification, and runs the VM instantiator in a loop. It also handles workload collection, background LSTM training, event logging, and SLA violation tracking.

### Full experiment (wake + register + verify + instantiator)
```bash
# Run with LSTM model, 70%/50% thresholds, 27 VMs
python orchestrator.py --model lstm --lim-max 70 --lim-med 50 --num-vms 27

# With a custom duration (default: 18 hours)
python orchestrator.py --model lstm --lim-max 70 --lim-med 50 --num-vms 27 --duration 24
```

### Modes (run only part of the pipeline)
```bash
# Only wake and register hosts, then stop
python orchestrator.py --wake-only

# Only run continuous verification (manage hosts on/off)
python orchestrator.py --verify-only --model lstm --lim-max 70 --lim-med 50

# Only create/delete VMs in a loop (load generation)
python orchestrator.py --instantiator-only --num-vms 27
```

### Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model` | `default` | Prediction model: `default`, `naive`, `arima`, `lstm` |
| `--lim-max` | `70` | Maximum RAM threshold (%) — triggers waking hosts |
| `--lim-med` | `30` | Medium RAM threshold (%) — triggers shutting down hosts |
| `--num-vms` | `27` | Number of VMs to create/delete per cycle |
| `--duration` | `18` | Experiment duration in hours |
| `--config` | — | Optional JSON config file |
| `--wake-only` | off | Wake and register hosts, then exit |
| `--verify-only` | off | Run only continuous verification |
| `--instantiator-only` | off | Run only VM create/delete loop |

**Note:** With the `lstm` model, the orchestrator starts background training threads for all registered hosts and logs events to `events_lstm_<timestamp>.json`.

**Prediction Models:**

- **default**: Uses current RAM readings from `ram_usage.get()`
- **naive**: Uses last known workload value (if recent) or current RAM
- **arima**: ARIMA(1,0,1) model fitted to workload history
- **lstm**: LSTM neural network with continuous background training

