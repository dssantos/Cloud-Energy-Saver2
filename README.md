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

### 3. Configure .env File
```env
WINDOWS_HOST=<windows_host_ip>
WINDOWS_USER=<windows_username>
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

```

---

## Usage

### Find and registry compute hosts
```bash
python ces.py --registrator

```

### Show current status of Compute nodes
```bash
python ces.py --status

```

### Initialize VMs to create load on cloud environment
```bash
# auto on and off 30 VMs 
python ces.py --instantiator 30

# Only on 5 VMs
python ces.py --on 5

# Only off 5 VMS
python ces.py --off 5
```

### Start checking loads and manage hosts state
```bash
# Set threshold of loads to manage hosts
python ces.py --verifier 70 30

# OR Manage hosts with arima predict model
python ces.py --verifier 70 30 arima

# OR Manage hosts with lstm predict model
python ces.py --verifier 70 30 lstm
```

**Prediction Models:**

- **default**: Uses current RAM readings from `ram_usage.get()`
- **naive**: Uses last known workload value (if recent) or current RAM
- **arima**: ARIMA(1,0,1) model fitted to workload history
- **lstm**: LSTM neural network with continuous background training