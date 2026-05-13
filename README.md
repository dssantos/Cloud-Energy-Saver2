# Cloud Energy Saver 2


## How to install
```bash
git clone https://github.com/dssantos/dssantos-Cloud-Energy-Saver2.git ces2
cd ces2
python3 -m venv .ces2
source .ces2/bin/activate
pip install -U pip
pip install -r requirements.txt

```

---

## Remote Windows VirtualBox Setup (REQUIRED)

**Complete this setup before running any commands below.**

The system controls VirtualBox VMs (controller, compute1, compute2, compute3) running on Windows host via SSH.

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

### 4. Download and Import Pre-configured VMs

Download pre-configured VirtualBox images (controller + compute) from:
[OpenStack VMs](https://mega.nz/#F!TbBmSA4b!YHuaruKoxMUFtyM6OXNsWQ)

After downloading:
1. Unzip the downloaded files
2. In VirtualBox on Windows: File → Import Appliance → Select the .ova files
3. Clone the compute VM to create compute2 and compute3 (right-click → Clone)
4. Ensure VMs are named: controller, compute1, compute2, compute3

### Test VM Start
```bash
ssh <windows_user>@<windows_host> "VBoxManage startvm <vm_name> --type=headless"
```

---

## How to use

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
# auto on and off 50 VMs 
python ces.py --instantiator 50

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