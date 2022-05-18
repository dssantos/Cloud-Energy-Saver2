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
