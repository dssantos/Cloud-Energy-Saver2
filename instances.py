#coding: utf-8
import json

import requests, header, subprocess, time, sys


def get():
	vms = []
	pos = length -1

	while pos > -1:

		vm =  vm_list[pos]['name']
		pos -= 1
		vms.append(vm)

	print(vms)
	return vms

def on(qt_on):

	pos = len(get()) + 1
	while qt_on > 0:
		vm = 'vm-%s' %pos
		print('ligando %s' %vm)
		command = "ssh user@controller '. admin-openrc && openstack server create --image cirros --flavor=1CPU_128RAM %s'" %vm
		run = subprocess.check_output(command, shell=True)  # Receives the output of the above command
		qt_on -= 1
		pos += 1

def off(qt_off):

	vms = get()
	pos = length	
	if qt_off <= length:

		while pos > length-qt_off:
			vm = vms[pos-1]
			print('desligando %s' %vm)
			command = "ssh user@controller '. admin-openrc && openstack server delete %s'" %vm
			run = subprocess.check_output(command, shell=True)  # Receives the output of the above command
			pos -= 1


	else:
		print('Só existem %s VMs para desligar' %length)

def auto_on(limit):

	while True:

		pos = len(get()) + 1
		vms = []
		for x in range(limit):
			for i in range(60,-1,-1):
				sys.stdout.write("Liga em: %3d\r"%i,)
				sys.stdout.flush()
				time.sleep(1)
			vm = 'vm-%s'%pos
			pos += 1
			print('ligando %s' %vm)
			command = "ssh user@controller '. admin-openrc && openstack server create --image cirros --flavor=1CPU_128RAM %s'" %vm
			run = subprocess.check_output(command, shell=True)  # Receives the output of the above command
			vms.append(vm)

		for vm in reversed(vms):
			for i in range(60,-1,-1):
				sys.stdout.write("Desliga em: %3d\r"%i,)
				sys.stdout.flush()
				time.sleep(1)
			print('desligando %s   ' %vm)
			command = "ssh user@controller '. admin-openrc && openstack server delete %s'" %vm
			run = subprocess.check_output(command, shell=True)  # Receives the output of the above command

try:
	r = requests.get('http://controller:8774/v2.1/servers', headers=header.get())
except requests.exceptions.ConnectionError as e:
	raise requests.exceptions.ConnectionError(f"{e}: This computer does not have communication with the Controller.\nCheck the requirements in https://github.com/dssantos/Cloud-Energy-Saver2")

vm_list = json.loads(r.content) # Returns the content of the queried URL
vm_list = vm_list['servers']
length = len(vm_list)


# SLA Monitoring Constants and Functions

SLA_TIMEOUT = 120  # 2 minutes timeout for VM allocation


def get_server_status(server_id):
	"""Get detailed server status from OpenStack API."""
	url = f"http://controller:8774/v2.1/servers/{server_id}"
	try:
		r = requests.get(url, headers=header.get())
		return json.loads(r.content)['server']
	except Exception as e:
		print(f'[ERROR] Failed to get server status: {e}')
		return {}


def extract_server_id(create_output):
	"""Extract server ID from create command output."""
	# Parse openstack server create output
	import re
	match = re.search(r'\|\s([a-f0-9\-]+)\s+\|', create_output)
	return match.group(1) if match else None


def create_instance_with_sla_check(vm_name):
	"""
	Create VM instance and monitor allocation.
	Returns: dict with vm_name, server_id, allocated, allocation_time, sla_violation, timestamp
	"""
	from datetime import datetime
	start_time = time.time()
	allocated = False
	sla_violation = False
	server_id = None

	# Create instance
	print(f'[SLA] Creating VM {vm_name}...')
	command = f"ssh user@controller '. admin-openrc && openstack server create --image cirros --flavor=1CPU_128RAM {vm_name}'"
	try:
		create_result = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT).decode('utf-8')
		server_id = extract_server_id(create_result)

		if not server_id:
			# Try to get server ID from API by name
			print(f'[SLA] Could not extract server ID from output, querying API...')
			time.sleep(5)  # Wait for server to appear in API
			r = requests.get('http://controller:8774/v2.1/servers', headers=header.get())
			servers = json.loads(r.content)['servers']
			for server in servers:
				if server['name'] == vm_name:
					server_id = server['id']
					break

		if server_id:
			# Monitor allocation
			print(f'[SLA] Monitoring allocation for {vm_name} (server_id: {server_id})...')

			while time.time() - start_time < SLA_TIMEOUT:
				server_status = get_server_status(server_id)

				# Check if allocated to a host
				if server_status.get('OS-EXT-SRV-ATTR:host'):
					allocated = True
					allocation_time = time.time() - start_time
					print(f'[SLA] {vm_name} allocated to {server_status.get("OS-EXT-SRV-ATTR:host")} in {allocation_time:.1f}s')
					break

				time.sleep(5)  # Check every 5 seconds
			else:
				# Timeout - SLA violation
				sla_violation = True
				allocation_time = SLA_TIMEOUT
				print(f'[SLA VIOLATION] {vm_name} failed to allocate within {SLA_TIMEOUT}s')

		else:
			print(f'[SLA ERROR] Could not get server ID for {vm_name}')
			sla_violation = True
			allocation_time = SLA_TIMEOUT

	except subprocess.CalledProcessError as e:
		print(f'[SLA ERROR] Failed to create VM {vm_name}: {e}')
		sla_violation = True
		allocation_time = SLA_TIMEOUT
	except Exception as e:
		print(f'[SLA ERROR] Unexpected error creating VM {vm_name}: {e}')
		sla_violation = True
		allocation_time = SLA_TIMEOUT

	return {
		'vm_name': vm_name,
		'server_id': server_id,
		'allocated': allocated,
		'allocation_time': allocation_time,
		'sla_violation': sla_violation,
		'timestamp': datetime.now().isoformat()
	}
