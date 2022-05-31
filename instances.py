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
				print("Liga em: %3d\r"%i,)
				time.sleep(1)
				sys.stdout.flush()
			vm = 'vm-%s'%pos
			pos += 1
			print('ligando %s' %vm)
			command = "ssh user@controller '. admin-openrc && openstack server create --image cirros --flavor=1CPU_128RAM %s'" %vm
			run = subprocess.check_output(command, shell=True)  # Receives the output of the above command
			vms.append(vm)

		for vm in reversed(vms):
			for i in range(60,-1,-1):
				print("Desliga em: %3d\r"%i,)
				time.sleep(1)
				sys.stdout.flush()
			print('desligando %s' %vm)
			command = "ssh user@controller '. admin-openrc && openstack server delete %s'" %vm
			run = subprocess.check_output(command, shell=True)  # Receives the output of the above command

try:
	r = requests.get('http://controller:8774/v2.1/servers', headers=header.get())
except requests.exceptions.ConnectionError as e:
	raise requests.exceptions.ConnectionError(f"{e}: This computer does not have communication with the Controller.\nCheck the requirements in https://github.com/dssantos/Cloud-Energy-Saver2")

vm_list = json.loads(r.content) # Returns the content of the queried URL
vm_list = vm_list['servers']
length = len(vm_list)
