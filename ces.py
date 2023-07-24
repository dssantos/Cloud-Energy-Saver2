#!/usr/bin/env python

import sys
from time import sleep

import registrator, verifier, status, instances


try:
	arg1 = sys.argv[1]
except:
    arg1 = 'empty'

valid_params = [
	'-r', '--registrator',
	'-v', '--verifier',
	'-i', '--instantiator',
	'-on', '--on',
	'-off', '--off',
	'-s', '--status',
]

help_msg = '''
#####  Cloud Energy Saver (CES) #####
Host state manager for OpenStack Cloud Computing environments that allows for power management experiments

Syntax:
	./ces [-option] [PARAMS]

Options and Parameters:

	-r,   --registrator                 	identifies and registers hosts
	-v,   --verifier [MAX] [MED] [MODEL]    starts idle and overload check
	                                    	MAX and MED are percentages of RAM in use on Compute hosts and represent the limits that define when to start hosts (when the environment is above MAX) or turn off hosts (when the environment is below the MED)
											MODEL (optional): default, naive, arima or lstm. Example: python ces.py -v 70 30 arima
	-i,   --instantiator [QT]           	starts a [QT] number of instances one by one, every 30 seconds, and then shut down one by one, continuously
	-on,  --on [QT]                     	starts a quantity [QT] of instances
	-off, --off [QT]                    	shut offs quantity [QT] of instances
	-s,   --status                      	shows information about Compute hosts
		'''

if arg1 not in valid_params:
    print(help_msg)
    
else:
	# try:
	if arg1 == '--registrator' or arg1 == '-r':
		registrator.run()
	if arg1 == '--verifier' or arg1 == '-v':
		if len(sys.argv) > 3:
			lim_max = int(sys.argv[2])
			lim_med = int(sys.argv[3])
			if len(sys.argv) > 4:
				predict_model = sys.argv[4]
				verifier.start(lim_max, lim_med, predict_model)
			verifier.start(lim_max, lim_med, 'default')
		else:
			print('Enter a maximum and an medium limit\nEx: python ces.py -v 40 30\n Optionaly you can pass an predict model. Ex: python ces.py -v 40 30 naive')
	if arg1 == '--instantiator' or arg1 == '-i' or arg1 == '-auto':
		if len(sys.argv) > 2:
			qt_instances = int(sys.argv[2])
			instances.auto_on(qt_instances)
		else:
			print('Inform a number of VMs to be instantiated\nEx: python ces.py -i 50')
	if arg1 == '--on' or arg1 == '-on':
		if len(sys.argv) > 2:
			qt_on = int(sys.argv[2])
			instances.on(qt_on)
		else:
			print('Enter a quantity of VMs to initiate\nEx: python ces.py -on 5')
	if arg1 == '--off' or arg1 == '-off':
		if len(sys.argv) > 2:
			qt_off = int(sys.argv[2])
			instances.off(qt_off)
		else:
			print('Enter a quantity of VMs to shut down\nEx: python ces.py -off 5')
	if arg1 == '--status' or arg1 == '-s':
		while True:
			try:
				hosts = status.get()
				if len(hosts) < 1:
					print("There are no registered Compute hosts!\nRun 'python ces.py -r' to register them")
				else:
					print("[Compute Hosts Status]\n")
					for host in hosts:
						print('%s [%s]' %(host['hostname'], host['state']))
						print('RAM: {} %'.format(host['ram']))
						try:
							print('VMs: %s\n' %host['vms'])
						except:
							pass
			except:
				pass
			sleep(10)
	# except:
	# 	print("Something is wrong with the OpenStack environment or this computer does not have communication with the Controller.\nCheck the requirements in https://github.com/dssantos/Cloud-Energy-Saver2")
