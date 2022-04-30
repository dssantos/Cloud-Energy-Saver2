#!/usr/bin/env python

import sys


if len(sys.argv) <= 1:

	print('Enter some parameter!\nHelp:\n./ces -h ')

else:
	arg1 = sys.argv[1]

	if arg1 == '--help' or arg1 == '-h':
		help_msg = '''
#####  Cloud Energy Saver (CES) #####
Host state manager for OpenStack Cloud Computing environments that allows for power management experiments

Syntax:
	./ces [-opção] [PARAMS]

Options and Parameters:

	-h,   --help                        shows this help
	-r,   --registrator                 identifies and registers hosts
	-v,   --verifier [MAX] [MED]        starts idle and overload check
	                                    MAX and MED are percentages of RAM in use on Compute hosts and represent the limits that define when to start hosts (when the environment is above MAX) or turn off hosts (when the environment is below the MED)
	-i,   --instantiator [QT]           starts a [QT] number of instances one by one, every 30 seconds, and then shut down one by one, continuously
	-on,  --on [QT]                     starts a quantity [QT] of instances
	-off, --off [QT]                    shut offs quantity [QT] of instances
	-s,   --status                      shows information about Compute hosts
		'''
		print(help)
	else:

		import registrator, verifier, status, instances

		
		if len(sys.argv) > 2:
			arg2 = int(sys.argv[2])


		if arg1 == '--registrator' or arg1 == '-r':
			registrator.run()

		if arg1 == '--verifier' or arg1 == '-v':
			if len(sys.argv) > 3:
				lim_max = int(sys.argv[2])
				lim_med = int(sys.argv[3])
				verifier.start(lim_max, lim_med)
			else:
				print('Enter a maximum and an medium limit\nEx: ./ces -v 40 30')

		if arg1 == '--instantiator' or arg1 == '-i' or arg1 == '-auto':
			if len(sys.argv) > 2:
				qt_instances = int(sys.argv[2])
				instances.auto_on(qt_instances)
			else:
				print('Inform a number of VMs to be instantiated\nEx: ./ces -i 50')
		
		if arg1 == '--on' or arg1 == '-on':
			if len(sys.argv) > 2:
				qt_on = int(sys.argv[2])
				instances.on(qt_on)
			else:
				print('Enter a quantity of VMs to initiate\nEx: ./ces -on 5')

		if arg1 == '--off' or arg1 == '-off':
			if len(sys.argv) > 2:
				qt_off = int(sys.argv[2])
				instances.off(qt_off)
			else:
				print('Enter a quantity of VMs to shut down\nEx: ./ces -off 5')

		if arg1 == '--status' or arg1 == '-s':
			hosts = status.get()

			if len(hosts) < 1:

				print("There are no registered Compute hosts!\nRun './ces -r' to register them")

			else:
				print("[Compute Hosts Status]\n")
				for host in hosts:
					print('%s [%s]' %(host['hostname'], host['state']))
					print('RAM: {} %'.format(host['ram']))
					try:
						print('VMs: %s\n' %host['vms'])
					except:
						pass
		# except:
		# 	print("Something is wrong with the OpenStack environment or this computer does not have communication with the Controller.\nCheck the requirements in https://github.com/dssantos/Cloud-Energy-Saver")