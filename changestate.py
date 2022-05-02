#coding: utf-8

import subprocess, mac
from subprocess import Popen, PIPE, STDOUT

def wake(host):

	mac_list = mac.get(host)
	mac_address = ''
	for mac_address in mac_list:
		command = "sudo etherwake -i eno1 %s" %mac_address
		output = subprocess.check_output(command, shell=True)

def shutdown(host):

	command = "ssh user@%s 'sudo shutdown now'" %host

	p = Popen(command, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT, close_fds=True) # Runs command and store STDOUT
	output = p.stdout.read()
	print(output)