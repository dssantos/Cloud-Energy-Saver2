#coding: utf-8

import mac
from os import system
from subprocess import Popen, PIPE, STDOUT

from wakeonlan import send_magic_packet


def wake(host):

	mac_list = mac.get(host)
	mac_address = ''
	for mac_address in mac_list:
		if mac_address[:8] == b'08:00:27':
			print(f'Iniciando VM {host}')
			system(f'vboxmanage startvm {host} --type=headless')
			break
		send_magic_packet(mac_address.decode(), interface="10.0.0.0")

def shutdown(host):

	command = "ssh user@%s 'sudo shutdown now'" %host

	p = Popen(command, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT, close_fds=True) # Runs command and store STDOUT
	output = p.stdout.read()
	print(output)