#coding: utf-8

import mac
import os
from dotenv import load_dotenv
from os import system
from subprocess import Popen, PIPE, STDOUT, CalledProcessError

from wakeonlan import send_magic_packet

# Carregar variáveis de ambiente
load_dotenv()

# Configuração do Windows com VirtualBox
WINDOWS_HOST = os.getenv('WINDOWS_HOST')
WINDOWS_USER = os.getenv('WINDOWS_USER')

def wake(host):

	mac_list = mac.get(host)
	mac_address = ''
	for mac_address in mac_list:
		if mac_address[:8] == b'08:00:27':
			print(f'Iniciando VM {host}')
			# Tentar VBoxManage local primeiro
			try:
				result = system(f'vboxmanage startvm {host} --type=headless')
				if result == 0:
					break
				print('VBoxManage local falhou, tentando via SSH...')
			except:
				print('VBoxManage local não disponível, tentando via SSH...')

			# Fallback: Executar VBoxManage remotamente no Windows via SSH
			vbox_cmd = f'ssh {WINDOWS_USER}@{WINDOWS_HOST} "VBoxManage startvm {host} --type=headless"'
			system(vbox_cmd)
			break
		else:
			if mac_address != b'00:00:00:00:00:00':
				print(f'Enviando Wake-on-LAN para {host}')
				send_magic_packet(mac_address.decode(), interface="10.0.0.0")

def shutdown(host):

	command = "ssh user@%s 'sudo shutdown now'" %host

	p = Popen(command, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT, close_fds=True) # Runs command and store STDOUT
	output = p.stdout.read()
	print(output)