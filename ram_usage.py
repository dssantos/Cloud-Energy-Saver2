#coding: utf-8

import subprocess, string, ast

def get(host): # The method must receive a host

		command = "ssh user@" + host + " 'free -m | grep Mem'" # Command for obtaining remote host RAM information
		
		try:
			output = subprocess.check_output(command, shell=True).decode()  # Receives the output of the above command
			mem_info = output.split() # Transforms strings of values separated by spaces in a list
			ram_usage = float(mem_info[2])/float(mem_info[1])*100 # Calculates the percentage of RAM usage

		except subprocess.CalledProcessError:
			ram_usage = 0.0  # If the host is not reached, the command will result in an error, so the RAM usage will be zero

		return ram_usage