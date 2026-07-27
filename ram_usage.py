#coding: utf-8

import subprocess, string, ast

def get(host): # The method must receive a host

		command = "ssh user@" + host + " 'free -m | grep Mem'" # Command for obtaining remote host RAM information

		try:
			# Add timeout to avoid hanging when host is down
			output = subprocess.check_output(command, shell=True, timeout=5,
				stderr=subprocess.DEVNULL).decode()  # suppress SSH stderr noise when host is unreachable
			mem_info = output.split() # Transforms strings of values separated by spaces in a list
			ram_usage = float(mem_info[2])/float(mem_info[1])*100 # Calculates the percentage of RAM usage

		except Exception:
			ram_usage = 0.0  # If the host is not reached, the command will result in an error, so the RAM usage will be zero

		return ram_usage


def get_extended(host):
	"""Collect multiple metrics from a host in a single SSH call.

	Returns a dict with: mem (%), swap (%), loadavg (1-min).
	Each field is 0.0 when the host is unreachable / parsing fails.
	Output order from the remote command: loadavg line, then 'Mem:' line, then 'Swap:' line.
	"""
	command = "ssh user@" + host + " 'cat /proc/loadavg; free -m | grep -E \"Mem|Swap\"'"
	result = {'mem': 0.0, 'swap': 0.0, 'loadavg': 0.0}
	try:
		output = subprocess.check_output(command, shell=True, timeout=5,
			stderr=subprocess.DEVNULL).decode()
		lines = [ln for ln in output.splitlines() if ln.strip()]
		for ln in lines:
			parts = ln.split()
			if parts and parts[0] == 'Mem:' and len(parts) >= 3:
				result['mem'] = float(parts[2]) / float(parts[1]) * 100
			elif parts and parts[0] == 'Swap:' and len(parts) >= 3 and float(parts[1]) > 0:
				result['swap'] = float(parts[2]) / float(parts[1]) * 100
			elif len(parts) >= 1 and '/' in ln and parts[0].replace('.', '', 1).isdigit():
				# /proc/loadavg: "0.52 0.43 0.38 2/1234 5678"
				result['loadavg'] = float(parts[0])
	except Exception:
		pass
	return result