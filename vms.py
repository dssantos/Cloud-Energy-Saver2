#coding: utf-8

import requests, json, header

def get(host_id): # The method must receive a host_id and the headers (token)
	
	try:
		r = requests.get('http://controller:8774/v2.1/os-hypervisors/%s'%host_id, headers=header.get())
		vms = json.loads(r.content)
		vms = vms[u'hypervisor']["running_vms"]

	except:
		vms = -1  # If the id is not found in the query, the command will result in an error, then vms will be zero

	return vms # Returns the amount of vms running on the host