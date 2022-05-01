#coding: utf-8
import json

import requests, header, ast, vms, ram_usage

def get():

    r = requests.get('http://controller:8774/v2.1/os-hypervisors', headers=header.get())
    hosts = json.loads(r.content) # Returns the content of the queried URL
    hosts = hosts['hypervisors']
    ## Tests
    # hosts = [{'status': 'enabled', 'state': 'down', 'id': 1, 'hypervisor_hostname': 'compute1'},{'status': 'enabled', 'state': 'down', 'id': 1, 'hypervisor_hostname': 'compute2'},{'status': 'enabled', 'state': 'down', 'id': 1, 'hypervisor_hostname': 'compute3'},{'status': 'enabled', 'state': 'up', 'id': 1, 'hypervisor_hostname': 'compute6'},{'status': 'enabled', 'state': 'up', 'id': 1, 'hypervisor_hostname': 'compute5'}]

    status = []

    try:
        file = open("registered.txt", "r+")
        registered = file.read()	
        registered = ast.literal_eval(registered)
    except:
        print('É preciso registrar os hosts do ambiente')
        registered = []

    for host in hosts:
        if host['hypervisor_hostname'] in registered:  # Check only the hosts registered in the registered.txt file
            state = host['state']
            host_id = host['id']
            hostname = host['hypervisor_hostname']
            vms_runnig = vms.get(host_id)
            if state == 'up':
                ram = ram_usage.get(hostname)
            else:
                ram = 0
            host_status = "{'id': " + str(host_id) + ", 'hostname': '" + hostname + "', 'state': '" + state + "', 'vms': " + str(vms_runnig) + ", 'ram': " + str(ram) + "}"
            host_status = ast.literal_eval(host_status)

            status.append(host_status)  ## Adds information from each active host to a list...

    return status