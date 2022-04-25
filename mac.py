#coding: utf-8

import subprocess, ast

def get(host):
    file = open("%s.txt"%host, "r+")
    macs = file.read()
    macs = ast.literal_eval(macs)
    return macs

def set(host):
    macs = []
    command1 = "ssh user@%s 'ls /sys/class/net'" %host
    try:
        list_intefaces = subprocess.check_output(command1, shell=True)
        list_intefaces = list_intefaces.decode().split('\n')

        for interface in list_intefaces:
            if interface != '':
                command = "ssh user@%s 'cat /sys/class/net/%s/address'" %(host, interface) # command to return mac address
                mac = subprocess.check_output(command, shell=True)  # Receives the output of the above command
                macs.append(mac.rstrip())

    except subprocess.CalledProcessError:

        print('Não foi possível obter o MAC de %s'%host)


    file = open("%s.txt"%host, "w+")
    file.write(str(macs))
    file.close()

    print( '%s %s'%(host, macs))