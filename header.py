#coding: utf-8
import json

import requests, os, time
def get():
    domain = 'Default'
    login = 'admin'
    password = '123456'
    project = 'admin'

    headers = {'Content-Type':'application/json'}
    body = """
    {
        "auth": {
            "identity": {
                "methods": [
                "password"
                ],
                "password": {
                    "user": {
                        "domain": {
                            "name": \""""+ domain +"""\"
                        },
                    "name": \""""+ login +"""\",
                    "password": \""""+ password +"""\"
                    }
                }
            },
            "scope": {
                "project": {
                    "domain": {
                        "name": \""""+ domain +"""\"
                    },
                    "name": \""""+ project +"""\"
                }
            }
        }
    }
    """
    now = time.time() # current timestamp
    try:
        modified = os.path.getmtime('token.txt') # Last file modification
    except:
        modified = 0 # if the file does not exist, modified will be zero, so it will enter the else to generate the token
    if now - modified < 3600: # check if modified is less than 1 hour (OpenStack Keystone token expires in 1 hour)
        file = open("token.txt")
        header = file.read()
        header = json.loads(header.replace('\'', '\"'))
    else: # generate new token 
        r = requests.post('http://controller:5000/v3/auth/tokens', data=body, headers=headers)
        token = r.headers['X-Subject-Token']
        header = {'X-Auth-Token':token}  # Token string
        file = open("token.txt", "w+")
        file.write(str(header))
        file.close()
    return header