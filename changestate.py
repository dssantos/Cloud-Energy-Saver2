#coding: utf-8

import mac
import os
from dotenv import load_dotenv
from os import system
from subprocess import Popen, PIPE, STDOUT, CalledProcessError
import threading
import time

from wakeonlan import send_magic_packet

# Carregar variáveis de ambiente
load_dotenv()

# Configuração do Windows com VirtualBox
WINDOWS_HOST = os.getenv('WINDOWS_HOST')
WINDOWS_USER = os.getenv('WINDOWS_USER')

def _try_wake_async(method_name, command):
    """Executa método de wake em background thread."""
    try:
        # Usa Popen para executar em background sem esperar
        Popen(command, shell=True, stdout=PIPE, stderr=PIPE, stdin=PIPE)
        print(f'  [WAKE] {method_name}: disparado')
    except Exception as e:
        print(f'  [WAKE] {method_name}: erro ao disparar - {e}')

def _ping_monitor(host, timeout_seconds, result_container):
    """Monitora host com ping até responder ou timeout."""
    start_time = time.time()
    print(f'  [PING] Iniciando monitoramento de {host} (timeout: {timeout_seconds}s)...')

    while time.time() - start_time < timeout_seconds:
        try:
            # Usa ping com timeout de 1s
            result = system(f'ping -c 1 -W 1 {host} 2>/dev/null >/dev/null')
            if result == 0:
                elapsed = time.time() - start_time
                print(f'  [PING] ✓ {host} respondeu após {elapsed:.1f}s')
                result_container['success'] = True
                result_container['time'] = elapsed
                return
        except:
            pass
        time.sleep(2)  # Verifica a cada 2 segundos

    elapsed = time.time() - start_time
    print(f'  [PING] ✗ {host} não respondeu após {elapsed:.1f}s (timeout)')
    result_container['success'] = False
    result_container['time'] = timeout_seconds

def wake(host, timeout_seconds=120):
    """
    Acorda host usando múltiplos métodos em paralelo.
    Retorna True se host responder ao ping dentro do timeout, False caso contrário.
    """
    mac_list = mac.get(host)
    result_container = {'success': False, 'time': 0}
    threads = []

    print(f'[WAKE] Iniciando wake paralelo para {host}...')

    # Disparar todos os métodos de wake em paralelo (fire-and-forget)
    for mac_address in mac_list:
        if mac_address[:8] == b'08:00:27':
            # VirtualBox VM - disparar ambos métodos
            threads.append(threading.Thread(
                target=_try_wake_async,
                args=('VBox Local', f'vboxmanage startvm {host} --type=headless'),
                daemon=True
            ))
            threads.append(threading.Thread(
                target=_try_wake_async,
                args=('VBox SSH', f'ssh {WINDOWS_USER}@{WINDOWS_HOST} "VBoxManage startvm {host} --type=headless"'),
                daemon=True
            ))
            break
        else:
            # Hardware real - usar Wake-on-LAN
            if mac_address != b'00:00:00:00:00:00':
                threads.append(threading.Thread(
                    target=_try_wake_async,
                    args=('WOL', f'wakeonlan {mac_address.decode()} -i 10.0.0.0'),
                    daemon=True
                ))

    # Iniciar todas as threads de wake (não aguardamos resultado)
    for thread in threads:
        thread.start()

    # Pequena pausa para os comandos iniciarem
    time.sleep(2)

    # Iniciar monitoramento com ping (este que determina o sucesso)
    ping_thread = threading.Thread(
        target=_ping_monitor,
        args=(host, timeout_seconds, result_container),
        daemon=False  # Não daemon para completar mesmo que main thread termine
    )
    ping_thread.start()

    # Aguardar resultado do ping
    ping_thread.join()

    success = result_container['success']
    elapsed = result_container['time']

    print(f'[WAKE] {host}: {"✓ Sucesso" if success else "✗ Falha"} ({elapsed:.1f}s)')
    return success

def shutdown(host):

	command = "ssh user@%s 'sudo shutdown now'" %host

	p = Popen(command, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT, close_fds=True) # Runs command and store STDOUT
	output = p.stdout.read()
	print(output)