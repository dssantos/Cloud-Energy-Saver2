#coding: utf-8
import os
from time import sleep
from datetime import datetime
import sys, status, changestate, ast
import threading

import workload, predict
import event_logger
from event_logger import Event

lstm_manager = None
experiment_start_time = None


def check_sla_violation(running_hosts, idle_hosts, offline_hosts, ram_avg, lim_max):
    """
    Check for SLA violation conditions.
    Returns True if system cannot handle current load.
    """
    if ram_avg > lim_max and len(offline_hosts) == 0 and len(idle_hosts) == 0:
        return True
    if len(running_hosts) == 0 and ram_avg > lim_max:
        return True
    return False


def log_initial_state():
    """Log the initial state of all registered hosts when experiment starts."""
    global experiment_start_time
    experiment_start_time = datetime.now()

    hosts = status.get()
    try:
        with open("registered.txt", "r") as file:
            registered = ast.literal_eval(file.read())
    except:
        registered = []

    for host in hosts:
        event_logger.logger.log(Event(
            timestamp=experiment_start_time.isoformat(),
            event_type='initial_state',
            hostname=host['hostname'],
            trigger_type='system',
            ram_avg=0.0,
            lim_max=0.0,
            lim_med=0.0,
            running_hosts=0,
            idle_hosts=0,
            offline_hosts=0,
            predicted_ram=None,
            actual_ram=float(host['ram']) if host.get('ram') else 0.0,
            initial_state=host['state'],  # 'up' or 'down'
            initial_vms=int(host['vms']) if host.get('vms') is not None else 0
        ))

    print(f'Logged initial state for {len(hosts)} hosts')


def log_final_state():
    """Log the final state of all registered hosts when experiment stops."""
    global experiment_start_time

    if experiment_start_time is None:
        return

    hosts = status.get()
    try:
        with open("registered.txt", "r") as file:
            registered = ast.literal_eval(file.read())
    except:
        registered = []

    for host in hosts:
        event_logger.logger.log(Event(
            timestamp=datetime.now().isoformat(),
            event_type='final_state',
            hostname=host['hostname'],
            trigger_type='system',
            ram_avg=0.0,
            lim_max=0.0,
            lim_med=0.0,
            running_hosts=0,
            idle_hosts=0,
            offline_hosts=0,
            predicted_ram=None,
            actual_ram=float(host['ram']) if host.get('ram') else 0.0,
            final_state=host['state'],  # 'up' or 'down'
            final_vms=int(host['vms']) if host.get('vms') is not None else 0
        ))

    print(f'Logged final state for {len(hosts)} hosts')


def run(lim_max, lim_med, predict_model):

    hosts = status.get()
    ram = []
    running = []
    idle = []
    offline = []

    # Store predictions for LSTM (to log with wake events)
    predictions = {}  # hostname -> (predicted_ram, actual_ram)

    try:
        file = open("registered.txt", "r+")
        registered = file.read()
        registered = ast.literal_eval(registered)
    except:
        print('É preciso registrar os hosts do ambiente')
        registered = []

    for host in hosts:
        if host['state'] == 'up':
            if host['vms'] > 0:
                running.append(host['hostname']) # Inserts the hosts that are connected (and have VMs) in an list of actives
            else:
                idle.append(host['hostname']) # Inserts hosts that are running (and do not have VMs) in a list of idlers

            actual_ram = host['ram']
            if predict_model == 'default':
                ram.append(actual_ram)
            elif predict_model == 'naive':
                print(f'Running {predict_model} predict model')
                predicted = predict.naive(host['hostname'])
                ram.append(predicted)
                if predict_model == 'lstm':  # Shouldn't happen but keeping for consistency
                    predictions[host['hostname']] = (predicted, actual_ram)
            elif predict_model == 'arima':
                print(f'Running {predict_model} predict model')
                predicted = predict.arima(host['hostname'])
                ram.append(predicted)
                if predict_model == 'lstm':
                    predictions[host['hostname']] = (predicted, actual_ram)
            elif predict_model == 'lstm':
                print(f'Running {predict_model} predict model')
                predicted = predict.lstm(host['hostname'])
                ram.append(predicted)
                predictions[host['hostname']] = (predicted, actual_ram)
            else:
                print(f'Predict model "{predict_model}" not supported yet, running default mode')
                ram.append(actual_ram)
        else:
            if host['hostname'] in registered:
                offline.append(host['hostname']) # Inserts hosts that are shut down (and registered) in an list of offline

    try:
        ram_avg = sum(ram) / len(ram) # Calculates an average of memory in use by active hosts
    except:
        ram_avg = 0

    print('ativos: ' + str(running))
    print('ociosos: ' + str(idle))
    print('offline: ' + str(offline))
    print('média de ram: %s' %ram_avg)

## Logic of the management of the hosts to be turned on and off

    if ram_avg > lim_max:						## If RAM is above the maximum limit
        if len(idle) > 0:
            if len(idle) > 1:					## They keep 1 idle on and shut off others
                for i in range(len(idle)-1):	# Turn off all except 1
                    print ('desligando %s' %idle[i+1])
                    event_logger.logger.log(Event(
                        timestamp=datetime.now().isoformat(),
                        event_type='shutdown',
                        hostname=idle[i+1],
                        trigger_type=predict_model,
                        ram_avg=ram_avg,
                        lim_max=lim_max,
                        lim_med=lim_med,
                        running_hosts=len(running),
                        idle_hosts=len(idle),
                        offline_hosts=len(offline)
                    ))
                    changestate.shutdown(idle[i+1])
        else:
            if len(offline) > 0:				# If there are offline hosts ...
                print('ligando %s' %offline[0])
                # Get prediction values for LSTM wake events
                pred_ram = None
                act_ram = None
                if predict_model == 'lstm' and predictions:
                    # Average of all predictions
                    all_preds = [p[0] for p in predictions.values() if p[0] is not None]
                    all_actuals = [p[1] for p in predictions.values() if p[1] is not None]
                    if all_preds:
                        pred_ram = sum(all_preds) / len(all_preds)
                    if all_actuals:
                        act_ram = sum(all_actuals) / len(all_actuals)

                event_logger.logger.log(Event(
                    timestamp=datetime.now().isoformat(),
                    event_type='wake',
                    hostname=offline[0],
                    trigger_type=predict_model,
                    ram_avg=ram_avg,
                    lim_max=lim_max,
                    lim_med=lim_med,
                    running_hosts=len(running),
                    idle_hosts=len(idle),
                    offline_hosts=len(offline),
                    predicted_ram=pred_ram,
                    actual_ram=act_ram
                ))
                changestate.wake(offline[0]) 			# Wake up the first offline host from the list
            else:
                if check_sla_violation(running, idle, offline, ram_avg, lim_max):
                    event_logger.logger.log(Event(
                        timestamp=datetime.now().isoformat(),
                        event_type='sla_violation',
                        hostname=None,
                        trigger_type=predict_model,
                        ram_avg=ram_avg,
                        lim_max=lim_max,
                        lim_med=lim_med,
                        running_hosts=len(running),
                        idle_hosts=len(idle),
                        offline_hosts=len(offline)
                    ))
                print('SLA VIOLAÇÃO: Não há mais hosts offline para ligar.\nO sistema está no limite!!!')
    else:
        if len(idle) > 0:
            if ram_avg >= lim_med:				## If RAM is between the medium and maximum limits
                for i in range(len(idle)-1):	# Turn off all except 1
                    print('desligando %s' %idle[i+1])
                    event_logger.logger.log(Event(
                        timestamp=datetime.now().isoformat(),
                        event_type='shutdown',
                        hostname=idle[i+1],
                        trigger_type=predict_model,
                        ram_avg=ram_avg,
                        lim_max=lim_max,
                        lim_med=lim_med,
                        running_hosts=len(running),
                        idle_hosts=len(idle),
                        offline_hosts=len(offline)
                    ))
                    changestate.shutdown(idle[i+1])
            else:
                if len(running) >= 1:		## If there is at least 1 active host
                    for host in idle:
                        print('desligando %s' %host)
                        event_logger.logger.log(Event(
                            timestamp=datetime.now().isoformat(),
                            event_type='shutdown',
                            hostname=host,
                            trigger_type=predict_model,
                            ram_avg=ram_avg,
                            lim_max=lim_max,
                            lim_med=lim_med,
                            running_hosts=len(running),
                            idle_hosts=len(idle),
                            offline_hosts=len(offline)
                        ))
                        changestate.shutdown(host)		# shut down all idle hosts
                else:								# Else...
                    for i in range(len(idle)-1):	# Turn off all except 1
                        print('desligando %s' %idle[i+1])
                        event_logger.logger.log(Event(
                            timestamp=datetime.now().isoformat(),
                            event_type='shutdown',
                            hostname=idle[i+1],
                            trigger_type=predict_model,
                            ram_avg=ram_avg,
                            lim_max=lim_max,
                            lim_med=lim_med,
                            running_hosts=len(running),
                            idle_hosts=len(idle),
                            offline_hosts=len(offline)
                        ))
                        changestate.shutdown(idle[i+1])

def start(lim_max, lim_med, predict_model):
    global lstm_manager, experiment_start_time

    # Initialize event logger with model-specific filename
    event_file = f'events_{predict_model}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    event_logger.logger = event_logger.EventLogger(event_file)
    print(f'Event logging initialized: {event_file}')

    # Log initial state
    log_initial_state()

    # Initialize LSTM training manager if needed
    if predict_model == 'lstm':
        try:
            with open("registered.txt", "r") as file:
                registered = ast.literal_eval(file.read())

            for hostname in registered:
                predict.lstm_manager.start_training(hostname)

            print(f'Started LSTM training for {len(registered)} hosts')
        except Exception as e:
            print(f'Error initializing LSTM training: {e}')

    # Start workload collection
    print('\n\nIniciando coleta de cargas de trabalho...\n')
    hosts = status.get()
    for host in hosts:
        threading.Thread(target=workload.save, args=[host['hostname']]).start()

    # Main verification loop
    try:
        while True:
            print('\n\nVerificando Hosts...\n')
            run(lim_max, lim_med, predict_model)

            for i in range(90,-1,-1):
                sys.stdout.write("  Próxima verificação: %3d\r"%i)
                sys.stdout.flush()
                sleep(1)
            print("  Próxima verificação:   0  ")
    except KeyboardInterrupt:
        print('\n\nStopping verifier...')
        # Log final state before exiting
        log_final_state()
        if predict_model == 'lstm':
            predict.lstm_manager.stop_training()
        raise
