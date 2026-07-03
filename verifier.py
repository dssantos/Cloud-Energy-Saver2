#coding: utf-8
import os
import time
from time import sleep
from datetime import datetime
import sys, status, changestate, ast
import threading
import smtplib
from email.message import EmailMessage
from dotenv import load_dotenv

import workload, predict
import event_logger
from event_logger import Event

# Load environment variables for email alerts
load_dotenv()

lstm_manager = None
experiment_start_time = None

# Emergency wake cooldown tracking
# Format: {hostname: timestamp_of_wake}
emergency_wake_cooldowns = {}
EMERGENCY_COOLDOWN_SECONDS = 300  # 5 minutes cooldown after emergency wake


def send_alert_email(hostname, target_state, timeout_seconds, details=None):
    """
    Send alert email when host fails to reach target state.
    Application continues after sending alert.
    """
    try:
        alert_email = os.getenv('ALERT_EMAIL')
        app_password = os.getenv('ALERT_APP_PASSWORD')

        if not alert_email or not app_password:
            print('[ALERT] Email credentials not configured, skipping email notification')
            return

        msg = EmailMessage()
        msg['Subject'] = f'CES ALERT: Host {hostname} failed to reach {target_state}'
        msg['From'] = alert_email
        msg['To'] = alert_email

        # Collect maximum useful information
        details_text = ""
        if details:
            details_text = f"\n\nDetails:\n{details}"

        body = f"""
CES Alert - Host State Change Timeout

Host: {hostname}
Target State: {target_state}
Timeout: {timeout_seconds} seconds ({timeout_seconds//60} minutes)
Timestamp: {datetime.now().isoformat()}

Note: Application continues running despite this failure.{details_text}

Please check the host and environment status.
"""
        msg.set_content(body)

        # Enviar via Gmail SMTP
        smtp_host = os.getenv('SMTP_HOST', 'smtp.gmail.com')
        smtp_port = int(os.getenv('SMTP_PORT', '587'))

        smtp = smtplib.SMTP(smtp_host, smtp_port)
        smtp.starttls()
        smtp.login(alert_email, app_password)
        smtp.send_message(msg)
        smtp.quit()
        print(f'[ALERT] Email sent for host {hostname}')
    except Exception as e:
        print(f'[ALERT] Failed to send email: {e}')


def wait_for_state_change(hostname, target_state, timeout=300, details_collector=None):
    """
    Wait for host to reach target state (up/down).
    Max wait: 5 minutes (300 seconds).
    On timeout: Send email alert and CONTINUE (don't exit).
    Returns: True if successful, False if timeout (but continues)
    """
    print(f'[STATE] Aguardando {hostname} -> {target_state} (timeout: {timeout}s)...')
    start_time = time.time()
    check_interval = 10  # Check every 10 seconds

    while time.time() - start_time < timeout:
        hosts = status.get()
        for host in hosts:
            if host['hostname'] == hostname and host['state'] == target_state:
                elapsed = time.time() - start_time
                print(f'[STATE] {hostname} alcançou {target_state} em {elapsed:.1f}s')
                return True
        time.sleep(check_interval)

    # Timeout - Send alert but DON'T exit
    elapsed = time.time() - start_time
    details = f"Elapsed time: {elapsed:.1f}s\nState transition did not complete within timeout."
    print(f'[TIMEOUT] {hostname} não alcançou {target_state} em {timeout}s')
    send_alert_email(hostname, target_state, timeout, details)

    # Collect additional data for analysis
    if details_collector:
        details_collector.record_timeout(hostname, target_state, elapsed)

    # Mark as failed but continue execution
    return False


def is_in_cooldown(hostname):
    """Check if host is in emergency wake cooldown period."""
    if hostname not in emergency_wake_cooldowns:
        return False
    elapsed = time.time() - emergency_wake_cooldowns[hostname]
    if elapsed > EMERGENCY_COOLDOWN_SECONDS:
        # Cooldown expired, remove from tracking
        del emergency_wake_cooldowns[hostname]
        return False
    return True


def calculate_ram_average(hosts_data, lim_max, predict_model='default'):
    """
    Calculate RAM average excluding overloaded hosts (RAM > lim_max).
    Returns: (avg_ram, overloaded_list, normal_list)
    """
    ram_values = []
    overloaded = []
    normal = []

    for host in hosts_data:
        if host['state'] == 'up':
            ram_val = host['ram']
            if predict_model == 'lstm':
                ram_val = predict.lstm(hostname=host['hostname'], steps_ahead=6)
            elif predict_model == 'naive':
                ram_val = predict.naive(host['hostname'])
            elif predict_model == 'arima':
                ram_val = predict.arima(host['hostname'])

            if ram_val > lim_max:
                overloaded.append(host['hostname'])
            else:
                normal.append(host['hostname'])
                ram_values.append(ram_val)

    avg_ram = sum(ram_values) / len(ram_values) if ram_values else 0
    return avg_ram, overloaded, normal


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
        else:
            if host['hostname'] in registered:
                offline.append(host['hostname']) # Inserts hosts that are shut down (and registered) in an list of offline

    # Calculate RAM average excluding overloaded hosts
    hosts_for_avg = status.get()
    ram_avg, overloaded, normal = calculate_ram_average(hosts_for_avg, lim_max, predict_model)

    print('ativos: ' + str(running))
    print('ociosos: ' + str(idle))
    print('offline: ' + str(offline))
    print(f'sobrecarregados: {overloaded}')
    print(f'normais: {normal}')
    print('média de ram: %s' %ram_avg)

## Logic of the management of the hosts to be turned on and off

    # EMERGENCY LOGIC: When all normal hosts are overloaded and there are offline hosts
    if len(overloaded) > 0 and len(normal) == 0 and len(offline) > 0:
        emergency_host = offline[0]
        print(f'EMERGÊNCIA: Todos os hosts normais sobrecarregados! Acordando {emergency_host}...')
        event_logger.logger.log(Event(
            timestamp=datetime.now().isoformat(),
            event_type='wake',
            hostname=emergency_host,
            trigger_type=f'{predict_model}_emergency',
            ram_avg=ram_avg,
            lim_max=lim_max,
            lim_med=lim_med,
            running_hosts=len(running),
            idle_hosts=len(idle),
            offline_hosts=len(offline)
        ))
        # Set cooldown to prevent immediate shutdown
        emergency_wake_cooldowns[emergency_host] = time.time()
        changestate.wake(emergency_host)
    elif ram_avg > lim_max:						## If RAM is above the maximum limit
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
                # Filter out hosts in cooldown
                idle_non_cooldown = [h for h in idle if not is_in_cooldown(h)]
                # Keep at least 1 host (either not in cooldown or the first one)
                shutdown_candidates = idle_non_cooldown[:-1] if len(idle_non_cooldown) > 1 else []
                for host in shutdown_candidates:	# Turn off all except 1
                    print('desligando %s' % host)
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
                    changestate.shutdown(host)
            else:
                if len(running) >= 1:		## If there is at least 1 active host
                    # Filter out hosts in cooldown - never shut down emergency hosts
                    idle_non_cooldown = [h for h in idle if not is_in_cooldown(h)]
                    cooldown_hosts = [h for h in idle if is_in_cooldown(h)]
                    # Log cooldown protected hosts
                    for host in cooldown_hosts:
                        elapsed = time.time() - emergency_wake_cooldowns[host]
                        remaining = EMERGENCY_COOLDOWN_SECONDS - elapsed
                        print(f'[COOLDOWN] {host} protegido por {remaining:.0f}s (emergency wake)')
                    for host in idle_non_cooldown:
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
                    # Filter out hosts in cooldown
                    idle_non_cooldown = [h for h in idle if not is_in_cooldown(h)]
                    # Keep at least 1 host (either not in cooldown or the first one)
                    shutdown_candidates = idle_non_cooldown[:-1] if len(idle_non_cooldown) > 1 else []
                    for host in shutdown_candidates:	# Turn off all except 1
                        print('desligando %s' % host)
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
                        changestate.shutdown(host)

def start(lim_max, lim_med, predict_model, continuous=False):
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
        if continuous:
            print('\n=== Modo Loop Contínuo ===')
            print('Pressione Ctrl+C para parar\n')
            while True:
                run(lim_max, lim_med, predict_model)
                sleep(1)  # Minimal delay for CPU
        else:
            # Original 90-second loop
            while True:
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
