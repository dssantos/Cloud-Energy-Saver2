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
import config
from event_logger import Event

# Load environment variables for email alerts
load_dotenv()

lstm_manager = None
experiment_start_time = None

# Emergency wake cooldown tracking
# Format: {hostname: timestamp_of_wake}
emergency_wake_cooldowns = {}
EMERGENCY_COOLDOWN_SECONDS = 300  # 5 minutes cooldown after emergency wake

# Recent-shutdown cooldown tracking (anti-flapping: don't re-wake a host right after shutting it down)
# Format: {hostname: timestamp_of_shutdown}
recent_shutdown_cooldowns = {}

# SLA violation: per-host transition tracking (edge detection across run() cycles).
# False->True per host -> one event per episode per host.
sla_violating_hosts = set()

# Previous host state tracking: {hostname: (state, vms)}.  Used to detect unexpected
# up->down transitions (crashes / hangs) that were NOT triggered by a verifier shutdown.
# If a host was 'up' with VMs>0 and goes down, it generates an SLA host_down_unexpected.
prev_host_state = {}

# Stuck-host reset cooldowns: {hostname: timestamp_of_last_reset}. Prevents repeated
# VBoxManage resets of the same host within STUCK_RESET_COOLDOWN_S.
host_reset_cooldowns = {}


def is_in_reset_cooldown(hostname):
    """True se o host foi resetado há menos de STUCK_RESET_COOLDOWN_S (não re-resetar)."""
    if hostname not in host_reset_cooldowns:
        return False
    elapsed = time.time() - host_reset_cooldowns[hostname]
    if elapsed > config.STUCK_RESET_COOLDOWN_S:
        del host_reset_cooldowns[hostname]
        return False
    return True


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


def is_in_shutdown_cooldown(hostname):
    """Check if host is in recent-shutdown cooldown (anti-flapping: avoid re-waking right after a shutdown)."""
    if hostname not in recent_shutdown_cooldowns:
        return False
    elapsed = time.time() - recent_shutdown_cooldowns[hostname]
    if elapsed > config.SHUTDOWN_COOLDOWN_SECONDS:
        del recent_shutdown_cooldowns[hostname]
        return False
    return True


def select_wake_host(offline_list):
    """Pick an offline host to wake, preferring those NOT in shutdown cooldown (anti-flapping).
    offline_list is sorted ascending by host number (wake lower numbers first).
    Fallback: if every offline host is in cooldown, wake the one shut down LONGEST ago
    (earliest shutdown timestamp) so a genuine emergency is never fully blocked."""
    candidates = [h for h in offline_list if not is_in_shutdown_cooldown(h)]
    if candidates:
        return candidates[0]
    # All offline are in shutdown cooldown. Wake the one shut down longest ago, but ONLY if it
    # is past the anti-flap window (SHUTDOWN_FLAP_BLOCK_S); otherwise return None so a genuine
    # emergency waits instead of immediately re-waking the host it just shut down (shut->wake flap).
    now = time.time()
    past_antiflap = [h for h in offline_list
                     if (now - recent_shutdown_cooldowns[h]) >= config.SHUTDOWN_FLAP_BLOCK_S]
    if past_antiflap:
        return min(past_antiflap, key=lambda h: recent_shutdown_cooldowns[h])
    return None


def shutdown_host(host):
    """Shutdown a host and record the time (feeds the anti-flapping shutdown cooldown)."""
    changestate.shutdown(host)
    recent_shutdown_cooldowns[host] = time.time()
    # Mark as intentionally shut down so prev_host_state doesn't flag it as unexpected
    prev_host_state[host] = ('down', 0)


def classify_hosts(hosts, registered):
    """
    Classify hosts into running / idle / offline lists (sorted).
      - running: up AND has VMs
      - idle:    up AND no VMs
      - offline: down AND registered
    Reused by run() and by the cluster_workload logger (single source of truth).
    """
    running = []
    idle = []
    offline = []
    for host in hosts:
        if host['state'] == 'up':
            if host['vms'] > 0:
                running.append(host['hostname'])
            else:
                idle.append(host['hostname'])
        else:
            if host['hostname'] in registered:
                offline.append(host['hostname'])
    # idle: reverse order (compute3, compute2, ...) - shutdown higher numbers first
    idle.sort(key=lambda x: int(''.join(filter(str.isdigit, x)) or 0), reverse=True)
    # offline: normal order (compute1, compute2, ...) - wake lower numbers first
    offline.sort(key=lambda x: int(''.join(filter(str.isdigit, x)) or 0))
    return running, idle, offline


def calculate_ram_average(hosts_data, lim_max, predict_model='default'):
    """
    Compute RAM averages for decision AND analysis.

    Returns: (avg_ram, overloaded, normal, avg_predicted, avg_actual)
      - avg_ram:      mean of NORMAL hosts' ram_val (decision metric; prediction-based
                      in lstm/naive/arima, live otherwise). Used for the shutdown lim_med
                      comparison. Unchanged from original behavior.
      - overloaded:   hostnames with ACTUAL ram > lim_max (installed overload).
      - normal:       hostnames with ACTUAL ram <= lim_max.
      - avg_predicted: mean of LSTM predictions of ALL up hosts (None when predict_model != 'lstm').
      - avg_actual:   mean of LIVE RAM of ALL up hosts (real cluster load, for analysis/CSV).

    Note: overloaded/normal classify by ACTUAL ram (not the prediction) so that the
    emergency branch means *installed* overload (per plan). Otherwise, in lstm mode,
    high predictions would set normal==0 and shadow the lstm_predictive wake branch.
    ram_val (the prediction) is still used for avg_ram (the decision metric).
    """
    ram_values = []
    overloaded = []
    normal = []
    actual_values = []
    predicted_values = []

    for host in hosts_data:
        if host['state'] == 'up':
            actual = host['ram']
            actual_values.append(actual)  # live reading (real load)
            ram_val = actual
            if predict_model == 'lstm':
                ram_val = predict.lstm(hostname=host['hostname'], steps_ahead=config.STEPS_AHEAD)
                predicted_values.append(ram_val)  # reuse the same prediction call
            elif predict_model == 'naive':
                ram_val = predict.naive(host['hostname'])
            elif predict_model == 'arima':
                ram_val = predict.arima(host['hostname'])

            # Classify by ACTUAL ram -> emergency = installed overload (not predicted).
            if actual > lim_max:
                overloaded.append(host['hostname'])
            else:
                normal.append(host['hostname'])
                ram_values.append(ram_val)  # decision metric (prediction where applicable)

    avg_ram = sum(ram_values) / len(ram_values) if ram_values else 0
    avg_actual = sum(actual_values) / len(actual_values) if actual_values else 0.0
    avg_predicted = sum(predicted_values) / len(predicted_values) if predicted_values else None
    return avg_ram, overloaded, normal, avg_predicted, avg_actual


def detect_sla_violations(hosts, lim_max):
    """SLA #1 (ram_over_threshold): host RAM > lim_max * (1 + SLA_RAM_MARGIN_PCT/100).

    Returns: dict hostname -> ('ram_over_threshold', ram) for currently violating hosts.
    SLA #4 (host_down_unexpected) is handled separately in run() via prev_host_state.
    """
    threshold = lim_max * (1 + config.SLA_RAM_MARGIN_PCT / 100.0)
    violating = {}
    for host in hosts:
        h = host['hostname']
        if host['state'] == 'up' and host['ram'] > threshold:
            violating[h] = ('ram_over_threshold', host['ram'])
    return violating


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
    global sla_violating_hosts

    hosts = status.get()

    try:
        file = open("registered.txt", "r+")
        registered = file.read()
        registered = ast.literal_eval(registered)
    except:
        print('É preciso registrar os hosts do ambiente')
        registered = []

    # Detect unexpected up->down transitions (crash / hang, not verifier-commanded shutdown).
    # Hosts that were up with VMs>0 and went down unexpectedly generate SLA host_down_unexpected.
    global prev_host_state
    unexpected_downs = {}
    for host in hosts:
        h = host['hostname']
        prev = prev_host_state.get(h, ('unknown', 0))
        curr_state = host['state']
        curr_vms = host.get('vms', 0)
        prev_state, prev_vms = prev if isinstance(prev, tuple) else (prev, 0)
        if prev_state == 'up' and curr_state != 'up':
            print(f'[STATE] {h} up->down (INESPERADO): ram={host.get("ram",0):.1f}% vms={curr_vms} (antes: {prev_vms}) — possível crash/travamento por sobrecarga.')
            if prev_vms > 0:
                unexpected_downs[h] = ('host_down_unexpected', 0.0)
        prev_host_state[h] = (curr_state, curr_vms)

    # Stuck-host auto-reset: host com vms > threshold E (down OU ram==0) -> reset VBoxManage
    # (uma vez por cooldown, em background thread para não bloquear o ciclo)
    for host in hosts:
        h = host['hostname']
        vms = host.get('vms', 0) or 0
        if vms > config.STUCK_HOST_VM_THRESHOLD and (host['state'] != 'up' or host.get('ram', 0) == 0):
            if is_in_reset_cooldown(h):
                continue
            print(f'[STUCK] {h} travado (vms={vms} state={host["state"]} ram={host.get("ram",0)}) -> reset forçado em background.')
            host_reset_cooldowns[h] = time.time()
            threading.Thread(target=changestate.force_reset, args=(h,), daemon=True).start()

    running, idle, offline = classify_hosts(hosts, registered)

    # ram_avg: decision metric (normal hosts only)
    # avg_actual: real load (all active hosts) -> logged to cluster CSV and events
    # avg_predicted: LSTM predictions mean (None when predict_model != 'lstm')
    ram_avg, overloaded, normal, avg_predicted, avg_actual = calculate_ram_average(hosts, lim_max, predict_model)

    print('ativos: ' + str(running))
    print('ociosos: ' + str(idle))
    print('offline: ' + str(offline))
    print(f'sobrecarregados: {overloaded}')
    print(f'normais: {normal}')
    print('média de ram (decisão): %s' % ram_avg)
    print(f'média real (avg_actual): {avg_actual:.1f}%')

    # SLA: per-host, transition-based (False->True per host). Two conditions:
    # 1) host RAM > lim_max * (1 + SLA_RAM_MARGIN_PCT/100)  --  ram_over_threshold
    # 4) host was up with VMs>0 last cycle and is now down without a verifier shutdown  --  host_down_unexpected
    current_violating = detect_sla_violations(hosts, lim_max)
    current_violating.update(unexpected_downs)
    new_violations = {h: v for h, v in current_violating.items() if h not in sla_violating_hosts}
    for h, (reason, ram) in new_violations.items():
        sla_thr = lim_max * (1 + config.SLA_RAM_MARGIN_PCT / 100.0)
        print(f'SLA VIOLAÇÃO [{reason}]: host={h} ram={ram:.1f}% (threshold={sla_thr:.1f}%)')
        event_logger.logger.log(Event(
            timestamp=datetime.now().isoformat(),
            event_type='sla_violation',
            hostname=h,
            trigger_type=reason,
            ram_avg=ram,
            lim_max=lim_max,
            lim_med=lim_med,
            running_hosts=len(running),
            idle_hosts=len(idle),
            offline_hosts=len(offline),
            predicted_ram=avg_predicted,
            actual_ram=ram
        ))
    sla_violating_hosts = set(current_violating.keys())

## Logic of the management of the hosts to be turned on and off

    # EMERGENCY: all normal hosts overloaded and offline hosts available
    if len(overloaded) > 0 and len(normal) == 0 and len(offline) > 0:
        emergency_host = select_wake_host(offline)
        if emergency_host is None:
            print('EMERGÊNCIA: offline em anti-flap (recém-desligados) — aguardando ao invés de re-acordar (SLA já registrado).')
        else:
            print(f'EMERGÊNCIA: Todos os hosts normais sobrecarregados! Acordando {emergency_host}...')
            event_logger.logger.log(Event(
                timestamp=datetime.now().isoformat(),
                event_type='wake',
                hostname=emergency_host,
                trigger_type=f'{predict_model}_emergency',
                ram_avg=avg_actual,
                lim_max=lim_max,
                lim_med=lim_med,
                running_hosts=len(running),
                idle_hosts=len(idle),
                offline_hosts=len(offline),
                predicted_ram=avg_predicted,
                actual_ram=avg_actual
            ))
            # Set cooldown to prevent immediate shutdown
            emergency_wake_cooldowns[emergency_host] = time.time()
            changestate.wake(emergency_host)

    # PREDICTIVE (lstm only): prediction > lim_max but real load still <= lim_max -> wake early
    elif (predict_model == 'lstm' and avg_predicted is not None
          and avg_predicted > lim_max and avg_actual <= lim_max
          and len(idle) == 0 and len(offline) > 0):
        predictive_host = select_wake_host(offline)
        if predictive_host is None:
            print('PREDITIVO: offline em anti-flap (recém-desligados) — aguardando.')
        else:
            print(f'PREDITIVO: previsão LSTM {avg_predicted:.1f}% > {lim_max}% (real {avg_actual:.1f}%). Acordando {predictive_host}...')
            event_logger.logger.log(Event(
                timestamp=datetime.now().isoformat(),
                event_type='wake',
                hostname=predictive_host,
                trigger_type='lstm_predictive',
                ram_avg=avg_actual,
                lim_max=lim_max,
                lim_med=lim_med,
                running_hosts=len(running),
                idle_hosts=len(idle),
                offline_hosts=len(offline),
                predicted_ram=avg_predicted,
                actual_ram=avg_actual
            ))
            changestate.wake(predictive_host)

    # REACTIVE: real load > lim_max, no idle buffer, offline available -> wake (default mode wakes here)
    elif avg_actual > lim_max and len(idle) == 0 and len(offline) > 0:
        reactive_host = select_wake_host(offline)
        if reactive_host is None:
            print('REATIVO: offline em anti-flap (recém-desligados) — aguardando.')
        else:
            print(f'REATIVO: carga real {avg_actual:.1f}% > {lim_max}%. Acordando {reactive_host}...')
            event_logger.logger.log(Event(
                timestamp=datetime.now().isoformat(),
                event_type='wake',
                hostname=reactive_host,
                trigger_type='reactive',
                ram_avg=avg_actual,
                lim_max=lim_max,
                lim_med=lim_med,
                running_hosts=len(running),
                idle_hosts=len(idle),
                offline_hosts=len(offline),
                predicted_ram=avg_predicted,
                actual_ram=avg_actual
            ))
            changestate.wake(reactive_host)

    # HIGH LOAD but cannot add capacity: keep idle hosts to absorb load (never shut down under high load)
    elif avg_actual > lim_max:
        if len(idle) > 0:
            print(f'Carga alta ({avg_actual:.1f}%) com idle disponível para absorver — mantendo hosts idle.')
        else:
            print(f'Carga alta ({avg_actual:.1f}%) sem hosts offline para acordar — limite de capacidade.')
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
                    shutdown_host(host)
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
                        shutdown_host(host)		# shut down all idle hosts
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
                        shutdown_host(host)

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
