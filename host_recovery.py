#coding: utf-8
"""
Registro de hosts que travaram (crash/hang) e coordenação de recuperação.

O verifier registra toda transição up->down inesperada (host_down_unexpected)
em crashed_hosts.json; o orchestrator, no início de cada ciclo do instantiator,
reinicia (VBoxManage poweroff + startvm) os hosts que continuam fora e limpa o
registro. Resets comandados pelo próprio orchestrator são marcados aqui para
não serem confundidos com crash pelo verifier (boot observado: <=1.8 min;
RESET_GRACE_S cobre com folga).
"""
import json
import time

REGISTRY_FILE = 'crashed_hosts.json'
RESET_GRACE_S = 600  # janela em que um reset comandado é respeitado (sem re-marcar/re-resetar)

# {hostname: {'ts', 'reason', 'ram', 'attempts', 'resetting_ts'}}
_registry = {}


def _load():
    try:
        with open(REGISTRY_FILE) as f:
            return json.load(f)
    except Exception:
        return {}


def _save():
    try:
        with open(REGISTRY_FILE, 'w') as f:
            json.dump(_registry, f, indent=2, sort_keys=True)
    except Exception as e:
        print(f'[HOST RECOVERY] erro salvando {REGISTRY_FILE}: {e}')


def _reload():
    global _registry
    _registry = _load()
    return _registry


def mark_resetting(hostname):
    """Orchestrator vai forçar reset deste host; verifier não deve registrar como crash."""
    _reload()
    entry = _registry.get(hostname, {'ts': time.time(), 'reason': 'commanded_reset', 'attempts': 0})
    entry['resetting_ts'] = time.time()
    _registry[hostname] = entry
    _save()


def mark_crashed(hostname, reason, ram=0.0):
    """Verifier: transição up->down inesperada (possível crash/travamento)."""
    _reload()
    now = time.time()
    entry = _registry.get(hostname, {'ts': now, 'reason': reason, 'attempts': 0})
    reset_ts = entry.get('resetting_ts')
    if reset_ts and now - reset_ts < RESET_GRACE_S:
        print(f'[HOST RECOVERY] {hostname}: down logo após reset comandado — não é crash.')
        return
    entry.update(ts=now, reason=reason, ram=ram)
    entry['attempts'] = entry.get('attempts', 0) + 1
    _registry[hostname] = entry
    _save()
    print(f'[HOST RECOVERY] {hostname} registrado como travado ({reason}, registro #{entry["attempts"]})')


def pending():
    """Hosts registrados como travados e fora da janela de graça de reset."""
    _reload()
    now = time.time()
    out = {}
    for h, e in _registry.items():
        reset_ts = e.get('resetting_ts')
        if reset_ts and now - reset_ts < RESET_GRACE_S:
            continue  # reset recente; aguardar boot antes de considerar travado de novo
        out[h] = e
    return out


def mark_recovered(hostname):
    """Remove do registro (host voltou sozinho ou foi reiniciado com sucesso)."""
    _reload()
    if hostname in _registry:
        del _registry[hostname]
        _save()
        print(f'[HOST RECOVERY] {hostname} removido do registro de travados')
