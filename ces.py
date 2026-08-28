#!/usr/bin/env python
"""
Cloud Energy Saver (CES) - Manual utilities CLI.

This module provides lightweight manual operations for inspecting and
managing the environment: register hosts, view status, and turn VMs
on/off. It is intended for debugging and one-off operations.

For full experiments (wake hosts + verification + VM load generation,
predictive models, event logging, SLA tracking), use the orchestrator:

    python orchestrator.py --help
"""

import sys
from time import sleep

import registrator, status, instances


DEPRECATED_MSG = (
    "Note: '%s' is now handled by the experiment orchestrator.\n"
    "  Use instead:\n"
    "    python orchestrator.py --model <model> --lim-max <MAX> --lim-med <MED> [--num-vms N]\n"
    "  Or for a specific stage:\n"
    "    python orchestrator.py --verify-only --model <model> --lim-max <MAX> --lim-med <MED>\n"
    "    python orchestrator.py --instantiator-only --num-vms N\n"
)

valid_params = [
    '-r', '--registrator',
    '-on', '--on',
    '-off', '--off',
    '-s', '--status',
    '-sc', '--scoreboard',
    '-v', '--verifier',
    '-i', '--instantiator',
    '-o', '--orchestrator',
]

help_msg = '''
#####  Cloud Energy Saver (CES) #####
Host state manager for OpenStack Cloud Computing environments that allows for power management experiments

Syntax:
	./ces [-option] [PARAMS]

Manual utilities:
	-r,   --registrator                  identifies and registers hosts
	-s,   --status                        shows information about Compute hosts (refreshes every 10s)
	-sc,  --scoreboard [N]                shows the top N LSTM models per host (default 5, refreshes every 10s)
	-on,  --on [QT]                       starts a quantity [QT] of instances
	-off, --off [QT]                      shuts down quantity [QT] of instances

Moved to the orchestrator (python orchestrator.py --help):
	-v,   --verifier [MAX] [MED] [MODEL]  -> python orchestrator.py --verify-only --model MODEL --lim-max MAX --lim-med MED
	-i,   --instantiator [QT]             -> python orchestrator.py --instantiator-only --num-vms QT
	-o,   --orchestrator                  -> python orchestrator.py
'''


def main():
    try:
        arg1 = sys.argv[1]
    except IndexError:
        print(help_msg)
        return

    if arg1 not in valid_params:
        print(help_msg)
        return

    # --- Manual utilities -------------------------------------------------
    if arg1 in ('--registrator', '-r'):
        registrator.run()

    elif arg1 in ('--status', '-s'):
        while True:
            try:
                hosts = status.get()
                if len(hosts) < 1:
                    print("There are no registered Compute hosts!\nRun 'python ces.py -r' to register them")
                else:
                    print("[Compute Hosts Status]\n")
                    for host in hosts:
                        print('%s [%s]' % (host['hostname'], host['state']))
                        print('RAM: {} %'.format(host['ram']))
                        try:
                            print('VMs: %s\n' % host['vms'])
                        except:
                            pass
            except:
                pass
            sleep(10)

    elif arg1 in ('--scoreboard', '-sc'):
        import scoreboard
        top_n = 5
        if len(sys.argv) > 2:
            try:
                top_n = int(sys.argv[2])
            except ValueError:
                top_n = 5
        while True:
            try:
                data = scoreboard.get(top_n=top_n)
                totals = scoreboard.total_models()
                print('[Scoreboard - LSTM multivariado]\n')
                if not data:
                    print('Nenhum modelo/placar encontrado '
                          '(o orchestrator no modo lstm ainda não pontuou).')
                for host, entries in data.items():
                    total = totals.get(host, len(entries))
                    print('--- %s (TOP %d de %d modelos) ---' % (host, len(entries), total))
                    for i, e in enumerate(entries):
                        mark = '  <-- winner' if i == 0 else ''
                        recent = e.get('recent_err')
                        r = '-' if recent is None else '%.2f' % recent
                        print('  RMSE %-8.3f  mean %-6.2f  n %-4d  recent %-6s  eff %-8.3f  %s%s' % (
                            e['rmse'], e['mean_err'], e['n'], r, e['effective'], e['filename'], mark))
                    print('')
            except Exception as ex:
                print('Erro ao ler o placar: %s' % ex)
            sleep(10)

    elif arg1 in ('--on', '-on'):
        if len(sys.argv) > 2:
            instances.on(int(sys.argv[2]))
        else:
            print('Enter a quantity of VMs to initiate\nEx: python ces.py -on 5')

    elif arg1 in ('--off', '-off'):
        if len(sys.argv) > 2:
            instances.off(int(sys.argv[2]))
        else:
            print('Enter a quantity of VMs to shut down\nEx: python ces.py -off 5')

    # --- Deprecated: moved to orchestrator --------------------------------
    elif arg1 in ('--verifier', '-v'):
        print(DEPRECATED_MSG % arg1)

    elif arg1 in ('--instantiator', '-i', '-auto'):
        print(DEPRECATED_MSG % arg1)

    elif arg1 in ('--orchestrator', '-o'):
        import orchestrator
        orchestrator.main()


if __name__ == '__main__':
    main()
