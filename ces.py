#!/usr/bin/env python
"""
Cloud Energy Saver (CES) - Atalho para o orchestrator.

Todas as operações (utilidades manuais e experimentos completos) estão
centralizadas no orchestrator.py; este arquivo apenas repassa os argumentos:

    python ces.py -r                     ==  python orchestrator.py --registrator
    python ces.py -s                     ==  python orchestrator.py --status
    python ces.py -sc 5                  ==  python orchestrator.py --scoreboard 5
    python ces.py -on 5                  ==  python orchestrator.py --on 5
    python ces.py -off 5                 ==  python orchestrator.py --off 5
    python ces.py -v --model lstm --lim-max 70 --lim-med 50
                                         ==  python orchestrator.py --verify-only ...
    python ces.py --model lstm --lim-max 70 --lim-med 50 --num-vms 27
                                         ==  python orchestrator.py --model lstm ...

Sem argumentos, mostra a ajuda (não dispara experimento).

Veja: python orchestrator.py --help
"""

import sys

import orchestrator


if __name__ == '__main__':
	if len(sys.argv) == 1:
		# Sem argumentos: mostra a ajuda em vez de disparar um experimento completo
		sys.argv.append('--help')
	orchestrator.main()
