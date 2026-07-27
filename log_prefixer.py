#!/usr/bin/env python
#coding: utf-8
"""Prefixa cada linha do stdin com [YYYY-MM-DD HH:MM:SS] e grava no stdout.
Trata \\r (countdowns) como separador de linha -> cada valor vira uma linha timestampada.
Le byte a byte e faz flush a cada linha => tempo real para `tail -f`.
Uso: python -u orchestrator.py ... | python log_prefixer.py > experiment_run_YYYYMMDD_HHMMSS.log
"""
import sys
import time


def ts():
    return time.strftime('%Y-%m-%d %H:%M:%S')


def main():
    pending = ''
    while True:
        ch = sys.stdin.read(1)
        if ch == '':
            if pending:
                sys.stdout.write('[%s] %s\n' % (ts(), pending))
                sys.stdout.flush()
            break
        if ch in '\r\n':
            sys.stdout.write('[%s] %s\n' % (ts(), pending))
            sys.stdout.flush()
            pending = ''
        else:
            pending += ch


if __name__ == '__main__':
    main()
