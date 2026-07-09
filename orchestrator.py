#coding: utf-8
"""
CES Experiment Orchestrator

Manages complete experiments from wake hosts to VM instantiation.
Provides real-time progress display and graceful cancellation.
"""

import argparse
import time
import sys
from datetime import datetime
import json
import ast

import status
import changestate
import instances
import verifier
import registrator


class ExperimentOrchestrator:
	"""Gerencia experimento completo do CES."""

	def __init__(self, lim_max=70, lim_med=30, predict_model='default',
				 num_vms=27, experiment_duration_hours=18, wake_only=False, verify_only=False, instantiator_only=False):
		self.lim_max = lim_max
		self.lim_med = lim_med
		self.predict_model = predict_model
		self.num_vms = num_vms
		self.experiment_duration = experiment_duration_hours * 3600
		self.wake_only = wake_only
		self.verify_only = verify_only
		self.instantiator_only = instantiator_only
		self.cancelled = False
		self.verification_active = False
		self.results = {
			'start_time': None,
			'end_time': None,
			'hosts_waked': [],
			'hosts_registered': [],
			'vms_created': [],
			'sla_violations': 0,
			'final_status': {}
		}

	def print_progress(self, step, total, message):
		"""Display progress bar."""
		percent = step / total * 100
		bar_length = 40
		filled = int(bar_length * step / total)
		bar = '█' * filled + '░' * (bar_length - filled)
		print(f'\r[{bar}] {percent:.1f}% - {message}', end='', flush=True)

	def wake_all_hosts(self):
		"""Wake controller and all registered hosts."""
		print('\n[1/6] Acordando hosts...')

		# Acordar controller primeiro
		print('\n   Acordando controller...')
		changestate.wake('controller')
		self.results['hosts_waked'].append('controller')
		time.sleep(5)

		# Depois acorda computes registrados
		try:
			with open("registered.txt", "r") as f:
				hosts = f.read().strip()
		except:
			print('   ! Arquivo registered.txt não encontrado')
			# Continuar mesmo sem computes

		# Parse the registered list
		try:
			registered = ast.literal_eval(hosts) if hosts else []
		except:
			# Try simple comma-separated format
			registered = [h.strip() for h in hosts.split(',') if h.strip()]

		for hostname in registered:
			if self.cancelled:
				break
			print(f'\n   Acordando {hostname}...')
			changestate.wake(hostname)
			self.results['hosts_waked'].append(hostname)
			time.sleep(3)  # Wait between wakes

		return self.results['hosts_waked']

	def wait_hosts_ready(self, timeout=300):
		"""Wait for controller and all computes to reach UP state."""
		print('\n[2/6] Aguardando hosts ficarem prontos...')
		start = time.time()

		while time.time() - start < timeout:
			if self.cancelled:
				break

			all_up = True
			computes_up = 0

			try:
				# Verificar via OpenStack se os computes estão UP
				import requests, header
				r = requests.get('http://controller:8774/v2.1/os-hypervisors',
							   headers=header.get(), timeout=5)
				hypervisors = __import__('json', fromlist=['loads']).loads(r.content)['hypervisors']

				for hv in hypervisors:
					hostname = hv['hypervisor_hostname']
					if hostname in self.results['hosts_waked'] or 'compute' in hostname.lower():
						if hv['state'] == 'up':
							computes_up += 1
						else:
							all_up = False

				# Verificar se temos pelo menos 3 computes UP
				if computes_up >= 3:
					all_up = True

			except Exception as e:
				# OpenStack ainda não está pronto, continuar aguardando
				all_up = False

			if all_up:
				print(f'\n   ✓ Todos hosts prontos! ({computes_up}/3 computes UP)')
				return True

			# Show countdown
			remaining = int(timeout - (time.time() - start))
			print(f'\r   Aguardando... {remaining//60}m {remaining%60}s ({computes_up}/3 UP)  ', end='', flush=True)
			time.sleep(5)

		print('\n   ✗ Timeout aguardando hosts')
		return False

	def register_hosts(self):
		"""Ensure hosts are registered - executes registrator if needed."""
		print('\n[3/6] Verificando registro de hosts...')
		try:
			with open("registered.txt", "r") as f:
				registered = f.read().strip()
				if registered and registered not in ['[]', '']:
					print(f'   ✓ Hosts já registrados: {registered}')
					return ast.literal_eval(registered)
		except:
			pass

		# Se não há hosts registrados ou arquivo está vazio, executar registrator
		print('   ! Nenhum host registrado, executando registrator...')
		registrator.run()

		# Verificar novamente após registrator
		try:
			with open("registered.txt", "r") as f:
				registered = f.read().strip()
				print(f'   ✓ Hosts registrados: {registered}')
				return ast.literal_eval(registered)
		except:
			print('   ! Erro ao registrar hosts')
			return []

	def run_verification_only(self):
		"""Run verifier in continuous mode without instantiator."""
		print('\n[VERIFY-ONLY] Iniciando verificação contínua...')
		print(f'   Limite MAX: {self.lim_max}%')
		print(f'   Limite MED: {self.lim_med}%')
		print(f'   Modelo: {self.predict_model}')
		print('   Pressione Ctrl+C para parar\n')

		try:
			verifier.start(self.lim_max, self.lim_med, self.predict_model, continuous=True)
		except KeyboardInterrupt:
			print('\n\n! Verificação interrompida')
			self.cancelled = True

	def run_instantiator_only(self):
		"""Run instantiator only - create/delete VMs in loop."""
		print('\n[INSTANTIATOR-ONLY] Criando e deletando VMs em loop...')
		print(f'   Número de VMs por ciclo: {self.num_vms}')
		print('   Pressione Ctrl+C para parar\n')

		try:
			self.start_instantiator()
		except KeyboardInterrupt:
			print('\n\n! Instantiator interrompido')
			self.cancelled = True
		finally:
			self.save_final_status()

	def start_verification(self):
		"""Start verification loop in background thread."""
		print('\n[4/6] Iniciando verificação em background...')
		import threading
		import event_logger
		from datetime import datetime

		# Initialize event logger with model-specific filename
		event_file = f'events_{self.predict_model}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
		event_logger.logger = event_logger.EventLogger(event_file)
		print(f'   Event logging: {event_file}')

		# Start workload collection for registered hosts
		try:
			with open("registered.txt", "r") as file:
				import ast
				registered = ast.literal_eval(file.read())
			print(f'   Iniciando workload collection para {len(registered)} hosts...')
			for hostname in registered:
				import workload
				threading.Thread(target=workload.save, args=[hostname], daemon=True).start()
		except:
			print('   ! Nenhum host registrado para workload collection')

		# Initialize LSTM training for LSTM model
		if self.predict_model == 'lstm':
			try:
				import predict
				print(f'   Iniciando LSTM training para {len(registered)} hosts...')
				for hostname in registered:
					predict.lstm_manager.start_training(hostname)
				print(f'   ✓ LSTM training iniciado')
			except Exception as e:
				print(f'   ! Erro ao iniciar LSTM training: {e}')

		def verification_loop():
			self.verification_active = True
			while self.verification_active and not self.cancelled:
				try:
					verifier.run(self.lim_max, self.lim_med, self.predict_model)
					time.sleep(10)  # Check every 10 seconds
				except Exception as e:
					print(f'\n[Verifier Error] {e}')

		thread = threading.Thread(target=verification_loop, daemon=True)
		thread.start()
		print('   ✓ Verifier iniciado em background')

	def start_instantiator(self):
		"""Start VM instantiation - continuous create/delete loop like instances.py."""
		print('\n[5/6] Iniciando instantiator (loop create/delete)...')
		print(f'   Criando {self.num_vms} VMs, depois deletando e repetindo...')

		cycle = 0
		try:
			while not self.cancelled:
				cycle += 1
				print(f'\n   === Ciclo {cycle} ===')

				# Obter VMs existentes e determinar próximo número
				try:
					r = __import__('requests').get('http://controller:8774/v2.1/servers',
											   headers=__import__('header').get())
					vm_list_data = __import__('json', fromlist=['loads']).loads(r.content)
					existing_vms = [vm['name'] for vm in vm_list_data.get('servers', [])]
				except:
					existing_vms = []

				start_pos = len(existing_vms) + 1

				# Criar todas VMs
				print(f'\n   Criando {self.num_vms} VMs (vm-{start_pos} a vm-{start_pos + self.num_vms - 1})...')
				created_vms = []

				for i in range(self.num_vms):
					if self.cancelled:
						break

					vm_pos = start_pos + i
					vm_name = f'vm-{vm_pos}'

					# Countdown 60s antes de cada criação
					for countdown in range(60, -1, -1):
						if self.cancelled:
							break
						sys.stdout.write(f'Liga em: {countdown:3d}\r')
						sys.stdout.flush()
						time.sleep(1)

					if self.cancelled:
						break

					print(f'ligando {vm_name}')
					result = instances.create_instance_with_sla_check(vm_name)
					self.results['vms_created'].append(result)
					created_vms.append(vm_name)

					if result.get('sla_violation'):
						self.results['sla_violations'] += 1

					self.print_progress(i + 1, self.num_vms, f'VMs criadas: {i+1}/{self.num_vms}')

				if self.cancelled:
					break

				# Pausa antes de deletar
				print(f'\n   Aguardando 30s antes de deletar...')
				time.sleep(30)

				# Deletar VMs em ordem reversa
				print(f'\n   Deletando {self.num_vms} VMs...')
				for i, vm_name in enumerate(reversed(created_vms)):
					if self.cancelled:
						break

					# Countdown 60s antes de cada deleção
					for countdown in range(60, -1, -1):
						if self.cancelled:
							break
						sys.stdout.write(f'Desliga em: {countdown:3d}\r')
						sys.stdout.flush()
						time.sleep(1)

					if self.cancelled:
						break

					print(f'desligando {vm_name}   ')
					instances.off(1)  # Delete last VM

					self.print_progress(i + 1, self.num_vms, f'VMs deletadas: {i+1}/{self.num_vms}')

				# Pausa antes do próximo ciclo
				print(f'\n   Aguardando 30s antes do próximo ciclo...')
				time.sleep(30)

		except KeyboardInterrupt:
			print('\n\n! Instantiator interrompido')
			self.cancelled = True

	def monitor_progress(self):
		"""Monitor and display progress."""
		print('\n[6/6] Monitorando experimento...')
		print('   Pressione Ctrl+C para cancelar (salva status final)')

		start_time = time.time()
		try:
			while time.time() - start_time < self.experiment_duration:
				if self.cancelled:
					break

				elapsed = time.time() - start_time
				remaining = self.experiment_duration - elapsed
				hours = int(remaining // 3600)
				mins = int((remaining % 3600) // 60)

				# Display status
				print(f'\r   Tempo restante: {hours}h {mins}m | VMs: {len(self.results["vms_created"])} | SLA: {self.results["sla_violations"]}',
					  end='', flush=True)
				time.sleep(10)
		except KeyboardInterrupt:
			print('\n\n! Cancelamento solicitado...')
			self.cancelled = True

	def save_final_status(self):
		"""Save final experiment status."""
		print('\n\n[FINAL] Salvando status final...')

		self.results['end_time'] = datetime.now().isoformat()
		self.verification_active = False

		# Collect current host status
		hosts_status = status.get()
		self.results['final_status'] = {
			'hosts': hosts_status,
			'total_vms': len(self.results['vms_created']),
			'successful_allocations': sum(1 for v in self.results['vms_created'] if v.get('allocated')),
			'sla_violations': self.results['sla_violations']
		}

		# Save to JSON
		filename = f'experiment_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
		with open(filename, 'w') as f:
			json.dump(self.results, f, indent=2)

		print(f'   ✓ Status salvo em {filename}')

	def run(self):
		"""Execute complete experiment."""
		self.results['start_time'] = datetime.now().isoformat()

		# Instantiator-only mode: Skip wake, register, verification
		if self.instantiator_only:
			print('\n=== Modo Instantiator-Only: Executa apenas criação/deleção de VMs ===')
			self.run_instantiator_only()
			return

		# Verify-only mode: Skip wake, register, instantiator
		if self.verify_only:
			print('\n=== Modo Verify-Only: Executa apenas verificação ===')
			self.run_verification_only()
			return

		try:
			self.wake_all_hosts()
			if not self.cancelled:
				self.wait_hosts_ready()
			if not self.cancelled:
				self.register_hosts()

			if self.wake_only:
				print('\n=== Modo Wake-Only: encerrando após wake e registro ===')
				return

			if not self.cancelled:
				self.start_verification()
			if not self.cancelled:
				self.start_instantiator()
			if not self.cancelled:
				self.monitor_progress()
		except Exception as e:
			print(f'\n✗ Erro: {e}')
			import traceback
			traceback.print_exc()
		finally:
			if not self.wake_only and not self.verify_only and not self.instantiator_only:
				self.save_final_status()


def main():
	parser = argparse.ArgumentParser(description='CES Experiment Orchestrator')
	parser.add_argument('--lim-max', type=float, default=70, help='Limite máximo de RAM (%)')
	parser.add_argument('--lim-med', type=float, default=30, help='Limite médio de RAM (%)')
	parser.add_argument('--model', default='default', choices=['default', 'naive', 'arima', 'lstm'])
	parser.add_argument('--num-vms', type=int, default=27, help='Número de VMs para instanciar')
	parser.add_argument('--duration', type=int, default=18, help='Duração do experimento (horas)')
	parser.add_argument('--config', help='Arquivo de configuração JSON')
	parser.add_argument('--wake-only', action='store_true', help='Executar apenas wake dos hosts')
	parser.add_argument('--verify-only', action='store_true', help='Executar apenas verificação contínua')
	parser.add_argument('--instantiator-only', action='store_true', help='Executar apenas criação/deleção de VMs')

	args = parser.parse_args()

	orchestrator = ExperimentOrchestrator(
		lim_max=args.lim_max,
		lim_med=args.lim_med,
		predict_model=args.model,
		num_vms=args.num_vms,
		experiment_duration_hours=args.duration,
		wake_only=args.wake_only,
		verify_only=args.verify_only,
		instantiator_only=args.instantiator_only
	)

	print('='*60)
	print('CES EXPERIMENT ORCHESTRATOR')
	print('='*60)
	print(f'Parâmetros:')
	print(f'  Limite MAX: {args.lim_max}%')
	print(f'  Limite MED: {args.lim_med}%')
	print(f'  Modelo: {args.model}')
	print(f'  VMs: {args.num_vms}')
	print(f'  Duração: {args.duration}h')
	if args.wake_only:
		print(f'  Modo: WAKE-ONLY')
	if args.verify_only:
		print(f'  Modo: VERIFY-ONLY')
	if args.instantiator_only:
		print(f'  Modo: INSTANTIATOR-ONLY')
	print('='*60)

	orchestrator.run()


if __name__ == '__main__':
	main()
