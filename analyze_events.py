#coding: utf-8
"""
CES Event Analysis Module

Analyzes events JSON files to generate metrics for TCC data.
Supports:
- Single log analysis with time filtering
- Dual log comparison with common time window
"""

import json
import argparse
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from collections import defaultdict


def load_events(filename: str) -> List[Dict]:
    """Load events from JSON file."""
    with open(filename, 'r') as f:
        return json.load(f)


def filter_events_by_time(events: List[Dict], start: Optional[datetime] = None, end: Optional[datetime] = None) -> List[Dict]:
    """Filter events by time range."""
    filtered = events
    if start:
        filtered = [e for e in filtered if datetime.fromisoformat(e['timestamp']) >= start]
    if end:
        filtered = [e for e in filtered if datetime.fromisoformat(e['timestamp']) <= end]
    return filtered


def calculate_unnecessary_activations_pct(events: List[Dict]) -> float:
    """
    Calculate percentage of unnecessary activations.
    An activation is unnecessary if the host was shut down shortly after.
    Definition: wake followed by shutdown within 15 minutes.
    """
    wakes = [e for e in events if e['event_type'] == 'wake']
    shutdowns = [e for e in events if e['event_type'] == 'shutdown']

    unnecessary = 0
    for wake in wakes:
        wake_time = datetime.fromisoformat(wake['timestamp'])
        wake_hostname = wake['hostname']

        # Find next shutdown for this host
        for shutdown in shutdowns:
            if shutdown['hostname'] == wake_hostname:
                shutdown_time = datetime.fromisoformat(shutdown['timestamp'])
                if shutdown_time > wake_time:
                    diff = (shutdown_time - wake_time).total_seconds() / 60
                    if diff < 15:  # Less than 15 minutes = unnecessary
                        unnecessary += 1
                    break

    if len(wakes) == 0:
        return 0.0
    return (unnecessary / len(wakes)) * 100


def calculate_avg_late_shutdown(events: List[Dict]) -> float:
    """
    Calculate average delay in shutdown (minutes after ideal point).
    Late shutdown = time RAM was below MED threshold but host stayed on.
    """
    shutdowns = [e for e in events if e['event_type'] == 'shutdown']
    total_late = 0
    count = 0

    for shutdown in shutdowns:
        if shutdown['ram_avg'] < shutdown['lim_med']:
            # This shutdown was late
            # Estimate: could have happened 1 cycle earlier (90 seconds)
            total_late += 1.5  # minutes
            count += 1

    return total_late / count if count > 0 else 0.0


def calculate_avg_anticipation_time(events: List[Dict]) -> float:
    """
    Calculate average anticipation time for LSTM activations (minutes).
    Anticipation = time between wake and when RAM actually crossed threshold.
    """
    wakes = [e for e in events if e['event_type'] == 'wake']
    total_anticipation = 0
    count = 0

    for wake in wakes:
        if wake.get('predicted_ram') and wake.get('actual_ram'):
            # If predicted was high but actual was still low, this was anticipation
            if wake['predicted_ram'] > wake['lim_max'] and wake['actual_ram'] < wake['lim_max']:
                # Estimate anticipation based on prediction difference
                total_anticipation += 5  # Conservative estimate: 5 minutes
                count += 1

    return total_anticipation / count if count > 0 else 0.0


def calculate_total_active_hours(events: List[Dict]) -> float:
    """
    Calculate total hours hosts were active (not offline).
    Accounts for initial_state, final_state, wake, and shutdown events.
    """
    # Get experiment boundaries from initial/final state events
    initial_events = [e for e in events if e['event_type'] == 'initial_state']
    final_events = [e for e in events if e['event_type'] == 'final_state']

    if not initial_events:
        # Fallback: use first event as start, last event as end
        if events:
            exp_start = min(datetime.fromisoformat(e['timestamp']) for e in events)
            exp_end = max(datetime.fromisoformat(e['timestamp']) for e in events)
        else:
            return 0.0
    else:
        exp_start = min(datetime.fromisoformat(e['timestamp']) for e in initial_events)
        if final_events:
            exp_end = max(datetime.fromisoformat(e['timestamp']) for e in final_events)
        else:
            exp_end = max(datetime.fromisoformat(e['timestamp']) for e in events)

    # Track each host's active periods
    host_active_periods = defaultdict(list)  # hostname -> [(start, end), ...]
    initially_up = set()  # hosts that were up at experiment start
    finally_up = set()  # hosts that are still up at experiment end

    # Find initially running hosts
    for e in initial_events:
        if e.get('initial_state') == 'up' and e['hostname']:
            initially_up.add(e['hostname'])

    # Find finally running hosts
    for e in final_events:
        if e.get('final_state') == 'up' and e['hostname']:
            finally_up.add(e['hostname'])

    # Process events to build active periods
    for e in events:
        if not e['hostname']:
            continue

        hostname = e['hostname']
        timestamp = datetime.fromisoformat(e['timestamp'])
        event_type = e['event_type']

        if event_type == 'wake':
            # Host became active
            host_active_periods[hostname].append((timestamp, None))  # No end time yet
        elif event_type == 'shutdown':
            # Host became inactive
            if hostname in host_active_periods:
                periods = host_active_periods[hostname]
                if periods and periods[-1][1] is None:
                    # Close the last open period
                    periods[-1] = (periods[-1][0], timestamp)

    # Calculate total active seconds
    total_seconds = 0

    for hostname, periods in host_active_periods.items():
        for start, end in periods:
            if end is None:
                # This host was still active at end of logged events
                if hostname in finally_up:
                    # Use final state timestamp if available
                    end = exp_end
                else:
                    # No final state, skip this incomplete period
                    continue
            if start < end:
                total_seconds += (end - start).total_seconds()

    # Add time for hosts that were initially up and never shut down
    for hostname in initially_up:
        if hostname not in host_active_periods and hostname in finally_up:
            # Host was up the entire time
            total_seconds += (exp_end - exp_start).total_seconds()
        elif hostname not in host_active_periods:
            # Host was initially up but no final state - use experiment end
            total_seconds += (exp_end - exp_start).total_seconds()

    return total_seconds / 3600  # Convert to hours


def count_sla_violations(events: List[Dict]) -> int:
    """Count SLA violation events."""
    return len([e for e in events if e['event_type'] == 'sla_violation'])


def get_experiment_duration(events: List[Dict]) -> Dict:
    """Get experiment duration info."""
    if not events:
        return {'start': None, 'end': None, 'duration_hours': 0}

    start = min(datetime.fromisoformat(e['timestamp']) for e in events)
    end = max(datetime.fromisoformat(e['timestamp']) for e in events)
    duration = (end - start).total_seconds() / 3600

    return {
        'start': start.isoformat(),
        'end': end.isoformat(),
        'duration_hours': duration
    }


def analyze_log(events: List[Dict], label: str, baseline_hours: Optional[float] = None) -> Dict:
    """Analyze a single log and return metrics."""
    duration_info = get_experiment_duration(events)
    total_active_hours = calculate_total_active_hours(events)

    # If no baseline specified, use actual duration * 3 (for 3 hosts)
    if baseline_hours is None:
        baseline_hours = duration_info['duration_hours'] * 3

    reduction_pct = (baseline_hours - total_active_hours) / baseline_hours * 100 if baseline_hours > 0 else 0
    economy_kwh = (baseline_hours - total_active_hours) * 3 * 92.5 / 1000  # 3 hosts, 92.5W avg

    return {
        'label': label,
        'experiment': duration_info,
        'metrics': {
            'unnecessary_activations_pct': calculate_unnecessary_activations_pct(events),
            'avg_late_shutdown_min': calculate_avg_late_shutdown(events),
            'avg_anticipation_min': calculate_avg_anticipation_time(events),
            'total_active_hours': total_active_hours,
            'sla_violations': count_sla_violations(events),
        },
        'derived': {
            'baseline_hours': baseline_hours,
            'reduction_pct': reduction_pct,
            'economy_kwh': economy_kwh,
        }
    }


def compare_logs(log1_file: str, start1: datetime, log2_file: str, start2: datetime, window_minutes: int, baseline_hours: Optional[float] = None) -> Dict:
    """
    Compare two logs using a common time window.
    Each log starts from its respective start time and extends for window_minutes.
    """
    # Calculate end times
    end1 = start1 + timedelta(minutes=window_minutes)
    end2 = start2 + timedelta(minutes=window_minutes)

    # Load and filter both logs
    all_events1 = load_events(log1_file)
    events1 = filter_events_by_time(all_events1, start1, end1)

    all_events2 = load_events(log2_file)
    events2 = filter_events_by_time(all_events2, start2, end2)

    # Analyze both logs
    result1 = analyze_log(events1, 'Reactive (default)', baseline_hours)
    result2 = analyze_log(events2, 'LSTM', baseline_hours)

    # Calculate comparison metrics (reduction/improvement)
    m1 = result1['metrics']
    m2 = result2['metrics']
    d1 = result1['derived']
    d2 = result2['derived']

    comparison = {
        'unnecessary_activations_reduction': None,
        'late_shutdown_reduction': None,
        'active_hours_reduction': None,
        'sla_violations_reduction': None,
    }

    if m1['unnecessary_activations_pct'] > 0:
        comparison['unnecessary_activations_reduction'] = (
            (m1['unnecessary_activations_pct'] - m2['unnecessary_activations_pct']) / m1['unnecessary_activations_pct'] * 100
        ) if m2['unnecessary_activations_pct'] < m1['unnecessary_activations_pct'] else 0

    if m1['avg_late_shutdown_min'] > 0:
        comparison['late_shutdown_reduction'] = m1['avg_late_shutdown_min'] - m2['avg_late_shutdown_min']

    if d1['reduction_pct'] > 0:
        comparison['active_hours_reduction_improvement'] = d2['reduction_pct'] - d1['reduction_pct']

    comparison['sla_violations_change'] = m1['sla_violations'] - m2['sla_violations']

    return {
        'window': {
            'duration_minutes': window_minutes,
            'duration_hours': window_minutes / 60,
            'log1': {'start': start1.isoformat(), 'end': end1.isoformat(), 'file': log1_file},
            'log2': {'start': start2.isoformat(), 'end': end2.isoformat(), 'file': log2_file},
        },
        'log1': result1,
        'log2': result2,
        'comparison': comparison,
    }


def print_comparison_report(report: Dict):
    """Print comparison report in formatted table."""
    w = report['window']
    r1 = report['log1']
    r2 = report['log2']
    c = report['comparison']
    m1 = r1['metrics']
    m2 = r2['metrics']
    d1 = r1['derived']
    d2 = r2['derived']

    print(f"\n{'='*70}")
    print(f"CES Comparison Report")
    print(f"{'='*70}")
    print(f"Time Window: {w['duration_hours']:.1f} hours ({w['duration_minutes']} minutes)")
    print(f"\n{r1['label']:<30} {r2['label']}")
    print(f"File: {w['log1']['file']:<20} File: {w['log2']['file']}")
    print(f"Range: {w['log1']['start']:<20} Range: {w['log2']['start']}")

    print(f"\n{'Metric':<35} {r1['label']:<15} {r2['label']:<15} {'Difference':<15}")
    print(f"{'-'*70}")

    print(f"{'Experiment Duration (h)':<35} {r1['experiment']['duration_hours']:<15.2f} {r2['experiment']['duration_hours']:<15.2f}")

    # Unnecessary Activations
    diff_unn = m1['unnecessary_activations_pct'] - m2['unnecessary_activations_pct']
    print(f"{'Unnecessary Activations (%)':<35} {m1['unnecessary_activations_pct']:<15.1f} {m2['unnecessary_activations_pct']:<15.1f} {diff_unn:>+14.1f}%")

    # Late Shutdown
    diff_late = m1['avg_late_shutdown_min'] - m2['avg_late_shutdown_min']
    print(f"{'Avg Late Shutdown (min)':<35} {m1['avg_late_shutdown_min']:<15.1f} {m2['avg_late_shutdown_min']:<15.1f} {diff_late:>+14.1f}")

    # Anticipation
    print(f"{'Avg Anticipation Time (min)':<35} {m1['avg_anticipation_min']:<15.1f} {m2['avg_anticipation_min']:<15.1f} {m2['avg_anticipation_min']-m1['avg_anticipation_min']:>+14.1f}")

    # Active Hours
    diff_hours = m1['total_active_hours'] - m2['total_active_hours']
    print(f"{'Total Active Hours (h)':<35} {m1['total_active_hours']:<15.1f} {m2['total_active_hours']:<15.1f} {diff_hours:>+14.1f}")

    # SLA Violations
    diff_sla = m1['sla_violations'] - m2['sla_violations']
    print(f"{'SLA Violations':<35} {m1['sla_violations']:<15d} {m2['sla_violations']:<15d} {diff_sla:>+14d}")

    print(f"\n{'Derived Metrics':<35} {r1['label']:<15} {r2['label']:<15} {'Difference':<15}")
    print(f"{'-'*70}")

    # Reduction % (energy)
    diff_reduction = d2['reduction_pct'] - d1['reduction_pct']
    print(f"{'Energy Reduction (%)':<35} {d1['reduction_pct']:<15.1f} {d2['reduction_pct']:<15.1f} {diff_reduction:>+14.1f}%")

    # Economy kWh
    diff_economy = d2['economy_kwh'] - d1['economy_kwh']
    print(f"{'Energy Economy (kWh)':<35} {d1['economy_kwh']:<15.2f} {d2['economy_kwh']:<15.2f} {diff_economy:>+14.2f}")

    print(f"{'='*70}\n")


def print_single_report(report: Dict):
    """Print single log report in formatted table."""
    m = report['metrics']
    d = report['derived']
    print(f"\n{'='*60}")
    print(f"CES Analysis Report: {report['file']}")
    print(f"{'='*60}")
    print(f"Experiment Duration: {report['experiment']['duration_hours']:.2f} hours")
    print(f"Time Range: {report['experiment']['start']} to {report['experiment']['end']}")
    print(f"\n{'Metrics':<40} {'Value':>15}")
    print(f"{'-'*60}")
    print(f"{'Unnecessary Activations':<40} {m['unnecessary_activations_pct']:>14.1f}%")
    print(f"{'Avg Late Shutdown':<40} {m['avg_late_shutdown_min']:>14.1f} min")
    print(f"{'Avg Anticipation Time':<40} {m['avg_anticipation_min']:>14.1f} min")
    print(f"{'Total Active Hours':<40} {m['total_active_hours']:>14.1f} h")
    print(f"{'SLA Violations':<40} {m['sla_violations']:>14d}")
    print(f"\n{'Derived Metrics':<40} {'Value':>15}")
    print(f"{'-'*60}")
    print(f"{'Baseline Hours':<40} {d['baseline_hours']:>14.1f} h")
    print(f"{'Reduction %':<40} {d['reduction_pct']:>14.1f}%")
    print(f"{'Energy Economy':<40} {d['economy_kwh']:>14.2f} kWh")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='Analyze CES events JSON file(s)')

    # Single log mode
    parser.add_argument('--log', help='Path to events JSON file (single mode)')
    parser.add_argument('--start', help='Start time (ISO format: 2026-06-13T10:00:00)')
    parser.add_argument('--end', help='End time (ISO format: 2026-06-13T18:00:00)')

    # Comparison mode
    parser.add_argument('--log1', help='Path to first events JSON file (comparison mode)')
    parser.add_argument('--start1', help='Start time for log1 (ISO format)')
    parser.add_argument('--log2', help='Path to second events JSON file (comparison mode)')
    parser.add_argument('--start2', help='Start time for log2 (ISO format)')
    parser.add_argument('--windowtime', type=int, help='Time window in minutes for comparison (e.g., 1080 for 18h)')

    # Common options
    parser.add_argument('--baseline', type=float, help='Baseline hours for comparison (default: duration × 3)')
    parser.add_argument('--summary', action='store_true', help='Show formatted summary')
    parser.add_argument('--json', action='store_true', help='Output JSON instead of formatted text')

    args = parser.parse_args()

    baseline = args.baseline

    # Comparison mode
    if args.log1 and args.log2 and args.start1 and args.start2 and args.windowtime:
        start1 = datetime.fromisoformat(args.start1)
        start2 = datetime.fromisoformat(args.start2)
        report = compare_logs(args.log1, start1, args.log2, start2, args.windowtime, baseline)

        if args.json:
            print(json.dumps(report, indent=2))
        else:
            print_comparison_report(report)

    # Single log mode
    elif args.log:
        start = datetime.fromisoformat(args.start) if args.start else None
        end = datetime.fromisoformat(args.end) if args.end else None

        all_events = load_events(args.log)
        events = filter_events_by_time(all_events, start, end)

        duration_info = get_experiment_duration(events)
        total_active_hours = calculate_total_active_hours(events)

        if baseline is None:
            baseline = duration_info['duration_hours'] * 3

        reduction_pct = (baseline - total_active_hours) / baseline * 100 if baseline > 0 else 0
        economy_kwh = (baseline - total_active_hours) * 3 * 92.5 / 1000

        report = {
            'file': args.log,
            'time_filter': {
                'start': start.isoformat() if start else None,
                'end': end.isoformat() if end else None,
            },
            'experiment': duration_info,
            'total_events': len(events),
            'metrics': {
                'unnecessary_activations_pct': calculate_unnecessary_activations_pct(events),
                'avg_late_shutdown_min': calculate_avg_late_shutdown(events),
                'avg_anticipation_min': calculate_avg_anticipation_time(events),
                'total_active_hours': total_active_hours,
                'sla_violations': count_sla_violations(events),
            },
            'derived': {
                'baseline_hours': baseline,
                'reduction_pct': reduction_pct,
                'economy_kwh': economy_kwh,
            }
        }

        if args.json or not args.summary:
            print(json.dumps(report, indent=2))
        else:
            print_single_report(report)

    else:
        parser.print_help()
        print("\nExamples:")
        print("  Single log:")
        print("    python analyze_events.py --log events.json --summary")
        print("  Comparison:")
        print("    python analyze_events.py --log1 events_default.json --start1 2026-06-13T10:00:00 \\")
        print("                              --log2 events_lstm.json --start2 2026-06-14T10:00:00 \\")
        print("                              --windowtime 1080 --summary")


if __name__ == '__main__':
    main()
