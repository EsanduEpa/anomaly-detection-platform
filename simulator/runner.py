"""
SIMULATOR RUNNER
================

Drives the generator, POSTs payloads to the FastAPI ingest endpoint, and writes
a GROUND TRUTH file alongside so Phase 3 can measure model accuracy honestly.

Usage
-----
    # normal run — collect training data
    python -m simulator.runner

    # collect 30 minutes then stop automatically
    python -m simulator.runner --duration 30

    # ⭐ RECOMMENDED TRAINING RUN — 60 real minutes.
    # Covers 30 simulated hours (full day AND night), yields ~3,600 samples
    # per service, and reaches all 23 scenarios including off_hours_surge.
    python -m simulator.runner --time-scale 30 --interval 1 --duration 60 \
        --ground-truth simulator/gt_train.jsonl

    # see every anomaly type this simulator can produce
    python -m simulator.runner --list-scenarios

    # force ONE scenario, to eyeball what it looks like
    python -m simulator.runner --scenario memory_leak --warmup 0

    # clean data only (no anomalies) — useful for training an autoencoder
    # on purely normal behaviour
    python -m simulator.runner --no-anomalies --duration 20

    # realistic production rate instead of training rate (few anomalies)
    python -m simulator.runner --anomaly-rate 0.002 --duration 60
"""

import argparse
import json
import signal
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import requests

from simulator import generator
from simulator.generator import all_states, configure_clock, generate_tick, sim_now
from simulator.scenarios import SCENARIOS, categories

DEFAULT_API_URL = "http://127.0.0.1:8080/api/v1/metrics"
DEFAULT_GROUND_TRUTH = Path("simulator/ground_truth.jsonl")

SERVICES = [
    {"service_name": "payment-service", "host": "prod-server-01"},
    {"service_name": "user-service",    "host": "prod-server-02"},
    {"service_name": "api-gateway",     "host": "prod-server-03"},
]

_stop = False


def _handle_sigint(signum, frame):
    global _stop
    _stop = True
    print("\n\n⏹  Stopping after this batch...\n")


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

SEVERITY_ICON = {"none": "  ", "mild": "🟡", "moderate": "🟠", "severe": "🔴"}


def print_scenario_table() -> None:
    print("\n" + "=" * 88)
    print(f"  ANOMALY CATALOG — {len(SCENARIOS)} scenarios")
    print("=" * 88)
    for category, names in categories().items():
        print(f"\n  ▸ {category.upper()}")
        for name in names:
            scenario = next(s for s in SCENARIOS if s.name == name)
            print(f"      {scenario.name:<24} {scenario.min_ticks:>3}-{scenario.max_ticks:<3} ticks   "
                  f"{scenario.description}")
    print("\n" + "=" * 88 + "\n")


def print_summary(sent: int, failed: int, labels: Counter, started: float,
                  hours_seen: set, scale: float) -> None:
    elapsed = time.time() - started
    anomalous = sum(count for key, count in labels.items() if key != "normal")
    total = sum(labels.values())

    print("\n" + "=" * 78)
    print("  RUN SUMMARY")
    print("=" * 78)
    print(f"  Real duration      : {elapsed/60:.1f} min")
    if scale > 1.0:
        print(f"  Simulated duration : {elapsed*scale/3600:.1f} hours   (time-scale ×{scale:g})")
        print(f"  Hours of day seen  : {len(hours_seen)}/24   "
              f"{'✅ full diurnal cycle' if len(hours_seen) >= 24 else '⚠️  partial coverage'}")
    print(f"  Payloads sent      : {sent}   (failed: {failed})")
    print(f"  Samples per service: ~{sent // max(1, len(SERVICES))}   ← the number ML cares about")
    print(f"  DB rows produced   : ~{sent * 7}   (7 metrics per payload)")
    if total:
        print(f"  Anomalous readings : {anomalous}  ({100*anomalous/total:.1f}%)")

    if anomalous:
        print("\n  Episodes seen per scenario")
        print("  " + "-" * 50)
        per_scenario: Counter = Counter()
        for state in all_states().values():
            per_scenario.update(state.episode_counts)
        for name, count in per_scenario.most_common():
            readings = labels.get(name, 0)
            print(f"    {name:<26} {count:>3} episodes   {readings:>5} readings")

        missing = [s.name for s in SCENARIOS if s.name not in per_scenario]
        if missing:
            print(f"\n  Not seen this run ({len(missing)}): {', '.join(missing)}")
            print("    Run longer, or force one with --scenario <name>.")

    shifted = {name: state.baseline_shift
               for name, state in all_states().items() if state.baseline_shift}
    if shifted:
        print("\n  ⚠️  Permanent baseline shifts (deployment_regression)")
        print("  " + "-" * 50)
        for name, shift in shifted.items():
            pretty = ", ".join(f"{k} ×{v:.2f}" for k, v in shift.items())
            print(f"    {name:<26} {pretty}")
        print("    These services now have a NEW normal — this is concept drift,")
        print("    which is exactly what Phase 7 drift detection must catch.")

    print("=" * 78 + "\n")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    global _stop
    signal.signal(signal.SIGINT, _handle_sigint)

    generator.WINDOW_WARMUP_TICKS = args.warmup
    configure_clock(scale=args.time_scale, start_hour=args.start_hour)

    ground_truth_path = Path(args.ground_truth)
    ground_truth_path.parent.mkdir(parents=True, exist_ok=True)
    ground_truth = ground_truth_path.open("a", encoding="utf-8")

    session = requests.Session()
    labels: Counter = Counter()
    hours_seen: set = set()
    sent = failed = 0
    started = time.time()
    deadline = started + args.duration * 60 if args.duration else None

    print("\n🚀 Simulator running")
    print(f"   API              : {args.api_url}")
    print(f"   Services         : {len(SERVICES)}")
    print(f"   Interval         : {args.interval}s")
    if args.time_scale > 1.0:
        sim_hours = (args.duration * 60 * args.time_scale / 3600) if args.duration else None
        print(f"   Time scale       : ×{args.time_scale:g}  "
              f"(1 real sec = {args.time_scale:g} simulated sec)")
        if sim_hours:
            print(f"   Simulated span   : {sim_hours:.1f} hours"
                  f"{'  ✅ full day covered' if sim_hours >= 24 else '  ⚠️  under 24h'}")
        print(f"   Clock starts at  : {sim_now().strftime('%H:%M')} UTC")
    print(f"   Anomalies        : {'OFF' if args.no_anomalies else f'ON (start prob {args.anomaly_rate})'}")
    if args.scenario:
        print(f"   Forced scenario  : {args.scenario}")
    print(f"   Ground truth     : {ground_truth_path}")
    print(f"   Warmup           : {args.warmup} normal ticks per service")
    if deadline:
        print(f"   Stops after      : {args.duration} min")
    print("   CTRL+C to stop\n")

    try:
        while not _stop:
            if deadline and time.time() >= deadline:
                print("\n⏱  Requested duration reached.\n")
                break

            for service in SERVICES:
                payload, label = generate_tick(
                    service_name=service["service_name"],
                    host=service["host"],
                    inject_anomaly=not args.no_anomalies,
                    start_probability=args.anomaly_rate,
                    force_scenario=args.scenario,
                )

                labels[label["anomaly_type"]] += 1
                hours_seen.add(int(label["hour_utc"]))

                record = {
                    "timestamp":    payload["timestamp"],
                    "service_name": payload["service_name"],
                    "host":         payload["host"],
                    **label,
                }
                ground_truth.write(json.dumps(record) + "\n")

                try:
                    response = session.post(args.api_url, json=payload, timeout=5)
                    if response.status_code == 202:
                        sent += 1
                        if not args.quiet:
                            m = payload["metrics"]
                            icon = SEVERITY_ICON.get(label["severity"], "  ")
                            tag = (f"{icon} {label['anomaly_type']}"
                                   f" {label['progress']*100:3.0f}%"
                                   if label["is_anomaly"] else "   normal")
                            print(f"  {payload['service_name']:<16}"
                                  f" cpu {m['cpu_usage']:5.1f}%"
                                  f" mem {m['memory_usage']:5.1f}%"
                                  f" lat {m['request_latency_ms']:7.1f}ms"
                                  f" rps {m['requests_per_sec']:7.1f}"
                                  f" err {m['error_rate']*100:5.2f}%"
                                  f" db {m['db_connections']:3d}"
                                  f" disk {m['disk_usage']:5.1f}%"
                                  f" │ {tag}")
                    else:
                        failed += 1
                        print(f"  ❌ {service['service_name']}: {response.status_code} {response.text[:160]}")

                except requests.exceptions.ConnectionError:
                    failed += 1
                    print("  ❌ Cannot reach the API. Is uvicorn running on port 8080?")
                except requests.exceptions.Timeout:
                    failed += 1
                    print(f"  ⌛ {service['service_name']}: request timed out")

            ground_truth.flush()
            if not args.quiet:
                print("  " + "─" * 108)

            time.sleep(args.interval)

    finally:
        ground_truth.close()
        print_summary(sent, failed, labels, started, hours_seen, args.time_scale)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate realistic server metrics with labelled anomalies.")
    parser.add_argument("--api-url", default=DEFAULT_API_URL,
                        help="ingest endpoint (default: %(default)s)")
    parser.add_argument("--interval", type=float, default=5.0,
                        help="seconds between batches (default: %(default)s)")
    parser.add_argument("--duration", type=float, default=0,
                        help="stop after N minutes (0 = run forever)")
    parser.add_argument("--anomaly-rate", type=float,
                        default=generator.ANOMALY_START_PROBABILITY,
                        help="per-tick chance of starting an episode "
                             "(default: %(default)s; use 0.002 for production-realistic)")
    parser.add_argument("--no-anomalies", action="store_true",
                        help="generate clean normal data only")
    parser.add_argument("--time-scale", type=float, default=1.0,
                        help="simulated seconds per real second. 80 means "
                             "18 real minutes covers a full 24h day, so the data "
                             "includes night-time behaviour (default: 1 = real time)")
    parser.add_argument("--start-hour", type=float, default=None,
                        help="UTC hour to start the simulated day at (0-24). "
                             "Default: the current real hour.")
    parser.add_argument("--scenario", choices=[s.name for s in SCENARIOS],
                        help="force one scenario repeatedly (for inspection/testing)")
    parser.add_argument("--warmup", type=int, default=generator.WINDOW_WARMUP_TICKS,
                        help="normal-only ticks before anomalies may start "
                             "(fills the rolling window cleanly)")
    parser.add_argument("--ground-truth", default=str(DEFAULT_GROUND_TRUTH),
                        help="JSONL file for labels (default: %(default)s)")
    parser.add_argument("--quiet", action="store_true",
                        help="suppress per-reading output")
    parser.add_argument("--list-scenarios", action="store_true",
                        help="print the anomaly catalog and exit")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    if args.list_scenarios:
        print_scenario_table()
        sys.exit(0)
    run(args)
