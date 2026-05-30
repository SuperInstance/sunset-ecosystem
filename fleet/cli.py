"""fleet/cli.py — CLI entry point for sunset-ecosystem.

Cross-pollinated from cocapn-health/cli.py.  Extended with fleet-specific
commands: status checks, test running, breeding triggers, and report generation.

Usage
-----
    sunset status                    # Check fleet health
    sunset test                      # Run test suite
    sunset breed --pool 50           # Trigger breeding cycle
    sunset report --type breeding    # Generate deck report
    sunset --format json status      # JSON output
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import replace

from fleet.config import get_config, FleetConfig
from fleet.health_check import FleetHealthChecker, ServiceDef


def cmd_status(args):
    cfg = get_config()
    services = [ServiceDef(s["name"], s["host"], s["port"], s.get("path", "/status")) for s in cfg.health_services()]

    # Host override
    host = args.host or os.environ.get("SUNSET_HOST")
    if host:
        services = [replace(svc, host=host) for svc in services]

    checker = FleetHealthChecker(services)
    results = checker.check_all()
    print(checker.report(results, format=args.format))

    if args.fail:
        down = sum(1 for r in results if not r.ok)
        if down > 0:
            sys.exit(1)


def cmd_test(args):
    import subprocess
    cmd = ["python3", "-m", "pytest", "tests/", "-v"]
    if args.tb:
        cmd += [f"--tb={args.tb}"]
    if args.k:
        cmd += ["-k", args.k]
    if args.x:
        cmd += ["-x"]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


def cmd_breed(args):
    from swarm.breeder_daemon_v2 import BreederDaemonV2
    from swarm.flux_gating import FluxGatingConfig
    from fleet.config import get_config

    cfg = get_config()
    flux_cfg = FluxGatingConfig(
        weight_bounds=cfg.flux_weight_bounds,
        max_l2_norm=cfg.flux_max_l2_norm,
        max_variance=cfg.flux_max_variance,
        max_chaos=cfg.flux_max_chaos,
        thermal_budget_gate=cfg.flux_thermal_budget_gate,
    )

    breeder = BreederDaemonV2(
        latent_dim=cfg.latent_dim,
        population_size=args.pool or cfg.breeding_pool_size,
        mutation_rate=cfg.mutation_rate,
        crossover_rate=cfg.crossover_rate,
        elitism=cfg.elitism,
        flux_config=flux_cfg,
    )

    print(f"Breeding {breeder.population_size} agents, {cfg.generation_limit} generations...")
    for gen in range(args.generations or cfg.generation_limit):
        results = breeder.cycle()
        print(f"Gen {gen}: {len(results)} candidates, top={max((r.fitness for r in results), default=0):.3f}")
        if args.watch and gen < (args.generations or cfg.generation_limit) - 1:
            time.sleep(args.watch)


def cmd_report(args):
    from fleet.deck import breeding_report, fleet_status, flux_gate_decision

    cfg = get_config()

    if args.type == "breeding":
        md = breeding_report(
            generation=args.generation or 0,
            pool_size=cfg.breeding_pool_size,
            pass_rate=args.pass_rate or 0.0,
            top_score=args.top_score or 0.0,
            flux_gate_blocks=args.blocks or 0,
            thermal_violations=args.thermal or 0,
            proof_count=args.proofs or 0,
        )
    elif args.type == "status":
        md = fleet_status(
            services_up=args.up or 0,
            services_down=args.down or 0,
            breeding_active=args.active or False,
            last_proof=args.last_proof,
            blockers=args.blockers.split(",") if args.blockers else [],
        )
    elif args.type == "flux":
        md = flux_gate_decision(
            candidate_id=args.candidate or "unknown",
            passed=args.passed or False,
            score=args.score or 0.0,
            violations={},
            proof_hash=args.proofs or "",
            vm_cycles=args.cycles or 0,
        )
    else:
        print(f"Unknown report type: {args.type}")
        sys.exit(1)

    if args.output:
        with open(args.output, "w") as f:
            f.write(md)
        print(f"Report written to {args.output}")
    else:
        print(md)


def main():
    parser = argparse.ArgumentParser(prog="sunset", description="Sunset Ecosystem CLI")
    parser.add_argument("--format", choices=["json", "md", "oneline"], default="md", help="Output format")
    parser.add_argument("--host", default=None, help="Override service host (env: SUNSET_HOST)")
    parser.add_argument("--config", default=None, help="Path to config YAML")

    sub = parser.add_subparsers(dest="command", help="Commands")

    # status
    p_status = sub.add_parser("status", help="Check fleet health")
    p_status.add_argument("--fail", action="store_true", help="Exit with error if any service down")
    p_status.set_defaults(func=cmd_status)

    # test
    p_test = sub.add_parser("test", help="Run test suite")
    p_test.add_argument("-k", default=None, help="Filter tests by keyword")
    p_test.add_argument("--tb", default="short", help="Traceback style")
    p_test.add_argument("-x", action="store_true", help="Stop on first failure")
    p_test.set_defaults(func=cmd_test)

    # breed
    p_breed = sub.add_parser("breed", help="Trigger breeding cycle")
    p_breed.add_argument("--pool", type=int, help="Population size")
    p_breed.add_argument("--generations", type=int, help="Generation limit")
    p_breed.add_argument("--watch", type=int, help="Sleep N seconds between generations")
    p_breed.set_defaults(func=cmd_breed)

    # report
    p_report = sub.add_parser("report", help="Generate deck report")
    p_report.add_argument("--type", choices=["breeding", "status", "flux"], default="breeding", help="Report type")
    p_report.add_argument("--output", default=None, help="Output file (default: stdout)")
    p_report.add_argument("--generation", type=int, help="Generation number")
    p_report.add_argument("--pass-rate", type=float, help="Pass rate")
    p_report.add_argument("--top-score", type=float, help="Top score")
    p_report.add_argument("--blocks", type=int, help="FLUX gate blocks")
    p_report.add_argument("--thermal", type=int, help="Thermal violations")
    p_report.add_argument("--proofs", type=int, help="Proof count")
    p_report.add_argument("--up", type=int, help="Services up")
    p_report.add_argument("--down", type=int, help="Services down")
    p_report.add_argument("--active", action="store_true", help="Breeding active")
    p_report.add_argument("--last-proof", default=None, help="Last proof hash")
    p_report.add_argument("--blockers", default=None, help="Comma-separated blockers")
    p_report.add_argument("--candidate", default=None, help="Candidate ID")
    p_report.add_argument("--passed", action="store_true", help="Candidate passed")
    p_report.add_argument("--score", type=float, help="Candidate score")
    p_report.add_argument("--cycles", type=int, help="VM cycles")
    p_report.set_defaults(func=cmd_report)

    args = parser.parse_args()

    if args.config:
        get_config(args.config)

    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
