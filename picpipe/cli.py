"""Command-line interface for the picpipe real-data pipeline.

    python -m picpipe.cli demo                  # ingest + analyze + optimize + figures
    python -m picpipe.cli summary
    python -m picpipe.cli query --component y_junction
    python -m picpipe.cli query --metric insertion_loss_mean
    python -m picpipe.cli filter --geom l_junction --min 1.0 --max 2.0
"""

from __future__ import annotations

import argparse
import sys

from picpipe.store import PicStore


def _cmd_demo(args) -> int:
    from picpipe.demo import run_demo
    out = run_demo(db_path=args.db, spectra_dir=args.spectra, root=args.root,
                   figdir=f"{args.root}/figures")
    print(f"Ingested {out['n_runs']} real Tidy3D runs.\n")
    print("=== SUMMARY ===")
    print(out["summary"].to_string(index=False))
    print("\n=== OPTIMIZATION ===")
    for k, v in out["optimization"].items():
        print(f"  {k}: {v}")
    print("\nFigures:")
    for k, v in out["figures"].items():
        print(f"  {k}: {v}")
    print(f"Summary CSV: {out['summary_csv']}")
    return 0


def _cmd_summary(args) -> int:
    with PicStore(args.root, read_only=True) as s:
        print(s.summary_table().to_string(index=False))
    return 0


def _cmd_query(args) -> int:
    with PicStore(args.root, read_only=True) as s:
        if args.component:
            print("== Runs ==")
            print(s.runs_for_component(args.component)[
                ["run_id", "tool", "is_synthetic", "wl_min_um", "wl_max_um"]].to_string(index=False))
            print("\n== Metrics ==")
            print(s.metrics(component_id=args.component)[
                ["metric_name", "metric_value", "units", "run_id"]].to_string(index=False))
        elif args.metric:
            print(s.metrics(metric_name=args.metric)[
                ["component_id", "metric_name", "metric_value", "units"]].to_string(index=False))
        else:
            print(s.components()[["component_id", "type", "name", "library"]].to_string(index=False))
    return 0


def _cmd_filter(args) -> int:
    with PicStore(args.root, read_only=True) as s:
        print(s.filter_runs_by_geometry(args.geom, min_val=args.min, max_val=args.max)[
            ["run_id", "component_id", "geom_value"]].to_string(index=False))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="picpipe", description="Real PIC data pipeline")
    p.add_argument("--root", default="data/pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo", help="run the end-to-end real-data demo")
    d.add_argument("--db", default="pic_data.db")
    d.add_argument("--spectra", default="data/spectra")
    d.set_defaults(func=_cmd_demo)

    sub.add_parser("summary", help="print the summary table").set_defaults(func=_cmd_summary)

    q = sub.add_parser("query", help="query the store")
    q.add_argument("--component"); q.add_argument("--metric")
    q.set_defaults(func=_cmd_query)

    f = sub.add_parser("filter", help="filter runs by a geometry parameter")
    f.add_argument("--geom", required=True); f.add_argument("--min", type=float)
    f.add_argument("--max", type=float)
    f.set_defaults(func=_cmd_filter)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
