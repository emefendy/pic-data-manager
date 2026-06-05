from __future__ import annotations

import argparse
import sys

from tidy3d_pipeline.store import PicStore


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
    p = argparse.ArgumentParser(prog="tidy3d_pipeline", description="Real PIC data pipeline")
    p.add_argument("--root", default="data/tidy3d_processed")
    sub = p.add_subparsers(dest="cmd", required=True)

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
