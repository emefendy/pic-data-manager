from __future__ import annotations

import argparse
import sys

from reallabdata_pipeline.store import ScreeningStore


def _cmd_metrics(args) -> int:
    with ScreeningStore(args.root) as s:
        print(s.device_metrics_summary().to_string(index=False))
    return 0


def _cmd_captures(args) -> int:
    with ScreeningStore(args.root) as s:
        print(s.captures(rep_hz=args.rep, edge_ns=args.edge, vlbl=args.vlbl).to_string(index=False))
    return 0


def _cmd_features(args) -> int:
    with ScreeningStore(args.root) as s:
        print(s.feature_table().to_string(index=False))
    return 0


def _cmd_logs(args) -> int:
    with ScreeningStore(args.root) as s:
        df = s.lab_logs()
        for _, r in df.iterrows():
            print(f"\n[{r['date']}] {r['title']}")
            print(f"  refs: {r['referenced_tags']}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="reallabdata_pipeline", description="ScreeningDatasetV2 pipeline")
    p.add_argument("--root", default="data/reallabdata_processed", help="store directory")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("metrics", help="device-level metrics").set_defaults(func=_cmd_metrics)

    c = sub.add_parser("captures", help="filter capture measurements")
    c.add_argument("--rep", type=float, default=None, help="rep rate (Hz)")
    c.add_argument("--edge", type=float, default=None, help="edge time (ns)")
    c.add_argument("--vlbl", default=None, help="drive variant: sym1/sym5/sym10/asymHi/asymLo")
    c.set_defaults(func=_cmd_captures)

    sub.add_parser("features", help="per-capture feature table").set_defaults(func=_cmd_features)
    sub.add_parser("logs", help="operator lab logs").set_defaults(func=_cmd_logs)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
