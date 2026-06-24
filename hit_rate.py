import subprocess
import re
import json
import argparse
import os
import sys
from datetime import datetime

BEFORE_SNAPSHOT = "hit_rate_before.json"
AFTER_SNAPSHOT = "hit_rate_after.json"


def fetch_metrics(ip, port):
    url = f"http://{ip}:{port}/metrics"
    cmd = f"unset http_proxy; unset https_proxy; sleep 3s; curl -s {url}"

    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
    if result.returncode != 0 or not result.stdout.strip():
        print(f"[WARN] Failed to fetch metrics from {ip}:{port}")
        return {}, {}, {}, {}

    output = result.stdout
    queries_hbm, queries_ext, hits_hbm, hits_ext = {}, {}, {}, {}

    for line in output.split('\n'):
        engine_match = re.search(r'engine="(\d+)"', line)
        if not engine_match:
            continue
        engine = int(engine_match.group(1))
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            value = float(parts[-1])
            if value.is_integer():
                value = int(value)
        except ValueError:
            continue

        # if 'vllm:prefix_cache_queries_total' in line and 'external' not in line:
        if 'vllm:prompt_tokens_total' in line and 'external' not in line:
            queries_hbm[engine] = value
        elif 'external_prefix_cache_queries_total' in line:
            queries_ext[engine] = value
        elif 'vllm:prefix_cache_hits_total' in line and 'external' not in line:
            hits_hbm[engine] = value
        elif 'external_prefix_cache_hits_total' in line:
            hits_ext[engine] = value

    return queries_hbm, queries_ext, hits_hbm, hits_ext


def snapshot_all(pods):
    data = {}
    for pod in pods:
        ip, port = pod.split(":")
        q_hbm, q_ext, h_hbm, h_ext = fetch_metrics(ip, port)
        data[pod] = {
            "queries_hbm": q_hbm,
            "queries_ext": q_ext,
            "hits_hbm": h_hbm,
            "hits_ext": h_ext,
        }
    return data


def save_snapshot(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"timestamp": datetime.now().isoformat(), "data": data}, f, indent=2)
    print(f"[OK] Snapshot saved to {path}")


def _int_keys(d):
    return {int(k): v for k, v in d.items()}


def load_snapshot(path):
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    data = obj["data"]
    for pod, metrics in data.items():
        for key in ("queries_hbm", "queries_ext", "hits_hbm", "hits_ext"):
            metrics[key] = _int_keys(metrics[key])
    return data


def _fmt(q, h):
    rate = f"{h / q:.2%}" if q else "0%"
    detail = f"{h}/{q}" if q else "0/0"
    return rate, detail


def calc_and_print(before, after):
    col1 = 25
    col2 = 20
    col3 = 22
    col4 = 20
    col5 = 22
    total_width = col1 + col2 + col3 + col4 + col5 + 6

    all_q_hbm, all_h_hbm, all_h_ext = 0, 0, 0

    print("\n" + "=" * total_width)
    print(f"{'endpoint':<{col1}} {'hbm_hit_rate':<{col2}} {'hbm(hit/query)':<{col3}} {'ext_hit_rate':<{col4}} {'ext(hit/query)':<{col5}}")
    print("-" * total_width)

    for pod in sorted(before.keys()):
        if pod not in after:
            continue
        b = before[pod]
        a = after[pod]

        engines = sorted(set(b["queries_hbm"].keys()) | set(a["queries_hbm"].keys()))

        for eid in engines:
            q_hbm = a["queries_hbm"].get(eid, 0) - b["queries_hbm"].get(eid, 0)
            h_hbm = a["hits_hbm"].get(eid, 0) - b["hits_hbm"].get(eid, 0)
            h_ext = a["hits_ext"].get(eid, 0) - b["hits_ext"].get(eid, 0)

            all_q_hbm += q_hbm
            all_h_hbm += h_hbm
            all_h_ext += h_ext

            label = f"{pod}/{eid}" if len(engines) > 1 else pod
            hbm_rate, hbm_detail = _fmt(q_hbm, h_hbm)
            ext_rate, ext_detail = _fmt(q_hbm, h_ext)
            print(f"{label:<{col1}} {hbm_rate:<{col2}} {hbm_detail:<{col3}} {ext_rate:<{col4}} {ext_detail:<{col5}}")

    print("-" * total_width)
    hbm_total_rate = f"{all_h_hbm / all_q_hbm:.2%}" if all_q_hbm else "0%"
    hbm_total_detail = f"{all_h_hbm}/{all_q_hbm}" if all_q_hbm else "0/0"
    ext_total_rate = f"{all_h_ext / all_q_hbm:.2%}" if all_q_hbm else "0%"
    ext_total_detail = f"{all_h_ext}/{all_q_hbm}" if all_q_hbm else "0/0"
    print(f"{'TOTAL':<{col1}} {hbm_total_rate:<{col2}} {hbm_total_detail:<{col3}} {ext_total_rate:<{col4}} {ext_total_detail:<{col5}}")
    print("=" * total_width)


def main():
    parser = argparse.ArgumentParser(description="Prefix cache hit rate calculator")
    parser.add_argument("--pods", nargs="+", help="Pod list, format: ip:port")
    parser.add_argument("--action", choices=["before", "after", "calc"], required=True,
                        help="before: save snapshot before test; after: save snapshot after test and calc; calc: calc from two snapshot files")
    parser.add_argument("--before-file", default=BEFORE_SNAPSHOT, help=f"Before snapshot file (default: {BEFORE_SNAPSHOT})")
    parser.add_argument("--after-file", default=AFTER_SNAPSHOT, help=f"After snapshot file (default: {AFTER_SNAPSHOT})")
    args = parser.parse_args()

    if args.action in ("before", "after") and not args.pods:
        parser.error(f"--action {args.action} requires --pods")

    if args.action == "before":
        data = snapshot_all(args.pods)
        save_snapshot(data, args.before_file)

    elif args.action == "after":
        before_data = load_snapshot(args.before_file)
        after_data = snapshot_all(args.pods)
        save_snapshot(after_data, args.after_file)
        calc_and_print(before_data, after_data)

    elif args.action == "calc":
        before_data = load_snapshot(args.before_file)
        after_data = load_snapshot(args.after_file)
        calc_and_print(before_data, after_data)


if __name__ == "__main__":
    main()

# 预埋后测试前执行： 
# python hit_rate.py --action before --pods "10.141.19.144:7850"

# 测试后执行：
# python hit_rate.py --action after --pods "10.141.19.144:7850"
