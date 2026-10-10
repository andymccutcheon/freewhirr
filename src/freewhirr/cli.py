"""Command-line interface: chat, serve, stats, models, config."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from freewhirr import __version__
from freewhirr.config import FreewhirrConfig, discover_config_path, load_config
from freewhirr.errors import FreewhirrError
from freewhirr.router import Freewhirr
from freewhirr.stats import collect_stats
from freewhirr.stealth import (
    build_watch_report,
    diff_and_score,
    format_report_lines,
    post_webhook,
    sleep_interval,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="freewhirr",
        description="Spend nothing until you choose to.",
    )
    parser.add_argument("--version", action="version", version=f"freewhirr {__version__}")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML or TOML config file (or set FREEWHIRR_CONFIG).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    chat = sub.add_parser("chat", help="Send a one-shot chat completion.")
    chat.add_argument("prompt", nargs="*", help="User prompt. Reads stdin if omitted.")
    chat.add_argument("--system", default=None, help="Optional system message.")
    chat.add_argument("--stream", action="store_true", help="Stream tokens to stdout.")
    chat.add_argument(
        "--complexity",
        choices=["low", "normal", "high"],
        default=None,
        help="Tag used by paid_only_high_complexity.",
    )
    chat.add_argument("--json", action="store_true", dest="as_json", help="Print the raw response.")

    serve = sub.add_parser("serve", help="Run an OpenAI-compatible proxy.")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8787)

    sub.add_parser("stats", help="Show free vs paid share and estimated savings.")
    models = sub.add_parser("models", help="List the free models the router would try.")
    models.add_argument("--refresh", action="store_true", help="Bypass the catalog cache.")
    models.add_argument(
        "--new",
        action="store_true",
        dest="only_new",
        help="Show $0 models first seen since the last catalog snapshot.",
    )
    models.add_argument(
        "--stealth",
        action="store_true",
        dest="only_stealth",
        help="Show models that look like stealth/cloaked drops.",
    )
    watch = sub.add_parser("watch", help="Poll OpenRouter for new and stealth free models.")
    watch.add_argument(
        "--interval",
        type=float,
        default=None,
        help="Seconds between polls (FREEWHIRR_WATCH_INTERVAL / watch_interval).",
    )
    watch.add_argument(
        "--once",
        action="store_true",
        help="Scan once and exit (useful for tests and cron).",
    )
    sub.add_parser("config", help="Print the resolved configuration (secrets redacted).")

    args = parser.parse_args(argv)
    try:
        if args.command == "chat":
            return _cmd_chat(args)
        if args.command == "serve":
            return _cmd_serve(args)
        if args.command == "stats":
            return _cmd_stats(args)
        if args.command == "models":
            return _cmd_models(args)
        if args.command == "watch":
            return _cmd_watch(args)
        if args.command == "config":
            return _cmd_config(args)
    except FreewhirrError as exc:
        print(f"freewhirr: {exc}", file=sys.stderr)
        return 2
    return 1


def _load(args: argparse.Namespace, **overrides: Any) -> Freewhirr:
    config = load_config(args.config, overrides=overrides)
    return Freewhirr(config)


def _cmd_chat(args: argparse.Namespace) -> int:
    prompt = " ".join(args.prompt).strip()
    if not prompt:
        prompt = sys.stdin.read().strip()
    if not prompt:
        print("freewhirr: no prompt provided", file=sys.stderr)
        return 2
    messages: list[dict[str, str]] = []
    if args.system:
        messages.append({"role": "system", "content": args.system})
    messages.append({"role": "user", "content": prompt})
    client = _load(args)
    try:
        if args.stream:
            for chunk in client.stream(messages=messages, complexity=args.complexity):
                for choice in chunk.get("choices") or []:
                    delta = choice.get("delta") or {}
                    text = delta.get("content")
                    if text:
                        sys.stdout.write(text)
                        sys.stdout.flush()
            sys.stdout.write("\n")
            return 0
        result = client.complete(messages=messages, complexity=args.complexity)
        if args.as_json:
            print(json.dumps(result.to_openai(), indent=2))
        else:
            print(result.content)
        return 0
    finally:
        client.close()


def _cmd_serve(args: argparse.Namespace) -> int:
    from freewhirr.proxy import run

    client = _load(args)
    print(f"freewhirr proxy on http://{args.host}:{args.port}/v1", file=sys.stderr)
    run(host=args.host, port=args.port, client=client)
    return 0


def _cmd_stats(args: argparse.Namespace) -> int:
    client = _load(args)
    try:
        print(collect_stats(client.store).format())
        return 0
    finally:
        client.close()


def _cmd_models(args: argparse.Namespace) -> int:
    client = _load(args)
    try:
        if args.only_new or args.only_stealth:
            _new_ids, reports = diff_and_score(client)
            selected = reports
            if args.only_new:
                selected = [item for item in selected if item.is_new]
            if args.only_stealth:
                selected = [item for item in selected if item.is_stealth]
            for line in format_report_lines(selected):
                print(line)
            return 0
        if args.refresh:
            client.refresh_models(force=True)
        for model_id in client.list_free_models():
            print(model_id)
        if client.config.exhaustion_policy.value == "paid":
            print("# paid fallback")
            for model_id in client.config.paid_models:
                print(model_id)
        return 0
    finally:
        client.close()


def _cmd_watch(args: argparse.Namespace) -> int:
    client = _load(args)
    interval = args.interval if args.interval is not None else client.config.watch_interval
    webhook = client.config.alert_webhook
    try:
        while True:
            _new_ids, reports = diff_and_score(client)
            report = build_watch_report(reports)
            print(
                f"watch: scanned={report.scanned} new={len(report.new)} "
                f"stealth={len(report.stealth)} excluded={len(report.excluded)}"
            )
            for line in format_report_lines([*report.new, *report.stealth]):
                print(line)
            if webhook:
                try:
                    post_webhook(webhook, report)
                except Exception as exc:  # webhook must not kill the poller
                    print(f"freewhirr: webhook failed: {exc}", file=sys.stderr)
            if args.once:
                return 0
            sleep_interval(interval)
    finally:
        client.close()


def _cmd_config(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    found = discover_config_path(args.config)
    payload = _redact(config)
    payload["_config_file"] = str(found) if found else None
    print(json.dumps(payload, indent=2, default=str))
    return 0


def _redact(config: FreewhirrConfig) -> dict[str, Any]:
    data = config.model_dump()
    if data.get("api_key"):
        data["api_key"] = "***"
    return data


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
