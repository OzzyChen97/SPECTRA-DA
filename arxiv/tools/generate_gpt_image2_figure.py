#!/usr/bin/env python3
"""Generate SPECTRA-DA Figure 1 candidates with the project AutoSOTA API config.

This script intentionally reads the workspace-level AutoSOTA `config.yaml`
instead of relying on a global OPENAI_API_KEY. It prints only non-secret
metadata.
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
from typing import Any

import yaml
from openai import OpenAI


def _read_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg


def _default_workspace_config() -> Path:
    return Path(__file__).resolve().parents[3] / "config.yaml"


def _response_image_bytes(result: Any) -> bytes:
    item = result.data[0]
    b64 = getattr(item, "b64_json", None)
    if b64:
        return base64.b64decode(b64)
    url = getattr(item, "url", None)
    if url:
        raise RuntimeError(
            "The image API returned a URL rather than base64 data. "
            "This script expects base64 output so generated assets can be saved "
            "directly without a second network fetch."
        )
    raise RuntimeError("No image payload found in API response.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=_default_workspace_config(),
        help="Workspace AutoSOTA config.yaml containing research_api_key/base_url.",
    )
    parser.add_argument(
        "--prompts",
        type=Path,
        default=Path("notes/gpt_image2_architecture_v2_prompts.jsonl"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("figures/gpt-image-2-candidates-v2"),
    )
    parser.add_argument("--size", default="1536x864")
    parser.add_argument("--quality", default="high")
    parser.add_argument("--limit", type=int, default=4)
    args = parser.parse_args()

    cfg = _read_config(args.config)
    api_key = (cfg.get("research_api_key") or cfg.get("openrouter_api_key") or "").strip()
    base_url = (cfg.get("research_base_url") or "").strip() or None
    if not api_key:
        raise SystemExit(f"No research_api_key/openrouter_api_key found in {args.config}")

    timeout_minutes = int(cfg.get("research_timeout_minutes") or 30)
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_minutes * 60)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    with args.prompts.open("r", encoding="utf-8") as f:
        specs = [json.loads(line) for line in f if line.strip()]
    specs = specs[: args.limit]

    print(
        json.dumps(
            {
                "config": str(args.config),
                "base_url_configured": bool(base_url),
                "model": "gpt-image-2",
                "size": args.size,
                "quality": args.quality,
                "num_candidates": len(specs),
                "out_dir": str(args.out_dir),
            },
            ensure_ascii=False,
        )
    )

    for idx, spec in enumerate(specs, start=1):
        out = args.out_dir / spec["out"]
        prompt = spec["prompt"]
        print(f"[{idx}/{len(specs)}] generating {out.name}", flush=True)
        result = client.images.generate(
            model="gpt-image-2",
            prompt=prompt,
            size=args.size,
            quality=args.quality,
            response_format="b64_json",
        )
        out.write_bytes(_response_image_bytes(result))
        print(f"[{idx}/{len(specs)}] saved {out}", flush=True)


if __name__ == "__main__":
    main()
