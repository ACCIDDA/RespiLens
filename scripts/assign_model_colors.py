"""
Gives every forecast model a permanent line style, stored in
app/src/theme/modelColors.json so charts look it up by model name alone.

A model's color and marker never change once assigned; only models new to the
file get picked. Each hub's main ensemble is the same reserved blue and its
baseline the same reserved gray. Models the hub excludes from its ensemble
(designated_model: false) are drawn dashed, except the hub's main ensemble.

The palette is too small for FluSight (over 40 solid models in one season), so:
  - Models that never forecast the same season of the same hub may share a color.
  - Two models that do overlap get a different color if any is free, otherwise
    the same color with a different marker.
"""

import argparse
import json
import re
import subprocess
from pathlib import Path

# MODEL_COLORS in app/src/theme/mantine.js, without its blue and gray: those mean
# hub ensemble and hub baseline.
PALETTE = [
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#bcbd22",
    "#17becf",
    "#3f51b5",
    "#b35900",
    "#0b6e4f",
    "#8e1b3a",
    "#5b2a86",
    "#5e3a1f",
    "#ad1f77",
    "#37717f",
    "#767600",
    "#00897b",
    "#d926a9",
    "#b48ede",
    "#7fb800",
    "#f26d5b",
    "#d4a017",
    "#a68a64",
]
MARKERS = ["circle", "square", "diamond", "triangle-up"]

# ENSEMBLE_COLOR and BASELINE_COLOR in app/src/theme/mantine.js
ENSEMBLE_COLOR = "#0d8ae6"
BASELINE_COLOR = "#a0a0a0"
# The default models in app/src/config/datasets.js
HUB_ENSEMBLES = {
    "FluSight-ensemble",
    "CovidHub-ensemble",
    "RSVHub-ensemble",
    "epiENGAGE-ensemble_short",
    "epiENGAGE-ensemble_long",
}
HUB_BASELINES = {
    "FluSight-baseline",
    "CovidHub-baseline",
    "RSVHub-baseline",
    "epiENGAGE-baseline",
}

OUTPUT_FILENAME = re.compile(r"^model-output/([^/]+)/(\d{4})-(\d{2})-\d{2}")
NOT_DESIGNATED = re.compile(r"^designated_model\s*:\s*false\b", re.IGNORECASE | re.MULTILINE)

DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent / "app/src/theme/modelColors.json"


def season_of(year: str, month: str) -> str:
    """Seasons start in September, matching app/src/utils/forecastSeasons.js."""
    start = int(year) if int(month) >= 9 else int(year) - 1
    return f"{start}-{start + 1}"


def read_hub(hub_path: Path) -> tuple[dict[str, dict], set[str]]:
    """Returns {model: {"seasons": set, "first": "YYYY-MM"}} and the models not designated."""
    # Lists the files from git so the output itself never has to be read
    listing = subprocess.run(
        ["git", "-C", str(hub_path), "ls-tree", "-r", "--name-only", "HEAD", "model-output"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout

    models: dict[str, dict] = {}
    for line in listing.splitlines():
        match = OUTPUT_FILENAME.match(line)
        if not match:
            continue
        model, year, month = match.groups()
        entry = models.setdefault(model, {"seasons": set(), "first": f"{year}-{month}"})
        entry["seasons"].add(season_of(year, month))
        entry["first"] = min(entry["first"], f"{year}-{month}")

    # A single top-level key, so no YAML dependency for it
    not_designated = {
        path.stem
        for path in (hub_path / "model-metadata").glob("*.y*ml")
        if NOT_DESIGNATED.search(path.read_text(errors="replace"))
    }
    return models, not_designated


def assign(hubs: dict[str, tuple[dict, set]], existing: dict[str, dict]) -> dict[str, dict]:
    """Keeps every existing color and marker, and picks them for new models."""
    seasons_by_hub: dict[tuple[str, str], set[str]] = {}
    for hub, (models, _) in hubs.items():
        for model, info in models.items():
            for season in info["seasons"]:
                seasons_by_hub.setdefault((hub, season), set()).add(model)

    overlaps: dict[str, set[str]] = {}
    for members in seasons_by_hub.values():
        for model in members:
            overlaps.setdefault(model, set()).update(members - {model})

    not_designated = set().union(*(hub_not_designated for _, hub_not_designated in hubs.values()))

    def order(model: str) -> tuple:
        # Latest season first, so current models get the strongest colors.
        # FluSight-only models last: the other hubs can stay fully distinct, it can't.
        hub_names = [hub for hub, (models, _) in hubs.items() if model in models]
        infos = [hubs[hub][0][model] for hub in hub_names]
        latest_start = max(int(season[:4]) for info in infos for season in info["seasons"])
        first = min(info["first"] for info in infos)
        return (-latest_start, hub_names == ["flusight"], first, model)

    registry: dict[str, dict] = {}
    for model, style in existing.items():
        registry[model] = {key: style[key] for key in ("color", "marker") if key in style}
        if model in not_designated:
            registry[model]["dashed"] = True

    for model in sorted(overlaps, key=order):
        if model in HUB_ENSEMBLES:
            # Hubs mark their own ensemble as excluded from it, but it's never dashed
            registry[model] = {"color": ENSEMBLE_COLOR}
            continue
        if model in HUB_BASELINES:
            registry[model] = {"color": BASELINE_COLOR}
            if model in not_designated:
                registry[model]["dashed"] = True
            continue
        if model in registry:
            continue
        dashed = model in not_designated
        neighbors = [registry[other] for other in overlaps[model] if other in registry]

        def cost(candidate: tuple[int, int]) -> tuple:
            color_index, marker_index = candidate
            color = PALETTE[color_index]
            marker = MARKERS[marker_index]
            same_color = [n for n in neighbors if n["color"] == color]
            same_style = [n for n in same_color if n.get("dashed", False) == dashed]
            identical = [n for n in same_style if n.get("marker", "circle") == marker]
            return (len(identical), len(same_style), len(same_color), color_index, marker_index)

        candidates = [
            (color_index, marker_index)
            for color_index in range(len(PALETTE))
            for marker_index in range(len(MARKERS))
        ]
        color_index, marker_index = min(candidates, key=cost)
        registry[model] = {"color": PALETTE[color_index]}
        if marker_index:
            registry[model]["marker"] = MARKERS[marker_index]
        if dashed:
            registry[model]["dashed"] = True

    return registry


def to_json(registry: dict[str, dict]) -> str:
    # One model per line, the way Prettier keeps it, so a new model is a one-line diff
    lines = [
        f"  {json.dumps(model)}: {{ {', '.join(f'{json.dumps(k)}: {json.dumps(v)}' for k, v in registry[model].items())} }}"
        for model in sorted(registry)
    ]
    return "{\n" + ",\n".join(lines) + "\n}\n"


def report(hubs: dict[str, tuple[dict, set]], registry: dict[str, dict]) -> None:
    for hub, (models, _) in hubs.items():
        latest = max(season for info in models.values() for season in info["seasons"])
        current = [
            model
            for model, info in models.items()
            if latest in info["seasons"] and model in registry
        ]
        styles = [
            (registry[m]["color"], registry[m].get("marker", "circle"), registry[m].get("dashed", False))
            for m in current
        ]
        colors = [(style[0], style[2]) for style in styles]
        print(
            f"{hub} {latest}: {len(current)} models, "
            f"{len(current) - len(set(colors))} share a color and line style, "
            f"{len(current) - len(set(styles))} share color, line style and marker"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--flusight-hub-path", type=Path)
    parser.add_argument("--rsv-hub-path", type=Path)
    parser.add_argument("--covid-hub-path", type=Path)
    parser.add_argument("--flu-metrocast-hub-path", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    paths = {
        "flusight": args.flusight_hub_path,
        "rsv": args.rsv_hub_path,
        "covid": args.covid_hub_path,
        "metrocast": args.flu_metrocast_hub_path,
    }
    hubs = {hub: read_hub(path) for hub, path in paths.items() if path}

    existing = json.loads(args.output.read_text()) if args.output.exists() else {}
    registry = assign(hubs, existing)
    args.output.write_text(to_json(registry))
    print(f"{len(registry) - len(existing)} new models, {len(registry)} total in {args.output}")
    report(hubs, registry)


if __name__ == "__main__":
    main()
