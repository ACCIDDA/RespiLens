import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "scripts"))

from assign_model_colors import (
    BASELINE_COLOR,
    ENSEMBLE_COLOR,
    HUB_BASELINES,
    HUB_ENSEMBLES,
    MARKERS,
    PALETTE,
    assign,
    to_json,
)


def _hub(seasons_by_model: dict, not_designated: set = frozenset()) -> tuple:
    models = {
        model: {"seasons": set(seasons), "first": f"{min(seasons)[:4]}-09"}
        for model, seasons in seasons_by_model.items()
    }
    return models, set(not_designated)


def test_existing_styles_never_change():
    existing = {"old": {"color": "#c7c7c7", "marker": "diamond"}}
    hubs = {"flusight": _hub({"old": ["2025-2026"], "new": ["2025-2026"]})}

    registry = assign(hubs, existing)

    assert registry["old"] == existing["old"]
    assert registry["new"]["color"] != "#c7c7c7"


def test_assigning_again_changes_nothing():
    hubs = {"rsv": _hub({f"m{i}": ["2025-2026"] for i in range(30)})}

    first = assign(hubs, {})

    assert assign(hubs, first) == first


def test_a_new_model_does_not_move_earlier_ones():
    before = {"rsv": _hub({"zeta": ["2025-2026"], "yota": ["2025-2026"]})}
    after = {"rsv": _hub({"zeta": ["2025-2026"], "yota": ["2025-2026"], "alpha": ["2025-2026"]})}

    first = assign(before, {})
    second = assign(after, first)

    assert {model: second[model] for model in first} == first


def test_overlapping_models_get_different_colors():
    hubs = {"covid": _hub({f"m{i}": ["2025-2026"] for i in range(len(PALETTE))})}

    registry = assign(hubs, {})

    assert len({style["color"] for style in registry.values()}) == len(PALETTE)


def test_models_from_different_seasons_can_share_a_color():
    hubs = {"covid": _hub({"retired": ["2023-2024"], "current": ["2025-2026"]})}

    registry = assign(hubs, {})

    assert registry["retired"]["color"] == registry["current"]["color"]


def test_more_models_than_colors_get_different_markers():
    count = len(PALETTE) * len(MARKERS)
    hubs = {"flusight": _hub({f"m{i}": ["2025-2026"] for i in range(count)})}

    registry = assign(hubs, {})

    styles = {(style["color"], style.get("marker", "circle")) for style in registry.values()}
    assert len(styles) == count


def test_only_hub_ensembles_are_blue_and_never_dashed():
    hubs = {
        "flusight": _hub(
            {"FluSight-ensemble": ["2025-2026"], "UVAFluX-Ensemble": ["2025-2026"]},
            {"FluSight-ensemble"},
        )
    }

    registry = assign(hubs, {})

    assert registry["FluSight-ensemble"] == {"color": ENSEMBLE_COLOR}
    assert registry["UVAFluX-Ensemble"]["color"] in PALETTE


def test_only_hub_baselines_are_gray_and_dashed_if_left_out():
    assert BASELINE_COLOR not in PALETTE
    hubs = {
        "flusight": _hub(
            {"FluSight-baseline": ["2025-2026"], "CMU-climate_baseline": ["2025-2026"]},
            {"FluSight-baseline", "CMU-climate_baseline"},
        ),
        "metrocast": _hub({"epiENGAGE-baseline": ["2025-2026"]}),
    }

    registry = assign(hubs, {})

    assert registry["FluSight-baseline"] == {"color": BASELINE_COLOR, "dashed": True}
    assert registry["epiENGAGE-baseline"] == {"color": BASELINE_COLOR}
    assert registry["CMU-climate_baseline"]["color"] in PALETTE
    assert registry["CMU-climate_baseline"]["dashed"] is True


def test_palette_is_the_app_palette_without_its_reserved_colors():
    theme = (ROOT / "app/src/theme/mantine.js").read_text()
    app_colors = re.findall(r'"(#[0-9a-f]{6})"', theme[: theme.index("];")])

    assert app_colors == ["#1f77b4"] + PALETTE[:6] + [BASELINE_COLOR] + PALETTE[6:]


def test_hub_ensembles_match_the_app():
    datasets = (ROOT / "app/src/config/datasets.js").read_text()
    theme = (ROOT / "app/src/theme/mantine.js").read_text()

    assert set(re.findall(r'[dD]efaultModel: "([^"]+)"', datasets)) == HUB_ENSEMBLES
    assert f'ENSEMBLE_COLOR = "{ENSEMBLE_COLOR}"' in theme
    assert f'BASELINE_COLOR = "{BASELINE_COLOR}"' in theme


def test_not_designated_models_are_dashed_and_follow_the_hub():
    hubs = {"rsv": _hub({"a": ["2025-2026"], "b": ["2025-2026"]}, {"b"})}

    registry = assign(hubs, {})
    assert "dashed" not in registry["a"]
    assert registry["b"]["dashed"] is True

    # The hub changes its mind: the line style follows, the color stays
    color = registry["b"]["color"]
    registry = assign({"rsv": _hub({"a": ["2025-2026"], "b": ["2025-2026"]}, {"a"})}, registry)
    assert registry["a"]["dashed"] is True
    assert "dashed" not in registry["b"]
    assert registry["b"]["color"] == color


def test_file_is_valid_json_with_one_model_per_line():
    registry = {"b": {"color": "#ff7f0e", "dashed": True}, "a": {"color": "#2ca02c"}}

    text = to_json(registry)

    assert json.loads(text) == registry
    assert text.splitlines()[1].startswith('  "a"')
    assert len(text.splitlines()) == 4
