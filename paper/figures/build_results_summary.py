from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = Path(__file__).resolve().parent
SUMMARY_CSV = ROOT / "RESULTS" / "final_resample_matrix_summary.csv"

IMAGE_SIZE = (2400, 1050)
BACKGROUND = "white"
GREEN = "#5fb760"
GREEN_DARK = "#2f8d46"
ORANGE = "#e67e22"
TEXT = "#222222"
AXIS = "#555555"
GRID = "#c5c5c5"

TARGET_LABELS = {
    "shared_static": "static-only",
    "sd_static_plus_dynamic_all": "static+dynamic",
    "sd_dynamic_all": "dynamic-only",
}


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = []
    if bold:
        candidates.extend(
            [
                "C:/Windows/Fonts/segoeuib.ttf",
                "C:/Windows/Fonts/arialbd.ttf",
            ]
        )
    candidates.extend(
        [
            "C:/Windows/Fonts/segoeui.ttf",
            "C:/Windows/Fonts/arial.ttf",
        ]
    )
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


FONT_TITLE = load_font(34, bold=True)
FONT_AXIS = load_font(22)
FONT_LABEL = load_font(19)
FONT_TICK = load_font(18)
FONT_WINS = load_font(22)


def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    left, top, right, bottom = draw.multiline_textbbox((0, 0), text, font=font, spacing=4, align="left")
    return right - left, bottom - top


def pretty_label(row: pd.Series) -> str:
    tag = str(row["tag"])
    family = str(row["model_family"])
    if family == "supervised_baseline":
        prefix = "Full baseline" if tag.startswith("full_") else "Interface baseline"
        return f"{prefix}, {row['mode']}"

    prefix = "Full GraphVAE" if tag.startswith("full_") else "Interface GraphVAE"
    mode = str(row["mode"])
    supervision = "semi" if row["supervision_mode"] == "semi_supervised" else "unsup"
    parts = [prefix, mode, supervision]
    target_policy = row["target_policy"]
    if pd.notna(target_policy) and mode == "SD":
        parts.append(TARGET_LABELS.get(str(target_policy), str(target_policy)))
    latent_dim = row["latent_dim"]
    if pd.notna(latent_dim):
        parts.append(f"z={int(latent_dim)}")
    return ", ".join(parts)


def linspace_ticks(start: float, end: float, step: float) -> list[float]:
    first = math.floor(start / step) * step
    last = math.ceil(end / step) * step
    ticks = []
    value = first
    while value <= last + 1e-9:
        ticks.append(round(value, 6))
        value += step
    return ticks


def draw_centered(draw: ImageDraw.ImageDraw, xy: tuple[float, float], text: str, font: ImageFont.ImageFont, fill: str) -> None:
    width, height = text_size(draw, text, font)
    x, y = xy
    draw.multiline_text((x - width / 2, y - height / 2), text, font=font, fill=fill, spacing=4, align="center")


def build_main_results() -> None:
    df = pd.read_csv(SUMMARY_CSV)
    ranking = df.sort_values("test_rmse_mean", ascending=True).head(12).copy()
    ranking["label"] = ranking.apply(pretty_label, axis=1)

    factor_rows = [
        ("Full over interface", 0.1005, "24 / 24"),
        ("S over SD\n(static-only target)", 0.0254, "11 / 12"),
        ("Semi over unsup, full S", 0.1640, "3 / 3"),
        ("Semi over unsup, full SD", 0.1604, "9 / 9"),
        ("Semi over unsup, interface S", 0.0290, "3 / 3"),
        ("Semi over unsup, interface SD", -0.0081, "3 / 9"),
    ]

    img = Image.new("RGB", IMAGE_SIZE, BACKGROUND)
    draw = ImageDraw.Draw(img)

    # Panel geometry
    left_label_left = 10
    left_label_right = 470
    left_plot_left = 490
    left_plot_right = 1365
    right_label_left = 1395
    right_label_right = 1785
    right_plot_left = 1810
    right_plot_right = 2360
    top = 92
    bottom = 885
    axis_label_y = 960

    draw_centered(draw, ((left_plot_left + left_plot_right) / 2, 38), "Top of the 52-model ranking", FONT_TITLE, TEXT)
    draw_centered(draw, ((right_plot_left + right_plot_right) / 2, 38), "Matched factor comparisons", FONT_TITLE, TEXT)

    # Left ranking panel
    rank_min = float((ranking["test_rmse_mean"] - ranking["test_rmse_std"]).min()) - 0.02
    rank_max = float((ranking["test_rmse_mean"] + ranking["test_rmse_std"]).max()) + 0.02
    rank_ticks = linspace_ticks(rank_min, rank_max, 0.05)
    rank_min = rank_ticks[0]
    rank_max = rank_ticks[-1]

    def rank_x(value: float) -> float:
        return left_plot_left + (value - rank_min) / (rank_max - rank_min) * (left_plot_right - left_plot_left)

    for tick in rank_ticks:
        x = rank_x(tick)
        draw.line((x, top, x, bottom), fill=GRID, width=1)
        label = f"{tick:.2f}"
        w, h = text_size(draw, label, FONT_TICK)
        draw.text((x - w / 2, bottom + 12), label, font=FONT_TICK, fill=TEXT)

    draw.rectangle((left_plot_left, top, left_plot_right, bottom), outline=AXIS, width=2)

    row_height = (bottom - top) / len(ranking)
    for idx, row in enumerate(ranking.itertuples(index=False)):
        y = top + row_height * (idx + 0.5)
        draw.line((rank_x(row.test_rmse_mean - row.test_rmse_std), y, rank_x(row.test_rmse_mean + row.test_rmse_std), y), fill=GREEN_DARK, width=3)
        cap = 7
        left_cap_x = rank_x(row.test_rmse_mean - row.test_rmse_std)
        right_cap_x = rank_x(row.test_rmse_mean + row.test_rmse_std)
        draw.line((left_cap_x, y - cap, left_cap_x, y + cap), fill=GREEN_DARK, width=3)
        draw.line((right_cap_x, y - cap, right_cap_x, y + cap), fill=GREEN_DARK, width=3)
        point_x = rank_x(row.test_rmse_mean)
        radius = 6
        draw.ellipse((point_x - radius, y - radius, point_x + radius, y + radius), fill=GREEN_DARK, outline=GREEN_DARK)

        label_w, label_h = text_size(draw, row.label, FONT_LABEL)
        draw.text((left_label_right - label_w - 10, y - label_h / 2), row.label, font=FONT_LABEL, fill=TEXT)

    draw_centered(
        draw,
        ((left_plot_left + left_plot_right) / 2, axis_label_y),
        "Test RMSE (mean +/- SD over 10 outer fits)",
        FONT_AXIS,
        TEXT,
    )

    # Right factor panel
    gains = [row[1] for row in factor_rows]
    min_gain = min(gains) - 0.012
    max_gain = max(gains) + 0.028
    gain_ticks = linspace_ticks(min_gain, max_gain, 0.025)
    min_gain = gain_ticks[0]
    max_gain = gain_ticks[-1]

    def factor_x(value: float) -> float:
        return right_plot_left + (value - min_gain) / (max_gain - min_gain) * (right_plot_right - right_plot_left)

    for tick in gain_ticks:
        x = factor_x(tick)
        draw.line((x, top, x, bottom), fill=GRID, width=1)
        label = f"{tick:.3f}"
        w, h = text_size(draw, label, FONT_TICK)
        draw.text((x - w / 2, bottom + 12), label, font=FONT_TICK, fill=TEXT)

    draw.rectangle((right_plot_left, top, right_plot_right, bottom), outline=AXIS, width=2)
    zero_x = factor_x(0.0)
    draw.line((zero_x, top, zero_x, bottom), fill=AXIS, width=2)

    factor_row_height = (bottom - top) / len(factor_rows)
    right_label_x = factor_x(max(gains)) + 12
    for idx, (label, gain, wins) in enumerate(factor_rows):
        y = top + factor_row_height * (idx + 0.5)
        bar_half = factor_row_height * 0.32
        x0 = factor_x(min(0.0, gain))
        x1 = factor_x(max(0.0, gain))
        fill = GREEN if gain >= 0 else ORANGE
        draw.rectangle((x0, y - bar_half, x1, y + bar_half), fill=fill, outline=fill)

        label_w, label_h = text_size(draw, label, FONT_LABEL)
        draw.multiline_text((right_label_right - label_w - 10, y - label_h / 2), label, font=FONT_LABEL, fill=TEXT, spacing=4, align="left")

        wins_w, wins_h = text_size(draw, wins, FONT_WINS)
        draw.text((right_label_x, y - wins_h / 2), wins, font=FONT_WINS, fill=TEXT)

    draw_centered(
        draw,
        ((right_plot_left + right_plot_right) / 2, axis_label_y),
        "Mean test-RMSE advantage for first option",
        FONT_AXIS,
        TEXT,
    )

    png_path = FIG_DIR / "main_results_summary.png"
    pdf_path = FIG_DIR / "main_results_summary.pdf"
    img.save(png_path)
    img.save(pdf_path, "PDF", resolution=300.0)


if __name__ == "__main__":
    build_main_results()
