from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


FIG_DIR = Path(__file__).resolve().parent
SIZE = (3200, 2550)
BG = "#fbfbf8"
TEXT = "#202124"
AXIS = "#6e6e6e"
BASELINE = "#dbe8f6"
BASELINE_EDGE = "#3f6ea4"
VAE = "#dff0e5"
VAE_EDGE = "#2b7b57"
NEUTRAL = "#f2efe6"
NEUTRAL_EDGE = "#8b7c63"
ACCENT = "#ececec"
RIDGE = "#eef4fb"
RIDGE_EDGE = "#567aa5"


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
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


FONT_TITLE = load_font(42, bold=True)
FONT_PANEL = load_font(34, bold=True)
FONT_BOX = load_font(24)
FONT_SMALL = load_font(21)


def multiline_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    left, top, right, bottom = draw.multiline_textbbox((0, 0), text, font=font, spacing=5, align="center")
    return right - left, bottom - top


def draw_centered_text(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: str,
) -> None:
    x0, y0, x1, y1 = box
    width, height = multiline_size(draw, text, font)
    draw.multiline_text(
        ((x0 + x1 - width) / 2, (y0 + y1 - height) / 2),
        text,
        font=font,
        fill=fill,
        spacing=5,
        align="center",
    )


def rounded_box(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    fill: str,
    outline: str,
    text: str,
    font: ImageFont.ImageFont,
) -> None:
    draw.rounded_rectangle(box, radius=24, fill=fill, outline=outline, width=4)
    draw_centered_text(draw, box, text, font, TEXT)


def center_left(box: tuple[int, int, int, int]) -> tuple[int, int]:
    x0, y0, _, y1 = box
    return x0, (y0 + y1) // 2


def center_right(box: tuple[int, int, int, int]) -> tuple[int, int]:
    _, y0, x1, y1 = box
    return x1, (y0 + y1) // 2


def center_top(box: tuple[int, int, int, int]) -> tuple[int, int]:
    x0, y0, x1, _ = box
    return (x0 + x1) // 2, y0


def center_bottom(box: tuple[int, int, int, int]) -> tuple[int, int]:
    x0, _, x1, y1 = box
    return (x0 + x1) // 2, y1


def arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int], color: str, width: int = 6) -> None:
    draw.line((start, end), fill=color, width=width)
    x0, y0 = start
    x1, y1 = end
    if abs(x1 - x0) >= abs(y1 - y0):
        direction = 1 if x1 >= x0 else -1
        tip = (x1, y1)
        base1 = (x1 - 18 * direction, y1 - 10)
        base2 = (x1 - 18 * direction, y1 + 10)
    else:
        direction = 1 if y1 >= y0 else -1
        tip = (x1, y1)
        base1 = (x1 - 10, y1 - 18 * direction)
        base2 = (x1 + 10, y1 - 18 * direction)
    draw.polygon([tip, base1, base2], fill=color)


def orthogonal_arrow(draw: ImageDraw.ImageDraw, points: list[tuple[int, int]], color: str, width: int = 6) -> None:
    if len(points) < 2:
        return
    for start, end in zip(points[:-2], points[1:-1]):
        draw.line((start, end), fill=color, width=width)
    arrow(draw, points[-2], points[-1], color, width=width)


def build() -> None:
    img = Image.new("RGB", SIZE, BG)
    draw = ImageDraw.Draw(img)

    draw.text((70, 40), "Residue-graph model families used in the comparison", font=FONT_TITLE, fill=TEXT)
    draw.text(
        (82, 105),
        "Both paths use the same encoder design, but they are trained and evaluated differently.",
        font=FONT_SMALL,
        fill="#555555",
    )

    common_box = (260, 150, 2940, 240)
    draw.rounded_rectangle(common_box, radius=24, outline=AXIS, width=3, fill="white")
    draw_centered_text(
        draw,
        common_box,
        "Common encoder design in both model families: linear node/edge projections -> 3 GINE layers -> mean + max graph pooling",
        FONT_SMALL,
        TEXT,
    )

    left_panel = (70, 300, 980, 2460)
    right_panel = (1020, 300, 3130, 2460)
    draw.rounded_rectangle(left_panel, radius=34, outline=BASELINE_EDGE, width=4, fill="white")
    draw.rounded_rectangle(right_panel, radius=34, outline=VAE_EDGE, width=4, fill="white")
    draw.text((110, 325), "Direct supervised baseline", font=FONT_PANEL, fill=BASELINE_EDGE)
    draw.text((1360, 325), "Masked GraphVAE", font=FONT_PANEL, fill=VAE_EDGE)

    baseline_input = (160, 440, 860, 580)
    baseline_proj = (160, 650, 860, 790)
    baseline_gine = (160, 860, 860, 1000)
    baseline_pool = (160, 1070, 860, 1210)
    baseline_embed = (160, 1280, 860, 1420)
    baseline_reg = (160, 1560, 460, 1700)
    baseline_pred = (560, 1560, 860, 1700)
    baseline_loss = (160, 1870, 460, 2010)
    baseline_truth = (560, 1870, 860, 2010)

    rounded_box(draw, baseline_input, NEUTRAL, NEUTRAL_EDGE, "Residue graph\nnode + edge features", FONT_BOX)
    rounded_box(draw, baseline_proj, NEUTRAL, NEUTRAL_EDGE, "Linear node / edge projection\n128 hidden units", FONT_BOX)
    rounded_box(draw, baseline_gine, NEUTRAL, NEUTRAL_EDGE, "3 x GINE layers\nedge-aware message passing", FONT_BOX)
    rounded_box(draw, baseline_pool, NEUTRAL, NEUTRAL_EDGE, "Global mean pooling +\nglobal max pooling", FONT_BOX)
    rounded_box(draw, baseline_embed, ACCENT, AXIS, "Pooled graph embedding", FONT_BOX)
    rounded_box(draw, baseline_reg, BASELINE, BASELINE_EDGE, "Two-layer MLP\nregressor", FONT_BOX)
    rounded_box(draw, baseline_pred, BASELINE, BASELINE_EDGE, "Predicted\nDelta G", FONT_BOX)
    rounded_box(draw, baseline_loss, BASELINE, BASELINE_EDGE, "Affinity loss +\nheld-out metrics", FONT_BOX)
    rounded_box(draw, baseline_truth, RIDGE, RIDGE_EDGE, "Experimental\nDelta G", FONT_BOX)

    arrow(draw, center_bottom(baseline_input), center_top(baseline_proj), AXIS)
    arrow(draw, center_bottom(baseline_proj), center_top(baseline_gine), AXIS)
    arrow(draw, center_bottom(baseline_gine), center_top(baseline_pool), AXIS)
    arrow(draw, center_bottom(baseline_pool), center_top(baseline_embed), AXIS)
    orthogonal_arrow(draw, [center_bottom(baseline_embed), (center_bottom(baseline_embed)[0], 1500), center_top(baseline_reg)], BASELINE_EDGE)
    arrow(draw, center_right(baseline_reg), center_left(baseline_pred), BASELINE_EDGE)
    orthogonal_arrow(draw, [center_bottom(baseline_pred), (center_bottom(baseline_pred)[0], 1800), (center_top(baseline_loss)[0], 1800), center_top(baseline_loss)], BASELINE_EDGE)
    orthogonal_arrow(draw, [center_left(baseline_truth), (500, center_left(baseline_truth)[1]), center_right(baseline_loss)], RIDGE_EDGE)

    panel_pad = 90
    branch_gap = 170
    branch_width = ((right_panel[2] - right_panel[0]) - 2 * panel_pad - branch_gap) // 2
    left_branch_x0 = right_panel[0] + panel_pad
    left_branch_x1 = left_branch_x0 + branch_width
    right_branch_x0 = left_branch_x1 + branch_gap
    right_branch_x1 = right_panel[2] - panel_pad
    shared_width = 1320
    shared_x0 = (right_panel[0] + right_panel[2] - shared_width) // 2
    shared_x1 = shared_x0 + shared_width

    vae_input = (shared_x0, 430, shared_x1, 560)
    vae_proj = (shared_x0, 620, shared_x1, 750)
    vae_gine = (shared_x0, 810, shared_x1, 940)
    vae_pool = (shared_x0, 1000, shared_x1, 1130)
    mu_box = (shared_x0, 1190, shared_x1, 1320)

    left_label_x = left_branch_x0 + 150
    right_label_x = right_branch_x0 + 160
    branch_label_y = 1385

    z_box = (left_branch_x0 + 60, 1450, left_branch_x0 + 420, 1590)
    aff_head = (left_branch_x0 + 520, 1450, left_branch_x1 - 20, 1590)
    node_dec = (left_branch_x0 + 30, 1710, left_branch_x0 + 340, 1850)
    edge_dec = (left_branch_x0 + 410, 1710, left_branch_x0 + 720, 1850)
    mask_targets = (left_branch_x0 + 120, 1980, left_branch_x0 + 630, 2120)
    aff_compare = (left_branch_x0 + 600, 1980, left_branch_x1 - 10, 2120)

    export_box = (right_branch_x0 + 40, 1450, right_branch_x1 - 40, 1590)
    ridge_box = (right_branch_x0 + 170, 1710, right_branch_x1 - 170, 1850)
    ridge_pred = (right_branch_x0 + 30, 1980, right_branch_x0 + 360, 2120)
    ridge_truth = (right_branch_x1 - 360, 1980, right_branch_x1 - 30, 2120)
    ridge_eval = (right_branch_x0 + 40, 2250, right_branch_x1 - 40, 2390)

    rounded_box(draw, vae_input, NEUTRAL, NEUTRAL_EDGE, "Masked residue graph\n+ mask indicators", FONT_BOX)
    rounded_box(draw, vae_proj, NEUTRAL, NEUTRAL_EDGE, "Linear node / edge projection\n128 hidden units", FONT_BOX)
    rounded_box(draw, vae_gine, NEUTRAL, NEUTRAL_EDGE, "3 x GINE layers\nedge-aware message passing", FONT_BOX)
    rounded_box(draw, vae_pool, NEUTRAL, NEUTRAL_EDGE, "Global mean pooling +\nglobal max pooling", FONT_BOX)
    rounded_box(draw, mu_box, VAE, VAE_EDGE, "mu head + log sigma^2 head", FONT_BOX)

    draw.text((left_label_x, branch_label_y), "VAE training", font=FONT_SMALL, fill=VAE_EDGE)
    draw.text((right_label_x, branch_label_y), "RidgeCV reporting", font=FONT_SMALL, fill=RIDGE_EDGE)

    rounded_box(draw, z_box, VAE, VAE_EDGE, "Sample z\ntraining path", FONT_BOX)
    rounded_box(draw, node_dec, VAE, VAE_EDGE, "Node decoder", FONT_BOX)
    rounded_box(draw, edge_dec, VAE, VAE_EDGE, "Edge decoder", FONT_BOX)
    rounded_box(draw, mask_targets, VAE, VAE_EDGE, "Reconstruction loss on\nmasked node / edge targets", FONT_BOX)

    rounded_box(draw, aff_head, VAE, VAE_EDGE, "Optional affinity head\nsemi-supervised only", FONT_BOX)
    rounded_box(draw, aff_compare, VAE, VAE_EDGE, "Affinity loss vs\nexperimental Delta G", FONT_BOX)

    rounded_box(draw, export_box, RIDGE, RIDGE_EDGE, "Export graph-level mu\nfor train / val / test", FONT_BOX)
    rounded_box(draw, ridge_box, RIDGE, RIDGE_EDGE, "RidgeCV on\nexported mu", FONT_BOX)
    rounded_box(draw, ridge_pred, RIDGE, RIDGE_EDGE, "Predicted\nDelta G", FONT_BOX)
    rounded_box(draw, ridge_truth, RIDGE, RIDGE_EDGE, "Held-out experimental\nDelta G", FONT_BOX)
    rounded_box(draw, ridge_eval, RIDGE, RIDGE_EDGE, "Reported GraphVAE metrics:\nRidgeCV prediction vs held-out Delta G", FONT_BOX)

    arrow(draw, center_bottom(vae_input), center_top(vae_proj), AXIS)
    arrow(draw, center_bottom(vae_proj), center_top(vae_gine), AXIS)
    arrow(draw, center_bottom(vae_gine), center_top(vae_pool), AXIS)
    arrow(draw, center_bottom(vae_pool), center_top(mu_box), VAE_EDGE)

    branch_y = mu_box[3] + 70
    orthogonal_arrow(draw, [center_bottom(mu_box), (center_bottom(mu_box)[0], branch_y), (center_top(z_box)[0], branch_y), center_top(z_box)], VAE_EDGE)
    orthogonal_arrow(draw, [center_bottom(mu_box), (center_bottom(mu_box)[0], branch_y), (center_top(aff_head)[0], branch_y), center_top(aff_head)], VAE_EDGE)
    orthogonal_arrow(draw, [center_bottom(z_box), (center_bottom(z_box)[0], 1660), (center_top(node_dec)[0], 1660), center_top(node_dec)], VAE_EDGE)
    orthogonal_arrow(draw, [center_bottom(z_box), (center_bottom(z_box)[0], 1660), (center_top(edge_dec)[0], 1660), center_top(edge_dec)], VAE_EDGE)
    orthogonal_arrow(draw, [center_bottom(node_dec), (center_bottom(node_dec)[0], 1930), (center_top(mask_targets)[0] - 140, 1930), (center_top(mask_targets)[0] - 140, center_top(mask_targets)[1]), center_top(mask_targets)], VAE_EDGE)
    orthogonal_arrow(draw, [center_bottom(edge_dec), (center_bottom(edge_dec)[0], 1930), (center_top(mask_targets)[0] + 140, 1930), (center_top(mask_targets)[0] + 140, center_top(mask_targets)[1]), center_top(mask_targets)], VAE_EDGE)
    orthogonal_arrow(draw, [center_bottom(aff_head), (center_bottom(aff_head)[0], 1930), center_top(aff_compare)], VAE_EDGE)

    orthogonal_arrow(draw, [center_bottom(mu_box), (center_bottom(mu_box)[0], branch_y), (center_top(export_box)[0], branch_y), center_top(export_box)], RIDGE_EDGE)
    orthogonal_arrow(draw, [center_bottom(export_box), (center_bottom(export_box)[0], 1660), center_top(ridge_box)], RIDGE_EDGE)
    orthogonal_arrow(draw, [center_bottom(ridge_box), (center_bottom(ridge_box)[0], 1930), (center_top(ridge_pred)[0], 1930), center_top(ridge_pred)], RIDGE_EDGE)
    orthogonal_arrow(draw, [center_bottom(ridge_pred), (center_bottom(ridge_pred)[0], 2200), (center_top(ridge_eval)[0] - 210, 2200), (center_top(ridge_eval)[0] - 210, center_top(ridge_eval)[1]), center_top(ridge_eval)], RIDGE_EDGE)
    orthogonal_arrow(draw, [center_bottom(ridge_truth), (center_bottom(ridge_truth)[0], 2200), (center_top(ridge_eval)[0] + 210, 2200), (center_top(ridge_eval)[0] + 210, center_top(ridge_eval)[1]), center_top(ridge_eval)], RIDGE_EDGE)

    png_path = FIG_DIR / "model_architecture_clean.png"
    pdf_path = FIG_DIR / "model_architecture_clean.pdf"
    img.save(png_path)
    img.save(pdf_path, "PDF", resolution=300.0)


if __name__ == "__main__":
    build()
