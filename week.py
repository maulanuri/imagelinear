import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import html
import streamlit.components.v1 as components
from io import BytesIO


# =========================
# Page configuration
# =========================
st.set_page_config(
    page_title="2D Transformation Demo (3x3)",
    page_icon="📐",
    layout="wide"
)

# =========================
# Multi-language dictionaries
# =========================

TERMS_MATH = {
    "shape": {
        "Indonesia": "Bentuk",
        "English": "Shape",
    },
    "square": {
        "Indonesia": "Persegi",
        "English": "Square",
    },
    "triangle": {
        "Indonesia": "Segitiga",
        "English": "Triangle",
    },
    "regular_polygon": {
        "Indonesia": "Poligon beraturan",
        "English": "Regular polygon",
    },
    "translation": {
        "Indonesia": "Translation (Pergeseran)",
        "English": "Translation",
    },
    "scaling": {
        "Indonesia": "Scaling (Skala)",
        "English": "Scaling",
    },
    "rotation": {
        "Indonesia": "Rotation (Rotasi)",
        "English": "Rotation",
    },
    "shearing": {
        "Indonesia": "Shearing (Geser)",
        "English": "Shearing",
    },
    "reflection": {
        "Indonesia": "Reflection (Refleksi)",
        "English": "Reflection",
    },
}

TERMS_UI = {
    "app_title": {
        "Indonesia": "Aplikasi Interaktif Transformasi 2D (Koordinat Homogen 3×3)",
        "English": "Interactive 2D Transformation App (Homogeneous Coordinates 3×3)",
    },
    "sidebar_title": {
        "Indonesia": "Kontrol Transformasi",
        "English": "Transformation Controls",
    },
    "sidebar_language": {
        "Indonesia": "Bahasa / Language",
        "English": "Language",
    },
    "sidebar_theme": {
        "Indonesia": "Tampilan",
        "English": "Appearance",
    },
    "theme_light": {
        "Indonesia": "Terang (Light)",
        "English": "Light",
    },
    "theme_dark": {
        "Indonesia": "Gelap (Dark)",
        "English": "Dark",
    },
    "export_select_label": {
        "Indonesia": "Pilih jenis export",
        "English": "Select export type",
    },
}


def tr_math(key: str, lang: str, default: str = "") -> str:
    row = TERMS_MATH.get(key)
    if not row:
        return default
    if lang in row:
        return row[lang]
    if "English" in row:
        return row["English"]
    return default


def tr_ui(key: str, lang: str, default: str = "") -> str:
    row = TERMS_UI.get(key)
    if not row:
        return default
    if lang in row:
        return row[lang]
    if "English" in row:
        return row["English"]
    return default


# =========================
# Theme state
# =========================
if "theme_mode" not in st.session_state:
    st.session_state["theme_mode"] = "Light"

# =========================
# Sidebar
# =========================
with st.sidebar:
    st.title(tr_ui("sidebar_title", "English", "Transformation Controls"))

    st.markdown(f"### {tr_ui('sidebar_language', 'English', 'Language')}")
    lang = st.selectbox(
        tr_ui("sidebar_language", "English", "Language"),
        ["English", "Indonesia"],
        index=0
    )

    st.markdown(f"### {tr_ui('sidebar_theme', lang, 'Appearance')}")
    theme_labels = [
        tr_ui("theme_light", lang, "Light"),
        tr_ui("theme_dark", lang, "Dark")
    ]
    current_idx = 0 if st.session_state.get("theme_mode", "Light") == "Light" else 1
    theme_label_selected = st.radio(
        tr_ui("sidebar_theme", lang, "Appearance"),
        options=theme_labels,
        index=current_idx
    )
    st.session_state["theme_mode"] = "Light" if theme_label_selected == theme_labels[0] else "Dark"

    st.markdown(f"### {tr_math('shape', lang, 'Shape')}")
    shape_choice = st.radio(
        tr_math("shape", lang, "Shape"),
        [
            tr_math("square", lang, "Square"),
            tr_math("triangle", lang, "Triangle"),
            tr_math("regular_polygon", lang, "Regular polygon")
        ],
        key="shape_type_radio"
    )

    if shape_choice == tr_math("square", lang, "Square"):
        shape_key = "square"
    elif shape_choice == tr_math("triangle", lang, "Triangle"):
        shape_key = "triangle"
    else:
        shape_key = "regular_polygon"

    if shape_key == "regular_polygon":
        n_sides = st.slider("Number of polygon sides", 3, 20, 5, 1)
        radius = st.number_input("Polygon radius", 0.1, 5.0, 1.0, 0.1)
    else:
        n_sides = None
        radius = None

    st.markdown(f"### {tr_math('translation', lang, 'Translation')}")
    tx = st.number_input("tx (shift in x)", value=0.0, step=0.1)
    ty = st.number_input("ty (shift in y)", value=0.0, step=0.1)

    st.markdown(f"### {tr_math('scaling', lang, 'Scaling')}")
    sx = st.number_input("sx (scale x)", value=1.0, step=0.1)
    sy = st.number_input("sy (scale y)", value=1.0, step=0.1)

    st.markdown(f"### {tr_math('rotation', lang, 'Rotation')}")
    theta = st.slider("θ (degrees, CCW)", -180, 180, 0, 5)

    st.markdown(f"### {tr_math('shearing', lang, 'Shearing')}")
    shx = st.number_input("shx (shear x)", value=0.0, step=0.1)
    shy = st.number_input("shy (shear y)", value=0.0, step=0.1)

    st.markdown(f"### {tr_math('reflection', lang, 'Reflection')}")
    reflection_mode = st.selectbox(
        "Reflection type",
        [
            "None",
            "Across x-axis",
            "Across y-axis",
            "Across line y = x",
            "Across origin (0,0)"
        ]
    )

    st.markdown("---")
    st.markdown("### Composition Order")
    sequence_label = st.selectbox(
        "Select transformation order",
        [
            "Translation → Rotation → Scaling → Shearing → Reflection",
            "Rotation → Scaling → Translation → Shearing → Reflection",
            "Scaling → Rotation → Translation → Shearing → Reflection",
            "Shearing → Scaling → Rotation → Translation → Reflection",
            "Reflection → Rotation → Scaling → Translation → Shearing"
        ]
    )

    show_grid = st.checkbox("Show grid", value=True)

# =========================
# Theme colors
# =========================
if st.session_state["theme_mode"] == "Dark":
    BG_MAIN = "#0F172A"
    BG_SIDEBAR = "#020617"
    TEXT_COLOR = "#E5E7EB"
    ACCENT_COLOR = "#10B981"
    GRID_COLOR = "#4B5563"
else:
    BG_MAIN = "#FFFFFF"
    BG_SIDEBAR = "#ECFDF5"
    TEXT_COLOR = "#111827"
    ACCENT_COLOR = "#10B981"
    GRID_COLOR = "#9CA3AF"

st.markdown(
    f"""
    <style>
    body {{
        background-color: {BG_MAIN};
        color: {TEXT_COLOR};
    }}
    .block-container {{
        padding-top: 1rem;
        padding-bottom: 1rem;
        background-color: {BG_MAIN};
        color: {TEXT_COLOR};
    }}
    .stSidebar > div {{
        background-color: {BG_SIDEBAR};
        color: {TEXT_COLOR};
    }}
    h1, h2, h3, h4, h5, h6 {{
        color: {ACCENT_COLOR};
    }}
    .stButton>button {{
        background-color: {ACCENT_COLOR};
        color: white;
        border-radius: 0.5rem;
        border: none;
    }}
    .stButton>button:hover {{
        background-color: #059669;
        color: white;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================
# Matrix & shape utilities
# =========================
def translation_matrix(tx, ty):
    return np.array([[1, 0, tx],
                     [0, 1, ty],
                     [0, 0, 1]], dtype=float)


def scaling_matrix(sx, sy):
    return np.array([[sx, 0,  0],
                     [0,  sy, 0],
                     [0,  0,  1]], dtype=float)


def rotation_matrix(theta_deg):
    theta = np.deg2rad(theta_deg)
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0],
                     [s,  c, 0],
                     [0,  0, 1]], dtype=float)


def shearing_matrix(shx, shy):
    return np.array([[1,  shx, 0],
                     [shy, 1,  0],
                     [0,  0,  1]], dtype=float)


def reflection_matrix(mode):
    if mode == "Across x-axis":
        return np.array([[1,  0, 0],
                         [0, -1, 0],
                         [0,  0, 1]], dtype=float)
    elif mode == "Across y-axis":
        return np.array([[-1, 0, 0],
                         [0,  1, 0],
                         [0,  0, 1]], dtype=float)
    elif mode == "Across line y = x":
        return np.array([[0, 1, 0],
                         [1, 0, 0],
                         [0, 0, 1]], dtype=float)
    elif mode == "Across origin (0,0)":
        return np.array([[-1, 0, 0],
                         [0, -1, 0],
                         [0,  0, 1]], dtype=float)
    else:
        return np.eye(3)


def apply_transform(points_xy, M):
    ones = np.ones((points_xy.shape[0], 1))
    pts_h = np.hstack([points_xy, ones])
    pts_trans_h = (M @ pts_h.T).T
    pts_trans_xy = pts_trans_h[:, :2] / pts_trans_h[:, 2][:, np.newaxis]
    return pts_trans_xy


def square_points():
    pts = np.array([[0, 0],
                    [1, 0],
                    [1, 1],
                    [0, 1],
                    [0, 0]], dtype=float)
    labels = ["A", "B", "C", "D", "A"]
    return pts, labels


def triangle_points():
    pts = np.array([[0, 0],
                    [1, 0],
                    [0.5, 1],
                    [0, 0]], dtype=float)
    labels = ["A", "B", "C", "A"]
    return pts, labels


def regular_polygon_points(n_sides=5, radius=1.0, center=(0.0, 0.0)):
    cx, cy = center
    angles = np.linspace(0, 2 * np.pi, n_sides, endpoint=False)
    x = cx + radius * np.cos(angles)
    y = cy + radius * np.sin(angles)
    pts = np.vstack([x, y]).T
    pts = np.vstack([pts, pts[0]])
    labels = [chr(ord("A") + i) for i in range(n_sides)] + ["A"]
    return pts, labels


def polygon_to_svg(points_xy, width=240, height=240, padding=20,
                   stroke_color="#10B981", bg_color="#FFFFFF"):
    if points_xy.shape[0] == 0:
        return f"<svg width='{width}' height='{height}'></svg>"

    xs, ys = points_xy[:, 0], points_xy[:, 1]
    min_x, max_x = xs.min(), xs.max()
    min_y, max_y = ys.min(), ys.max()
    span_x = max(max_x - min_x, 1e-6)
    span_y = max(max_y - min_y, 1e-6)
    scale = min(
        (width - 2 * padding) / span_x,
        (height - 2 * padding) / span_y
    )

    svg_points = []
    for x, y in points_xy:
        sx = padding + (x - min_x) * scale
        sy = height - (padding + (y - min_y) * scale)
        svg_points.append(f"{sx:.2f},{sy:.2f}")

    points_str = " ".join(svg_points)
    points_str = html.escape(points_str)

    svg = f"""
    <svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
        <rect x="0" y="0" width="{width}" height="{height}" fill="{bg_color}" />
        <polyline points="{points_str}"
                  fill="none"
                  stroke="{stroke_color}"
                  stroke-width="2" />
    </svg>
    """
    return svg


# =========================
# Transformation matrices & composite
# =========================
T = translation_matrix(tx, ty)
S = scaling_matrix(sx, sy)
R = rotation_matrix(theta)
H = shearing_matrix(shx, shy)
F = reflection_matrix(reflection_mode)

mapping = {
    "Translation → Rotation → Scaling → Shearing → Reflection": ["T", "R", "S", "H", "F"],
    "Rotation → Scaling → Translation → Shearing → Reflection": ["R", "S", "T", "H", "F"],
    "Scaling → Rotation → Translation → Shearing → Reflection": ["S", "R", "T", "H", "F"],
    "Shearing → Scaling → Rotation → Translation → Shearing": ["H", "S", "R", "T", "F"],
    "Reflection → Rotation → Scaling → Translation → Shearing": ["F", "R", "S", "T", "H"],
}
order = mapping[sequence_label]
symbol_to_matrix = {"T": T, "S": S, "R": R, "H": H, "F": F}

M_composite = np.eye(3)
for symbol in order:
    M_composite = symbol_to_matrix[symbol] @ M_composite

# =========================
# Points & transformed points
# =========================
if shape_key == "square":
    pts, labels = square_points()
elif shape_key == "triangle":
    pts, labels = triangle_points()
else:
    pts, labels = regular_polygon_points(
        n_sides=int(n_sides),
        radius=radius,
        center=(0.0, 0.0)
    )

pts_trans = apply_transform(pts, M_composite)

df_before = pd.DataFrame({"Label": labels, "x": pts[:, 0], "y": pts[:, 1]})
df_after = pd.DataFrame({"Label": [lab + "'" for lab in labels],
                         "x": pts_trans[:, 0], "y": pts_trans[:, 1]})

# =========================
# Main layout
# =========================
st.title(tr_ui("app_title", lang, "Interactive 2D Transformation App (Homogeneous Coordinates 3×3)"))

col_plot, col_info = st.columns([2, 1])

with col_plot:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(df_before["x"], df_before["y"], "-o", color="gray", label="Before")
    for x, y, lab in zip(df_before["x"], df_before["y"], df_before["Label"]):
        ax.text(x, y, f" {lab}", color="gray", fontsize=8)

    ax.plot(df_after["x"], df_after["y"], "-o", color=ACCENT_COLOR, label="After")
    for x, y, lab in zip(df_after["x"], df_after["y"], df_after["Label"]):
        ax.text(x, y, f" {lab}", color=ACCENT_COLOR, fontsize=8)

    ax.set_aspect("equal", "box")
    ax.set_xlabel("x", color=TEXT_COLOR)
    ax.set_ylabel("y", color=TEXT_COLOR)
    ax.set_title("Shape Before and After Transformation", color=TEXT_COLOR)

    ax.tick_params(colors=TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_color(TEXT_COLOR)

    if show_grid:
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7, color=GRID_COLOR)

    all_x = np.concatenate([df_before["x"].values, df_after["x"].values])
    all_y = np.concatenate([df_before["y"].values, df_after["y"].values])
    margin = 1.0
    xmin, xmax = all_x.min() - margin, all_x.max() + margin
    ymin, ymax = all_y.min() - margin, all_y.max() + margin
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    fig.patch.set_facecolor(BG_MAIN)
    ax.set_facecolor(BG_MAIN)

    st.pyplot(fig)

with col_info:
    st.subheader("3×3 Transformation Matrices")

    def format_matrix_html(M, title="Matrix"):
        html_str = f"<b>{title}</b><br><table>"
        for row in M:
            html_str += "<tr>" + "".join(
                f"<td style='padding:2px 8px;'>{val: .3f}</td>" for val in row
            ) + "</tr>"
        html_str += "</table>"
        return html_str

    st.markdown(format_matrix_html(T, "Translation T"), unsafe_allow_html=True)
    st.markdown(format_matrix_html(S, "Scaling S"), unsafe_allow_html=True)
    st.markdown(format_matrix_html(R, "Rotation R"), unsafe_allow_html=True)
    st.markdown(format_matrix_html(H, "Shearing H"), unsafe_allow_html=True)
    st.markdown(format_matrix_html(F, "Reflection F"), unsafe_allow_html=True)
    st.markdown("---")
    st.markdown(f"**Theme mode:** {st.session_state['theme_mode']}")
    st.markdown(f"**Active language:** {lang}")
    st.markdown(f"**Composition order:** {' → '.join(order)}")
    st.markdown(format_matrix_html(M_composite, "Composite Matrix M"), unsafe_allow_html=True)

# SVG preview
st.markdown("## Quick SVG Preview")
col_svg, col_caption = st.columns([1, 1])
with col_svg:
    svg_html = polygon_to_svg(
        pts_trans,
        width=260,
        height=260,
        padding=24,
        stroke_color=ACCENT_COLOR,
        bg_color=BG_MAIN
    )
    components.html(svg_html, height=280)

with col_caption:
    st.markdown("SVG preview shows the final shape using the active theme.")

# Coordinate tables
st.markdown("## Point Coordinates")
col_b, col_a = st.columns(2)
with col_b:
    st.markdown("### Before Transformation")
    st.dataframe(df_before, use_container_width=True)
with col_a:
    st.markdown("### After Transformation")
    st.dataframe(df_after, use_container_width=True)

# =========================
# PDF report (A4) helper
# =========================
def create_full_pdf(df_before, df_after, T, S, R, H, F, M_composite,
                    shape_name, lang, tx, ty, sx, sy, theta, shx, shy,
                    reflection_mode, order):
    from datetime import datetime

    buf = BytesIO()
    # A4 landscape in inches (approx.)
    pdf_fig = plt.figure(figsize=(11.69, 8.27), dpi=120)
    pdf_fig.patch.set_facecolor("white")

    pdf_fig.text(
        0.5, 0.96,
        "2D Transformation Report (Homogeneous Coordinates 3×3)",
        fontsize=14, fontweight="bold", ha="center"
    )

    gs = pdf_fig.add_gridspec(
        2, 2,
        left=0.08, right=0.95,
        top=0.92, bottom=0.08,
        hspace=0.35, wspace=0.3
    )

    # Top-left: plot
    ax_plot = pdf_fig.add_subplot(gs[0, 0])
    ax_plot.plot(df_before["x"], df_before["y"], "-o", color="gray", label="Before",
                 markersize=5, linewidth=1.5)
    for x, y, lab in zip(df_before["x"], df_before["y"], df_before["Label"]):
        ax_plot.text(x, y, f" {lab}", fontsize=8, color="gray")

    ax_plot.plot(df_after["x"], df_after["y"], "-o", color="green", label="After",
                 markersize=5, linewidth=1.5)
    for x, y, lab in zip(df_after["x"], df_after["y"], df_after["Label"]):
        ax_plot.text(x, y, f" {lab}", fontsize=8, color="green")

    ax_plot.set_aspect("equal", "box")
    ax_plot.set_xlabel("x", fontsize=9)
    ax_plot.set_ylabel("y", fontsize=9)
    ax_plot.set_title("Transformation Visualization", fontsize=10, fontweight="bold")
    ax_plot.grid(True, linestyle="--", alpha=0.4, linewidth=0.5)
    ax_plot.legend(fontsize=8)
    ax_plot.tick_params(labelsize=8)

    # Top-right: parameters
    ax_param = pdf_fig.add_subplot(gs[0, 1])
    ax_param.axis("off")

    param_text = f"""REPORT INFO
Shape        : {shape_name}
Language     : {lang}
Generated at : {datetime.now().strftime('%Y-%m-%d %H:%M')}

TRANSFORMATION PARAMETERS
tx (shift X) = {tx:8.3f}
ty (shift Y) = {ty:8.3f}
sx (scale X) = {sx:8.3f}
sy (scale Y) = {sy:8.3f}
θ (rotation) = {theta:8.1f}°
shx (shear X)= {shx:8.3f}
shy (shear Y)= {shy:8.3f}
Reflection   : {reflection_mode}

COMPOSITION ORDER
{' → '.join(order)}
"""

    ax_param.text(
        0.05, 0.95, param_text,
        transform=ax_param.transAxes,
        fontsize=8.5,
        verticalalignment="top",
        family="monospace",
        bbox=dict(
            boxstyle="round,pad=0.8",
            facecolor="#E8F4F8",
            edgecolor="#4B9BC4",
            linewidth=1.2,
            alpha=0.9
        )
    )

    # Bottom-left: coordinate table
    ax_table = pdf_fig.add_subplot(gs[1, 0])
    ax_table.axis("off")

    table_data = [["Label", "x", "y", "x'", "y'"]]
    for i in range(len(df_before)):
        table_data.append([
            str(df_before["Label"].iloc[i]),
            f"{df_before['x'].iloc[i]:.3f}",
            f"{df_before['y'].iloc[i]:.3f}",
            f"{df_after['x'].iloc[i]:.3f}",
            f"{df_after['y'].iloc[i]:.3f}",
        ])

    table = ax_table.table(
        cellText=table_data,
        cellLoc="center",
        loc="center",
        bbox=[0, 0, 1, 1]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1, 1.6)

    for j in range(len(table_data[0])):
        table[(0, j)].set_facecolor("#4B9BC4")
        table[(0, j)].set_text_props(weight="bold", color="white")

    for i in range(1, len(table_data)):
        for j in range(len(table_data[0])):
            if i % 2 == 0:
                table[(i, j)].set_facecolor("#F0F0F0")
            else:
                table[(i, j)].set_facecolor("#FFFFFF")

    ax_table.text(
        0.5, 1.12,
        "Point Coordinates (Before & After)",
        transform=ax_table.transAxes,
        fontsize=9,
        fontweight="bold",
        ha="center"
    )

    # Bottom-right: matrices
    ax_m = pdf_fig.add_subplot(gs[1, 1])
    ax_m.axis("off")

    def fmt_row(row):
        return f"{row[0]:7.3f} {row[1]:7.3f} {row[2]:7.3f}"

    matrices_text = f"""TRANSFORMATION MATRICES (3×3)

Translation T            Scaling S
{fmt_row(T[0])}      {fmt_row(S[0])}
{fmt_row(T[1])}      {fmt_row(S[1])}
{fmt_row(T[2])}      {fmt_row(S[2])}

Rotation R              Shearing H
{fmt_row(R[0])}      {fmt_row(H[0])}
{fmt_row(R[1])}      {fmt_row(H[1])}
{fmt_row(R[2])}      {fmt_row(H[2])}

Reflection F
{fmt_row(F[0])}
{fmt_row(F[1])}
{fmt_row(F[2])}

Composite Matrix M
{fmt_row(M_composite[0])}
{fmt_row(M_composite[1])}
{fmt_row(M_composite[2])}
"""

    ax_m.text(
        0.05, 0.95,
        matrices_text,
        transform=ax_m.transAxes,
        fontsize=7.5,
        verticalalignment="top",
        family="monospace",
        bbox=dict(
            boxstyle="round,pad=0.7",
            facecolor="#FFFEF0",
            edgecolor="#D4A574",
            linewidth=1.2,
            alpha=0.95
        )
    )

    pdf_fig.savefig(buf, format="pdf", bbox_inches="tight", dpi=150)
    buf.seek(0)
    plt.close(pdf_fig)
    return buf.getvalue()


def get_figure_bytes(fig, fmt="png"):
    buf = BytesIO()
    fig.savefig(buf, format=fmt, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()


# =========================
# Export / Download
# =========================
st.markdown("## Export to Download")
export_type = st.selectbox(
    tr_ui("export_select_label", lang, "Select export type"),
    ["Plot (PNG/JPG)", "Full Report (PDF A4)", "Coordinate Data (CSV)"]
)

if export_type == "Plot (PNG/JPG)":
    png_bytes = get_figure_bytes(fig, fmt="png")
    st.download_button(
        label="Download Plot (PNG)",
        data=png_bytes,
        file_name="transform_2d.png",
        mime="image/png"
    )

    jpg_bytes = get_figure_bytes(fig, fmt="jpg")
    st.download_button(
        label="Download Plot (JPG)",
        data=jpg_bytes,
        file_name="transform_2d.jpg",
        mime="image/jpeg"
    )

elif export_type == "Full Report (PDF A4)":
    if shape_key == "square":
        shape_name_display = tr_math("square", lang, "Square")
    elif shape_key == "triangle":
        shape_name_display = tr_math("triangle", lang, "Triangle")
    else:
        shape_name_display = f"Regular polygon ({n_sides} sides)" if n_sides else "Regular polygon"

    pdf_bytes = create_full_pdf(
        df_before, df_after, T, S, R, H, F, M_composite,
        shape_name_display, lang,
        tx, ty, sx, sy, theta, shx, shy, reflection_mode, order
    )

    st.download_button(
        label="📕 Download Full Report (PDF A4)",
        data=pdf_bytes,
        file_name="transform_2d_report.pdf",
        mime="application/pdf",
        key="download_pdf_full"
    )

    st.markdown(
        """
        <style>
        button#download_pdf_full {
            background-color: #DC2626 !important;
            color: white !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

else:
    csv_before = df_before.to_csv(index=False).encode("utf-8")
    csv_after = df_after.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Coordinates Before (CSV)",
        data=csv_before,
        file_name="coords_before.csv",
        mime="text/csv"
    )
    st.download_button(
        label="Download Coordinates After (CSV)",
        data=csv_after,
        file_name="coords_after.csv",
        mime="text/csv"
    )
