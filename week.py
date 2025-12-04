import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import html
import streamlit.components.v1 as components

# =========================
# Konfigurasi halaman
# =========================
st.set_page_config(
    page_title="Demo Transformasi 2D (3x3)",
    page_icon="📐",
    layout="wide"
)

ACCENT_COLOR = "#10B981"  # hijau utama (emerald)
# Sedikit custom CSS untuk tema hijau & layout
st.markdown(
    f"""
    <style>
    :root {{
        --accent: {ACCENT_COLOR};
    }}
    .block-container {{
        padding-top: 1rem;
        padding-bottom: 1rem;
    }}
    h1, h2, h3, h4, h5, h6 {{
        color: {ACCENT_COLOR};
    }}
    .stSidebar > div {{
        background-color: #ECFDF5;
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
# Fungsi utilitas matriks
# =========================

def translation_matrix(tx, ty):
    """Matriks translasi 3x3 (koordinat homogen)."""
    return np.array([
        [1, 0, tx],
        [0, 1, ty],
        [0, 0, 1]
    ], dtype=float)

def scaling_matrix(sx, sy):
    """Matriks skala 3x3."""
    return np.array([
        [sx, 0,  0],
        [0,  sy, 0],
        [0,  0,  1]
    ], dtype=float)

def rotation_matrix(theta_deg):
    """Matriks rotasi 3x3 (derajat, CCW)."""
    theta = np.deg2rad(theta_deg)
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1]
    ], dtype=float)

def shearing_matrix(shx, shy):
    """Matriks shear 3x3."""
    return np.array([
        [1,  shx, 0],
        [shy, 1,  0],
        [0,  0,  1]
    ], dtype=float)

def reflection_matrix(mode):
    """
    Matriks refleksi 3x3:
    - Tidak ada
    - Terhadap sumbu x
    - Terhadap sumbu y
    - Terhadap garis y = x
    - Terhadap titik asal (0,0)
    """
    if mode == "Terhadap sumbu x":
        return np.array([
            [1,  0, 0],
            [0, -1, 0],
            [0,  0, 1]
        ], dtype=float)
    elif mode == "Terhadap sumbu y":
        return np.array([
            [-1, 0, 0],
            [0,  1, 0],
            [0,  0, 1]
        ], dtype=float)
    elif mode == "Terhadap garis y = x":
        # Tukar x dan y
        return np.array([
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, 1]
        ], dtype=float)
    elif mode == "Terhadap titik asal (0,0)":
        return np.array([
            [-1, 0, 0],
            [0, -1, 0],
            [0,  0, 1]
        ], dtype=float)
    else:  # "Tidak ada"
        return np.eye(3)

def apply_transform(points_xy, M):
    """
    Terapkan transformasi homogen 3x3 pada array titik (N,2).
    Mengembalikan array (N,2) hasil transformasi.
    """
    ones = np.ones((points_xy.shape[0], 1))
    pts_h = np.hstack([points_xy, ones])  # (N,3)
    pts_trans_h = (M @ pts_h.T).T         # (N,3)
    pts_trans_xy = pts_trans_h[:, :2] / pts_trans_h[:, 2][:, np.newaxis]
    return pts_trans_xy

# =========================
# Bentuk dasar
# =========================

def square_points():
    """
    Persegi satuan dengan titik ditutup A→B→C→D→A.
    """
    pts = np.array([
        [0, 0],
        [1, 0],
        [1, 1],
        [0, 1],
        [0, 0]
    ], dtype=float)
    labels = ["A", "B", "C", "D", "A"]
    return pts, labels

def triangle_points():
    """
    Segitiga sederhana.
    """
    pts = np.array([
        [0, 0],
        [1, 0],
        [0.5, 1],
        [0, 0]
    ], dtype=float)
    labels = ["A", "B", "C", "A"]
    return pts, labels

def regular_polygon_points(n_sides=5, radius=1.0, center=(0.0, 0.0)):
    """
    Menghasilkan titik-titik poligon beraturan dengan n_sides.
    Titik terakhir = titik pertama untuk menutup poligon.
    """
    cx, cy = center
    angles = np.linspace(0, 2 * np.pi, n_sides, endpoint=False)
    x = cx + radius * np.cos(angles)
    y = cy + radius * np.sin(angles)
    pts = np.vstack([x, y]).T
    pts = np.vstack([pts, pts[0]])  # tutup poligon
    labels = [chr(ord("A") + i) for i in range(n_sides)] + ["A"]
    return pts, labels

# =========================
# Utilitas tampilan matriks & SVG
# =========================

def format_matrix_html(M, title="Matriks"):
    """Format matriks 3x3 menjadi HTML sederhana."""
    html_str = f"<b>{title}</b><br><table>"
    for row in M:
        html_str += "<tr>" + "".join(
            f"<td style='padding:2px 8px;'>{val: .3f}</td>" for val in row
        ) + "</tr>"
    html_str += "</table>"
    return html_str

def polygon_to_svg(points_xy, width=240, height=240, padding=20, stroke_color="#10B981"):
    """
    Menghasilkan string HTML <svg> sederhana untuk preview bentuk 2D.
    Hanya menggambar bentuk hasil transformasi (polyline tertutup).
    """
    if points_xy.shape[0] == 0:
        return f"<svg width='{width}' height='{height}'></svg>"

    xs = points_xy[:, 0]
    ys = points_xy[:, 1]

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
        <rect x="0" y="0" width="{width}" height="{height}" fill="white" />
        <polyline points="{points_str}"
                  fill="none"
                  stroke="{stroke_color}"
                  stroke-width="2" />
    </svg>
    """
    return svg

# =========================
# Sidebar: kontrol
# =========================

with st.sidebar:
    st.title("Kontrol Transformasi")

    st.markdown("### Bentuk Dasar")
    shape_type = st.radio(
        "Pilih bentuk:",
        ["Persegi", "Segitiga", "Poligon beraturan"]
    )

    if shape_type == "Poligon beraturan":
        n_sides = st.slider(
            "Jumlah sisi poligon",
            min_value=3,
            max_value=20,
            value=5,
            step=1
        )
        radius = st.number_input(
            "Radius poligon",
            min_value=0.1,
            max_value=5.0,
            value=1.0,
            step=0.1
        )
    else:
        n_sides = None
        radius = None

    st.markdown("---")
    st.markdown("### Translation")
    tx = st.number_input("tx (geser sumbu x)", value=0.0, step=0.1)
    ty = st.number_input("ty (geser sumbu y)", value=0.0, step=0.1)

    st.markdown("### Scaling")
    sx = st.number_input("sx (skala sumbu x)", value=1.0, step=0.1)
    sy = st.number_input("sy (skala sumbu y)", value=1.0, step=0.1)

    st.markdown("### Rotation")
    theta = st.slider("θ (derajat, CCW)", min_value=-180, max_value=180, value=0, step=5)

    st.markdown("### Shearing")
    shx = st.number_input("shx (shear x)", value=0.0, step=0.1)
    shy = st.number_input("shy (shear y)", value=0.0, step=0.1)

    st.markdown("### Reflection")
    reflection_mode = st.selectbox(
        "Jenis refleksi",
        [
            "Tidak ada",
            "Terhadap sumbu x",
            "Terhadap sumbu y",
            "Terhadap garis y = x",
            "Terhadap titik asal (0,0)"
        ]
    )

    st.markdown("---")
    st.markdown("### Urutan Komposisi")
    sequence_label = st.selectbox(
        "Pilih urutan transformasi",
        [
            "Translation → Rotation → Scaling → Shearing → Reflection",
            "Rotation → Scaling → Translation → Shearing → Reflection",
            "Scaling → Rotation → Translation → Shearing → Reflection",
            "Shearing → Scaling → Rotation → Translation → Reflection",
            "Reflection → Rotation → Scaling → Translation → Shearing"
        ]
    )

    show_grid = st.checkbox("Tampilkan grid", value=True)

# =========================
# Hitung matriks transformasi
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
    "Shearing → Scaling → Rotation → Translation → Reflection": ["H", "S", "R", "T", "F"],
    "Reflection → Rotation → Scaling → Translation → Shearing": ["F", "R", "S", "T", "H"],
}
order = mapping[sequence_label]

symbol_to_matrix = {
    "T": T,
    "S": S,
    "R": R,
    "H": H,
    "F": F
}

M_composite = np.eye(3)
for symbol in order:
    M_composite = symbol_to_matrix[symbol] @ M_composite

# =========================
# Data titik & transformasi
# =========================

if shape_type == "Persegi":
    pts, labels = square_points()
elif shape_type == "Segitiga":
    pts, labels = triangle_points()
else:  # Poligon beraturan
    pts, labels = regular_polygon_points(
        n_sides=int(n_sides),
        radius=radius,
        center=(0.0, 0.0)
    )

pts_trans = apply_transform(pts, M_composite)

# =========================
# Layout utama
# =========================

st.title("Aplikasi Interaktif Transformasi 2D (Koordinat Homogen 3×3)")
st.markdown(
    """
Aplikasi ini mendemonstrasikan bagaimana translation, scaling, rotation, shearing, 
dan reflection pada 2D direpresentasikan dengan matriks 3×3 dalam koordinat homogen, 
serta bagaimana transformasi-transformasi tersebut dapat dikomposisikan menjadi satu 
matriks komposit.
"""
)

col_plot, col_info = st.columns([2, 1])

# -------------------------
# Plot Matplotlib
# -------------------------
with col_plot:
    fig, ax = plt.subplots(figsize=(6, 6))

    # Bentuk awal
    ax.plot(pts[:, 0], pts[:, 1], "-o", color="gray", label="Sebelum")
    for (x, y), lab in zip(pts, labels):
        ax.text(x, y, f" {lab}", color="gray")

    # Bentuk setelah transformasi
    ax.plot(pts_trans[:, 0], pts_trans[:, 1], "-o", color=ACCENT_COLOR, label="Sesudah")
    for (x, y), lab in zip(pts_trans, labels):
        ax.text(x, y, f" {lab}'", color=ACCENT_COLOR)

    ax.set_aspect("equal", "box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Bentuk Sebelum dan Sesudah Transformasi")
    if show_grid:
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

    all_x = np.concatenate([pts[:, 0], pts_trans[:, 0]])
    all_y = np.concatenate([pts[:, 1], pts_trans[:, 1]])
    margin = 1.0
    xmin, xmax = all_x.min() - margin, all_x.max() + margin
    ymin, ymax = all_y.min() - margin, all_y.max() + margin
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    st.pyplot(fig)

# -------------------------
# Info matriks & urutan
# -------------------------
with col_info:
    st.subheader("Matriks Transformasi 3×3")
    st.markdown(format_matrix_html(T, "Translation T"), unsafe_allow_html=True)
    st.markdown(format_matrix_html(S, "Scaling S"), unsafe_allow_html=True)
    st.markdown(format_matrix_html(R, "Rotation R"), unsafe_allow_html=True)
    st.markdown(format_matrix_html(H, "Shearing H"), unsafe_allow_html=True)
    st.markdown(format_matrix_html(F, "Reflection F"), unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(f"**Urutan komposisi dipilih:** {sequence_label}")
    st.markdown(f"**Notasi urutan:** {' → '.join(order)}")
    st.markdown(format_matrix_html(M_composite, "Matriks Komposit M"), unsafe_allow_html=True)

# =========================
# Preview SVG ringan
# =========================

st.markdown("## Preview Cepat (SVG)")

col_svg, col_caption = st.columns([1, 1])

with col_svg:
    svg_html = polygon_to_svg(pts_trans, width=260, height=260, padding=24, stroke_color=ACCENT_COLOR)
    components.html(svg_html, height=280)

with col_caption:
    st.markdown(
        """
Preview ini menggambar bentuk akhir dalam SVG sederhana sehingga terasa ringan 
saat parameter diubah, sementara plot Matplotlib tetap menyediakan visualisasi 
lengkap dengan grid dan label titik.
"""
    )

# =========================
# Tabel koordinat
# =========================

st.markdown("## Koordinat Titik Sebelum dan Sesudah Transformasi")

df_before = pd.DataFrame({
    "Label": labels,
    "x": pts[:, 0],
    "y": pts[:, 1]
})

df_after = pd.DataFrame({
    "Label": [lab + "'" for lab in labels],
    "x'": pts_trans[:, 0],
    "y'": pts_trans[:, 1]
})

col_before, col_after = st.columns(2)

with col_before:
    st.markdown("### Sebelum Transformasi")
    st.dataframe(df_before, use_container_width=True)

with col_after:
    st.markdown("### Sesudah Transformasi")
    st.dataframe(df_after, use_container_width=True)

st.markdown(
    """
Gunakan kontrol di sidebar untuk mengubah parameter transformasi dan urutan komposisi, 
lalu amati bagaimana matriks, koordinat, dan bentuk pada bidang 2D ikut berubah sesuai 
konsep transformasi linear dan koordinat homogen 3×3.
"""
)
