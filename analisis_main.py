import streamlit as st
import numpy as np
from PIL import Image
import io
import cv2  # make sure: pip install opencv-python
import zipfile

st.set_page_config(page_title="Matrix Image Processing", layout="wide")

# =========================================================
# BASIC UTILITIES
# =========================================================

def pil_to_array(img: Image.Image) -> np.ndarray:
    img = img.convert("RGBA")
    return np.array(img)

def array_to_pil(arr: np.ndarray) -> Image.Image:
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def add_alpha_channel(arr: np.ndarray) -> np.ndarray:
    if arr.shape[-1] == 4:
        return arr
    h, w, _ = arr.shape
    alpha = 255 * np.ones((h, w, 1), dtype=arr.dtype)
    return np.concatenate([arr, alpha], axis=-1)

# =========================================================
# STATE: UNDO / REDO + SAVED RESULTS
# =========================================================

def init_state():
    if "history" not in st.session_state:
        st.session_state.history = []          # list of PIL images
    if "redo_stack" not in st.session_state:
        st.session_state.redo_stack = []       # list of PIL images
    if "current_image" not in st.session_state:
        st.session_state.current_image = None  # PIL image
    if "saved_results" not in st.session_state:
        st.session_state.saved_results = []    # list of (name, PIL image)

def push_history(img: Image.Image):
    if st.session_state.current_image is not None:
        st.session_state.history.append(st.session_state.current_image.copy())
    st.session_state.current_image = img.copy()
    st.session_state.redo_stack.clear()

def undo():
    if st.session_state.history:
        last = st.session_state.history.pop()
        if st.session_state.current_image is not None:
            st.session_state.redo_stack.append(st.session_state.current_image.copy())
        st.session_state.current_image = last

def redo():
    if st.session_state.redo_stack:
        img = st.session_state.redo_stack.pop()
        if st.session_state.current_image is not None:
            st.session_state.history.append(st.session_state.current_image.copy())
        st.session_state.current_image = img

def save_current_result(name: str):
    if st.session_state.current_image is not None and name.strip():
        st.session_state.saved_results.append((name.strip(), st.session_state.current_image.copy()))

def make_zip_from_results(results):
    mem_zip = io.BytesIO()
    with zipfile.ZipFile(mem_zip, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, img in results:
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            zf.writestr(f"{name}.png", buf.getvalue())
    mem_zip.seek(0)
    return mem_zip

# =========================================================
# GEOMETRIC TRANSFORMATIONS (MANUAL 3x3 MATRIX)
# =========================================================

def apply_affine_transform(img_arr: np.ndarray, M: np.ndarray) -> np.ndarray:
    img_arr = add_alpha_channel(img_arr)
    h, w, _ = img_arr.shape
    out = np.zeros_like(img_arr)
    Minv = np.linalg.inv(M)

    for y_out in range(h):
        for x_out in range(w):
            src_coord = Minv @ np.array([x_out, y_out, 1.0])
            x_src, y_src = src_coord[0], src_coord[1]
            if 0 <= x_src < w and 0 <= y_src < h:
                x0, y0 = int(x_src), int(y_src)
                out[y_out, x_out] = img_arr[y0, x0]
    return out

def get_translation_matrix(tx: float, ty: float) -> np.ndarray:
    return np.array([[1, 0, tx],
                     [0, 1, ty],
                     [0, 0, 1]], dtype=float)

def get_scaling_matrix(sx: float, sy: float, cx: float, cy: float) -> np.ndarray:
    T1 = get_translation_matrix(-cx, -cy)
    S = np.array([[sx, 0, 0],
                  [0, sy, 0],
                  [0, 0,  1]], dtype=float)
    T2 = get_translation_matrix(cx, cy)
    return T2 @ S @ T1

def get_rotation_matrix(angle_deg: float, cx: float, cy: float) -> np.ndarray:
    rad = np.deg2rad(angle_deg)
    cos_a = np.cos(rad)
    sin_a = np.sin(rad)
    T1 = get_translation_matrix(-cx, -cy)
    R = np.array([[cos_a, -sin_a, 0],
                  [sin_a,  cos_a, 0],
                  [0,      0,     1]], dtype=float)
    T2 = get_translation_matrix(cx, cy)
    return T2 @ R @ T1

def get_shearing_matrix(shx: float, shy: float, cx: float, cy: float) -> np.ndarray:
    T1 = get_translation_matrix(-cx, -cy)
    Sh = np.array([[1,  shx, 0],
                   [shy, 1,  0],
                   [0,   0,  1]], dtype=float)
    T2 = get_translation_matrix(cx, cy)
    return T2 @ Sh @ T1

def get_reflection_matrix(axis: str, cx: float, cy: float) -> np.ndarray:
    T1 = get_translation_matrix(-cx, -cy)
    if axis == "Horizontal":
        R = np.array([[1,  0, 0],
                      [0, -1, 0],
                      [0,  0, 1]], dtype=float)
    elif axis == "Vertical":
        R = np.array([[-1, 0, 0],
                      [0,  1, 0],
                      [0,  0, 1]], dtype=float)
    else:  # Both
        R = np.array([[-1, 0, 0],
                      [0, -1, 0],
                      [0,  0, 1]], dtype=float)
    T2 = get_translation_matrix(cx, cy)
    return T2 @ R @ T1

# =========================================================
# MANUAL CONVOLUTION (BLUR & SHARPEN)
# =========================================================

def manual_convolution_gray(img_gray: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    kh, kw = kernel.shape
    pad_h = kh // 2
    pad_w = kw // 2
    padded = np.pad(img_gray, ((pad_h, pad_h), (pad_w, pad_w)), mode="edge")
    h, w = img_gray.shape
    out = np.zeros_like(img_gray, dtype=float)

    for y in range(h):
        for x in range(w):
            region = padded[y:y+kh, x:x+kw]
            out[y, x] = np.sum(region * kernel)

    out = np.clip(out, 0, 255)
    return out.astype(np.uint8)

def blur_filter(img_arr: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    img_arr = img_arr.astype(np.float32)
    rgb = img_arr[..., :3]
    alpha = img_arr[..., 3]
    gray = np.mean(rgb, axis=2).astype(np.uint8)
    kernel = np.ones((kernel_size, kernel_size), dtype=float) / (kernel_size * kernel_size)
    blurred_gray = manual_convolution_gray(gray, kernel)
    blurred_rgb = np.stack([blurred_gray]*3, axis=-1)
    out = np.dstack([blurred_rgb, alpha])
    return out

def sharpen_filter(img_arr: np.ndarray) -> np.ndarray:
    img_arr = img_arr.astype(np.float32)
    rgb = img_arr[..., :3]
    alpha = img_arr[..., 3]
    gray = np.mean(rgb, axis=2).astype(np.uint8)
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]], dtype=float)
    sharp_gray = manual_convolution_gray(gray, kernel)
    sharp_rgb = np.stack([sharp_gray]*3, axis=-1)
    out = np.dstack([sharp_rgb, alpha])
    return out

# =========================================================
# SPECIAL FEATURES: BACKGROUND REMOVAL (HSV & GRABCUT) + OTHER FILTERS
# =========================================================

def remove_background_hsv(pil_img: Image.Image,
                          lower_hsv=(0, 0, 200),
                          upper_hsv=(180, 25, 255)) -> Image.Image:
    img_bgr = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array(lower_hsv, dtype=np.uint8)
    upper = np.array(upper_hsv, dtype=np.uint8)
    mask_bg = cv2.inRange(hsv, lower, upper)
    mask_fg = cv2.bitwise_not(mask_bg)
    bgra = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = mask_fg
    img_rgba = cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGBA)
    return Image.fromarray(img_rgba)

def remove_background_grabcut(pil_img: Image.Image,
                              rect_scale: float = 0.9,
                              iters: int = 5) -> Image.Image:
    """
    Simple automatic GrabCut: use a central rectangle as probable foreground,
    then extract foreground with alpha.[web:39][web:53][web:37]
    """
    img_bgr = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
    h, w = img_bgr.shape[:2]
    rw = int(w * rect_scale)
    rh = int(h * rect_scale)
    x = (w - rw) // 2
    y = (h - rh) // 2
    rect = (x, y, rw, rh)

    mask = np.zeros((h, w), np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    cv2.grabCut(img_bgr, mask, rect, bgdModel, fgdModel, iters, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 255).astype("uint8")

    bgra = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = mask2
    img_rgba = cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGBA)
    return Image.fromarray(img_rgba)

def grayscale_filter(pil_img: Image.Image) -> Image.Image:
    return pil_img.convert("L").convert("RGBA")

def edge_detection(pil_img: Image.Image, low: int = 100, high: int = 200) -> Image.Image:
    img = np.array(pil_img.convert("RGB"))
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(img_gray, low, high)
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    edges_rgba = np.dstack([edges_rgb, np.full(edges.shape, 255, dtype=np.uint8)])
    return Image.fromarray(edges_rgba)

def invert_colors(pil_img: Image.Image) -> Image.Image:
    arr = np.array(pil_img.convert("RGBA"))
    arr[..., :3] = 255 - arr[..., :3]
    return Image.fromarray(arr)

# =========================================================
# PAGE: HOME / INTRODUCTION
# =========================================================

def page_home():
    st.title("Matrix Transformations in Image Processing")
    st.write(
        "This app demonstrates matrix operations and convolution for geometric transformations, "
        "filtering, and special features such as background removal and edge detection."
    )
    st.header("Matrix Transformations (Overview)")
    st.write(
        "- Transformations such as translation, scaling, rotation, shearing, and reflection "
        "can be represented with a 3x3 matrix (homogeneous coordinates).\n"
        "- Each pixel is mapped to a new position by multiplying the coordinate vector "
        "by the transformation matrix."
    )
    st.header("Convolution & Filtering")
    st.write(
        "- Convolution uses a small kernel that slides over the image.\n"
        "- Examples: blur (smoothing) and sharpen (enhancement) using specific kernels."
    )
    st.header("Special Features")
    st.write(
        "- Background Removal (HSV / GrabCut).\n"
        "- Grayscale, Edge Detection (Canny), Invert Colors.\n"
        "- Undo / Redo for transformations.\n"
        "- Save multiple results and download them as a ZIP file."
    )

# =========================================================
# PAGE: IMAGE PROCESSING TOOLS
# =========================================================

def page_tools():
    init_state()

    st.title("Image Processing Tools")

    st.sidebar.header("Upload Image")
    uploaded_file = st.sidebar.file_uploader("Upload image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None and st.session_state.current_image is None:
        base_img = Image.open(uploaded_file)
        st.session_state.current_image = base_img.copy()
        st.session_state.history.clear()
        st.session_state.redo_stack.clear()

    if st.session_state.current_image is None:
        st.info("Please upload an image first using the sidebar.")
        return

    current_img = st.session_state.current_image
    img_arr = pil_to_array(current_img)
    h, w, _ = img_arr.shape
    cx, cy = w / 2, h / 2

    tool = st.sidebar.selectbox(
        "Select operation",
        [
            "Translation",
            "Scaling",
            "Rotation",
            "Shearing",
            "Reflection",
            "Blur (Convolution)",
            "Sharpen (Convolution)",
            "Background Removal (HSV)",
            "Background Removal (GrabCut)",
            "Grayscale",
            "Edge Detection",
            "Invert Color"
        ]
    )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Current Image")
        st.image(current_img, use_container_width=True)

    transformed_img = None

    if tool == "Translation":
        st.sidebar.subheader("Translation Parameters")
        tx = st.sidebar.slider("tx (pixels)", -200, 200, 50)
        ty = st.sidebar.slider("ty (pixels)", -200, 200, 50)
        M = get_translation_matrix(tx, ty)
        out = apply_affine_transform(img_arr, M)
        transformed_img = array_to_pil(out)

    elif tool == "Scaling":
        st.sidebar.subheader("Scaling Parameters")
        sx = st.sidebar.slider("Scale X", 0.1, 3.0, 1.2)
        sy = st.sidebar.slider("Scale Y", 0.1, 3.0, 1.2)
        M = get_scaling_matrix(sx, sy, cx, cy)
        out = apply_affine_transform(img_arr, M)
        transformed_img = array_to_pil(out)

    elif tool == "Rotation":
        st.sidebar.subheader("Rotation Parameters")
        angle = st.sidebar.slider("Angle (deg)", -180, 180, 45)
        M = get_rotation_matrix(angle, cx, cy)
        out = apply_affine_transform(img_arr, M)
        transformed_img = array_to_pil(out)

    elif tool == "Shearing":
        st.sidebar.subheader("Shearing Parameters")
        shx = st.sidebar.slider("Shear X", -1.0, 1.0, 0.3)
        shy = st.sidebar.slider("Shear Y", -1.0, 1.0, 0.0)
        M = get_shearing_matrix(shx, shy, cx, cy)
        out = apply_affine_transform(img_arr, M)
        transformed_img = array_to_pil(out)

    elif tool == "Reflection":
        st.sidebar.subheader("Reflection Parameters")
        axis = st.sidebar.selectbox("Axis", ["Horizontal", "Vertical", "Both"])
        M = get_reflection_matrix(axis, cx, cy)
        out = apply_affine_transform(img_arr, M)
        transformed_img = array_to_pil(out)

    elif tool == "Blur (Convolution)":
        st.sidebar.subheader("Blur Parameters")
        k = st.sidebar.slider("Kernel size (odd)", 1, 9, 3, step=2)
        out = blur_filter(add_alpha_channel(img_arr), kernel_size=k)
        transformed_img = array_to_pil(out)

    elif tool == "Sharpen (Convolution)":
        st.sidebar.subheader("Sharpen Filter")
        out = sharpen_filter(add_alpha_channel(img_arr))
        transformed_img = array_to_pil(out)

    elif tool == "Background Removal (HSV)":
        st.sidebar.subheader("HSV Threshold for Background")
        h_min = st.sidebar.slider("H min", 0, 180, 0)
        s_min = st.sidebar.slider("S min", 0, 255, 0)
        v_min = st.sidebar.slider("V min", 0, 255, 200)
        h_max = st.sidebar.slider("H max", 0, 180, 180)
        s_max = st.sidebar.slider("S max", 0, 255, 25)
        v_max = st.sidebar.slider("V max", 0, 255, 255)
        transformed_img = remove_background_hsv(
            current_img,
            lower_hsv=(h_min, s_min, v_min),
            upper_hsv=(h_max, s_max, v_max)
        )

    elif tool == "Background Removal (GrabCut)":
        st.sidebar.subheader("GrabCut Parameters")
        rect_scale = st.sidebar.slider("Foreground rect scale", 0.5, 1.0, 0.9)
        iters = st.sidebar.slider("Iterations", 1, 10, 5)
        transformed_img = remove_background_grabcut(current_img, rect_scale=rect_scale, iters=iters)

    elif tool == "Grayscale":
        transformed_img = grayscale_filter(current_img)

    elif tool == "Edge Detection":
        st.sidebar.subheader("Edge Detection (Canny)")
        low = st.sidebar.slider("Low threshold", 0, 255, 100)
        high = st.sidebar.slider("High threshold", 0, 255, 200)
        transformed_img = edge_detection(current_img, low, high)

    elif tool == "Invert Color":
        transformed_img = invert_colors(current_img)

    # ----- Right column: transformed image + actions -----
    with col2:
        st.subheader("Transformed Preview")
        if transformed_img is not None:
            st.image(transformed_img, use_container_width=True)
        else:
            st.info("Adjust parameters to see a preview.")

        col_a, col_b, col_c = st.columns(3)
        if transformed_img is not None:
            with col_a:
                if st.button("Apply", type="primary"):
                    push_history(transformed_img)
            with col_b:
                if st.button("Save result"):
                    default_name = f"{tool.replace(' ', '_').lower()}_{len(st.session_state.saved_results)+1}"
                    save_current_result(default_name)
            with col_c:
                if st.button("Download this image"):
                    buf = io.BytesIO()
                    transformed_img.save(buf, format="PNG")
                    byte_im = buf.getvalue()
                    st.download_button(
                        label="Download PNG",
                        data=byte_im,
                        file_name="transformed.png",
                        mime="image/png",
                        key="download_single"
                    )

    # ----- Undo / Redo & ZIP download -----
    st.markdown("---")
    col_u, col_r, col_z = st.columns([1, 1, 2])
    with col_u:
        if st.button("Undo"):
            undo()
    with col_r:
        if st.button("Redo"):
            redo()
    with col_z:
        st.write(f"Saved results: {len(st.session_state.saved_results)}")
        if st.session_state.saved_results:
            zip_buffer = make_zip_from_results(st.session_state.saved_results)
            st.download_button(
                label="Download all saved results as ZIP",
                data=zip_buffer,
                file_name="results.zip",
                mime="application/zip",
                key="download_zip"
            )

# =========================================================
# PAGE: TEAM MEMBERS
# =========================================================

def page_team():
    st.title("Team Members")
    st.write("This page lists all team members and briefly describes their roles.")

    members = [
        {"name": "Name 1", "role": "Main Streamlit developer & UI integration.", "photo_path": "assets/nama1.jpg"},
        {"name": "Name 2", "role": "Matrix transformations implementation and debugging.", "photo_path": "assets/nama2.jpg"},
        {"name": "Name 3", "role": "Convolution filters & special features implementation.", "photo_path": "assets/nama3.jpg"},
        {"name": "Name 4", "role": "Documentation, report, and Streamlit Cloud deployment.", "photo_path": "assets/nama4.jpg"},
    ]

    cols = st.columns(2)
    for i, m in enumerate(members):
        with cols[i % 2]:
            st.subheader(m["name"])
            try:
                st.image(m["photo_path"], width=200)
            except Exception:
                st.info(f"Add a photo at path: {m['photo_path']}")
            st.write(m["role"])

    st.header("How the App Works (Briefly)")
    st.write(
        "- The user uploads an image on the Image Processing Tools page.\n"
        "- The user selects a transformation, filter, or special feature and adjusts parameters.\n"
        "- The app builds the corresponding transformation matrix or image operation, "
        "then displays the result next to the current image.\n"
        "- Undo/Redo lets the user step through previous transformations, and multiple results "
        "can be saved and downloaded together as a ZIP file."
    )

# =========================================================
# MAIN NAVIGATION
# =========================================================

def main():
    init_state()
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select page:",
        ("Home", "Image Processing Tools", "Team Members")
    )

    if page == "Home":
        page_home()
    elif page == "Image Processing Tools":
        page_tools()
    else:
        page_team()

if __name__ == "__main__":
    main()
