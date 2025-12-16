import streamlit as st
import numpy as np
import cv2
from PIL import Image
from math import radians, sin, cos
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
import tempfile
import os

# ================== LANGUAGE & THEME ==================
st.sidebar.title("⚙️ Settings")

lang = st.sidebar.selectbox("🌐 Language / Bahasa", ["English", "Indonesia"])
theme = st.sidebar.selectbox("🎨 Theme", ["💗 Pink", "🌙 Dark Mode", "☀ Light Mode"])

def apply_theme(theme_choice):
    if theme_choice == "💗 Pink":
        st.markdown(
            """
            <style>
            [data-testid="stAppViewContainer"] {
                background-color: #ffe6f2;
            }
            .stApp {
                background-color: #ffe6f2;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
    elif theme_choice == "🌙 Dark Mode":
        st.markdown(
            """
            <style>
            [data-testid="stAppViewContainer"] {
                background-color: #0e1117;
                color: white;
            }
            .stApp {
                background-color: #0e1117;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <style>
            [data-testid="stAppViewContainer"] {
                background-color: #ffffff;
            }
            .stApp {
                background-color: #ffffff;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

apply_theme(theme)

# ================== TRANSLATION DICT ==================
T = {
    "English": {
        "nav": ["🏠 Home", "🖼 Image Processing", "✂ Background Removal", "👥 Team", "📄 Report"],
        "home_title": "✨ Matrix Image Processing & Computer Vision ✨",
        "home_subtitle": "Transform images, remove background, generate PDF — All in one place 🎓",
        "home_hint": "Choose theme on the sidebar 🎨 and start exploring features!",
        "home_box1": "🛠 Transform Images with matrix math",
        "home_box2": "🎨 Edit & Filter using convolution",
        "home_box3": "📄 Download PDF Report automatically",
        "home_foot": "Use the sidebar to navigate ➡",

        "img_header": "🖼 Image Processing Tools",
        "img_upload": "Upload Image",
        "img_tool": "Select Tool",
        "img_tool_opts": ["Matrix Transform", "Convolution Filter"],
        "transform_label": "Transformation Type",
        "transform_opts": ["Translation", "Scaling", "Rotation", "Shearing", "Reflection"],
        "translation_tx": "Move X (pixels)",
        "translation_ty": "Move Y (pixels)",
        "scaling_sx": "Scale X",
        "scaling_sy": "Scale Y",
        "rotation_ang": "Angle (degrees)",
        "shear_x": "Shear X factor",
        "shear_y": "Shear Y factor",
        "reflection_axis": "Reflection Axis",
        "reflection_opts": ["Horizontal (x-axis)", "Vertical (y-axis)"],
        "btn_apply": "Apply Transformation",
        "conv_filter": "Select Filter",
        "conv_opts": ["Blur", "Sharpen", "Edge Detection", "Emboss"],
        "kernel_size": "Kernel Size",

        "bg_header": "✂ Background Removal",
        "bg_upload": "Upload image for background removal",
        "bg_x": "ROI X position",
        "bg_y": "ROI Y position",
        "bg_w": "ROI Width",
        "bg_h": "ROI Height",
        "bg_btn": "Remove Background",
        "bg_save": "Save Result",

        "team_title": "👥 Our Team",
        "team_subtitle": "Meet our awesome team members!",
        "team_sid": "Student ID:",
        "team_role": "Role:",
        "team_group": "Group:",
        "team_contribution": "Contribution:",

        "report_header": "📄 Generate PDF Report",
        "report_title": "Report Title",
        "report_default": "Matrix Image Processing Report",
        "report_btn": "Create PDF Report",
        "report_error": "❗ Please process an image first before generating report",
        "report_download": "📥 Download PDF Report",
        "report_success": "✅ PDF Created Successfully!",

        "orig_caption": "Original Image",
        "transformed_caption": "Transformed Result 🎉",
        "filtered_caption": "Filtered Result 🎉",
        "bg_removed_caption": "Background Removed 🎉",
        "roi_preview": "ROI Selection Preview",
        
        "download_original": "Download Original",
        "download_processed": "Download Processed",
    },
    "Indonesia": {
        "nav": ["🏠 Beranda", "🖼 Pemrosesan Gambar", "✂ Hapus Background", "👥 Tim", "📄 Laporan"],
        "home_title": "✨ Pemrosesan Citra Matriks & Computer Vision ✨",
        "home_subtitle": "Transformasi gambar, hapus background, buat PDF — Semua dalam satu aplikasi 🎓",
        "home_hint": "Pilih tema di sidebar 🎨 dan mulai eksplor fitur!",
        "home_box1": "🛠 Transformasi Gambar dengan matriks",
        "home_box2": "🎨 Edit & Filter dengan konvolusi",
        "home_box3": "📄 Unduh Laporan PDF secara otomatis",
        "home_foot": "Gunakan sidebar untuk navigasi ➡",

        "img_header": "🖼 Alat Pemrosesan Gambar",
        "img_upload": "Unggah Gambar",
        "img_tool": "Pilih Alat",
        "img_tool_opts": ["Transformasi Matriks", "Filter Konvolusi"],
        "transform_label": "Jenis Transformasi",
        "transform_opts": ["Translasi", "Skala", "Rotasi", "Shearing", "Refleksi"],
        "translation_tx": "Geser X (piksel)",
        "translation_ty": "Geser Y (piksel)",
        "scaling_sx": "Skala X",
        "scaling_sy": "Skala Y",
        "rotation_ang": "Sudut (derajat)",
        "shear_x": "Faktor Shear X",
        "shear_y": "Faktor Shear Y",
        "reflection_axis": "Sumbu Refleksi",
        "reflection_opts": ["Horizontal (sumbu-x)", "Vertikal (sumbu-y)"],
        "btn_apply": "Terapkan Transformasi",
        "conv_filter": "Pilih Filter",
        "conv_opts": ["Blur", "Tajamkan", "Deteksi Tepi", "Emboss"],
        "kernel_size": "Ukuran Kernel",

        "bg_header": "✂ Hapus Background",
        "bg_upload": "Unggah gambar untuk hapus background",
        "bg_x": "Posisi X ROI",
        "bg_y": "Posisi Y ROI",
        "bg_w": "Lebar ROI",
        "bg_h": "Tinggi ROI",
        "bg_btn": "Hapus Background",
        "bg_save": "Simpan Hasil",

        "team_title": "👥 Tim Kami",
        "team_subtitle": "Kenalan dengan anggota tim kami!",
        "team_sid": "NIM:",
        "team_role": "Peran:",
        "team_group": "Kelompok:",
        "team_contribution": "Kontribusi:",

        "report_header": "📄 Buat Laporan PDF",
        "report_title": "Judul Laporan",
        "report_default": "Laporan Pemrosesan Citra Matriks",
        "report_btn": "Buat Laporan PDF",
        "report_error": "❗ Harap proses gambar terlebih dahulu sebelum membuat laporan",
        "report_download": "📥 Unduh Laporan PDF",
        "report_success": "✅ PDF Berhasil Dibuat!",

        "orig_caption": "Gambar Asli",
        "transformed_caption": "Hasil Transformasi 🎉",
        "filtered_caption": "Hasil Filter 🎉",
        "bg_removed_caption": "Background Terhapus 🎉",
        "roi_preview": "Preview Seleksi ROI",
        
        "download_original": "Unduh Asli",
        "download_processed": "Unduh Hasil",
    }
}

t = T[lang]

# ================== NAVIGATION ==================
st.sidebar.markdown("---")
st.sidebar.subheader("📍 Navigation / Navigasi")
page = st.sidebar.radio("Go to:", t["nav"], label_visibility="collapsed")

# ================== SESSION STATE ==================
if "original_image" not in st.session_state:
    st.session_state.original_image = None
if "processed_image" not in st.session_state:
    st.session_state.processed_image = None
if "bg_removed_image" not in st.session_state:
    st.session_state.bg_removed_image = None
if "transformation_params" not in st.session_state:
    st.session_state.transformation_params = {}
if "filter_params" not in st.session_state:
    st.session_state.filter_params = {}

# ================== UTILITY FUNCTIONS ==================
def safe_display_image(image_path, size=(150, 150)):
    """Safely display image with error handling"""
    try:
        if os.path.exists(image_path):
            img = Image.open(image_path)
            img = img.resize(size)
            return img
        else:
            # Return a placeholder if image doesn't exist
            return Image.new('RGB', size, color='lightgray')
    except Exception as e:
        st.error(f"Error loading image: {str(e)}")
        return Image.new('RGB', size, color='lightgray')

def create_download_button(image, filename, label):
    """Create a download button for images"""
    from io import BytesIO
    
    if image is not None:
        # Convert PIL Image to bytes
        buf = BytesIO()
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        image.save(buf, format="PNG")
        byte_im = buf.getvalue()
        
        return st.download_button(
            label=label,
            data=byte_im,
            file_name=filename,
            mime="image/png"
        )
    return None

# ================== MATRIX TRANSFORMATION FUNCTIONS ==================
def translation_matrix(tx, ty):
    """Create translation matrix"""
    return np.float32([[1, 0, tx], [0, 1, ty]])

def scaling_matrix(sx, sy):
    """Create scaling matrix"""
    return np.float32([[sx, 0, 0], [0, sy, 0]])

def rotation_matrix(angle, cx=None, cy=None):
    """Create rotation matrix"""
    r = radians(angle)
    if cx is None or cy is None:
        return np.float32([[cos(r), -sin(r), 0], [sin(r), cos(r), 0]])
    else:
        return np.float32([
            [cos(r), -sin(r), cx - cos(r) * cx + sin(r) * cy],
            [sin(r), cos(r), cy - sin(r) * cx - cos(r) * cy]
        ])

def shearing_matrix(shx, shy):
    """Create shearing matrix"""
    return np.float32([[1, shx, 0], [shy, 1, 0]])

def reflection_matrix(axis):
    """Create reflection matrix"""
    if "Horizontal" in axis or "x" in axis.lower():
        return np.float32([[1, 0, 0], [0, -1, 0]])
    else:
        return np.float32([[-1, 0, 0], [0, 1, 0]])

def apply_affine_transform(img, M):
    """Apply affine transformation to image"""
    h, w = img.shape[:2]
    
    # Calculate output dimensions
    corners = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    ones = np.ones((4, 1), dtype=np.float32)
    corners_homogeneous = np.hstack([corners, ones])
    
    # Apply transformation
    M_homogeneous = np.vstack([M, [0, 0, 1]])
    transformed_corners = (M_homogeneous @ corners_homogeneous.T).T
    
    # Calculate new dimensions
    min_x, max_x = int(transformed_corners[:, 0].min()), int(transformed_corners[:, 0].max())
    min_y, max_y = int(transformed_corners[:, 1].min()), int(transformed_corners[:, 1].max())
    new_w, new_h = max_x - min_x, max_y - min_y
    
    # Adjust matrix for positive coordinates
    shift = np.float32([[1, 0, -min_x], [0, 1, -min_y]])
    shift_3x3 = np.vstack([shift, [0, 0, 1]])
    combined = shift_3x3 @ M_homogeneous
    M_final = combined[:2, :]
    
    # Apply warp affine
    result = cv2.warpAffine(img, M_final, (new_w, new_h))
    return result

# ================== CONVOLUTION FILTERS ==================
def get_convolution_kernel(filter_name, kernel_size=3):
    """Get convolution kernel based on filter name"""
    if kernel_size == 3:
        kernels = {
            "Blur": np.ones((3, 3), dtype=np.float32) / 9.0,
            "Sharpen": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32),
            "Edge Detection": np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float32),
            "Emboss": np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], dtype=np.float32),
            "Tajamkan": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32),
            "Deteksi Tepi": np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float32),
        }
    else:  # 5x5 kernel
        kernels = {
            "Blur": np.ones((5, 5), dtype=np.float32) / 25.0,
            "Sharpen": np.array([
                [-1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1],
                [-1, -1, 25, -1, -1],
                [-1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1]
            ], dtype=np.float32),
            "Edge Detection": np.array([
                [-1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1],
                [-1, -1, 24, -1, -1],
                [-1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1]
            ], dtype=np.float32),
        }
    
    return kernels.get(filter_name, kernels["Blur"])

# ================== BACKGROUND REMOVAL ==================
def remove_background_grabcut(image_array, x, y, w, h, iterations=5):
    """Remove background using GrabCut algorithm"""
    try:
        img = image_array.copy()
        height, width = img.shape[:2]
        
        # Ensure ROI is within image bounds
        x = max(0, min(x, width - 1))
        y = max(0, min(y, height - 1))
        w = max(10, min(w, width - x))
        h = max(10, min(h, height - y))
        
        # Initialize mask
        mask = np.zeros((height, width), np.uint8)
        
        # Background and foreground models
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        
        # Apply GrabCut with rectangle initialization
        cv2.grabCut(img, mask, (x, y, w, h), bgd_model, fgd_model, iterations, cv2.GC_INIT_WITH_RECT)
        
        # Create binary mask: 0 for background, 1 for foreground
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        
        # Apply mask to image
        result = img * mask2[:, :, np.newaxis]
        
        # Create transparent background (RGBA)
        rgba = cv2.cvtColor(result, cv2.COLOR_RGB2RGBA)
        rgba[:, :, 3] = mask2 * 255
        
        return rgba
    except Exception as e:
        st.error(f"Error in background removal: {str(e)}")
        return None

# ================== PDF REPORT GENERATION ==================
def generate_pdf_report(title, original_img, processed_img, params=None):
    """Generate PDF report with images and parameters"""
    try:
        # Create temporary file for PDF
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        temp_path = temp_file.name
        temp_file.close()
        
        # Create PDF canvas
        c = canvas.Canvas(temp_path, pagesize=A4)
        width, height = A4
        
        # Set title
        c.setFont("Helvetica-Bold", 20)
        c.drawString(50, height - 50, title)
        
        # Add timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        c.setFont("Helvetica", 10)
        c.drawString(50, height - 70, f"Generated on: {timestamp}")
        
        # Save temporary images
        temp_orig = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        temp_proc = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        
        original_img.save(temp_orig.name)
        processed_img.save(temp_proc.name)
        
        # Add images to PDF
        c.drawString(50, height - 100, "Original Image:")
        try:
            c.drawImage(temp_orig.name, 50, height - 300, 200, 200)
        except:
            pass
        
        c.drawString(300, height - 100, "Processed Image:")
        try:
            c.drawImage(temp_proc.name, 300, height - 300, 200, 200)
        except:
            pass
        
        # Add parameters if available
        if params:
            c.drawString(50, height - 320, "Processing Parameters:")
            y_pos = height - 340
            for key, value in params.items():
                c.drawString(60, y_pos, f"- {key}: {value}")
                y_pos -= 20
        
        # Add footer
        c.setFont("Helvetica-Oblique", 8)
        c.drawString(50, 30, "Generated by Matrix Image Processing App")
        
        c.save()
        
        # Clean up temporary image files
        os.unlink(temp_orig.name)
        os.unlink(temp_proc.name)
        
        return temp_path
    except Exception as e:
        st.error(f"Error generating PDF: {str(e)}")
        return None

# ================== HOME PAGE ==================
if page == t["nav"][0]:
    st.markdown(f"<h1 style='text-align: center;'>{t['home_title']}</h1>", unsafe_allow_html=True)
    st.markdown(f"<h3 style='text-align: center;'>{t['home_subtitle']}</h3>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align: center;'>{t['home_hint']}</p>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Features section
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"### 📊 {t['home_box1'].split(' ')[-1]}")
        st.info(t['home_box1'])
    
    with col2:
        st.markdown(f"### 🎯 {t['home_box2'].split(' ')[-1]}")
        st.info(t['home_box2'])
    
    with col3:
        st.markdown(f"### 📈 {t['home_box3'].split(' ')[-1]}")
        st.info(t['home_box3'])
    
    st.markdown("---")
    
    # Quick start guide
    st.markdown("### 🚀 Quick Start Guide")
    st.markdown("""
    1. **Upload an image** in the Image Processing section
    2. **Choose a tool** - Matrix Transform or Convolution Filter
    3. **Adjust parameters** using the sliders
    4. **Apply the transformation** and see the result
    5. **Generate a PDF report** of your work
    """)
    
    st.success(t['home_foot'])

# ================== IMAGE PROCESSING PAGE ==================
elif page == t["nav"][1]:
    st.header(t["img_header"])
    
    # File uploader
    uploaded_file = st.file_uploader(
        t["img_upload"], 
        type=['png', 'jpg', 'jpeg', 'bmp'],
        help="Upload an image file (PNG, JPG, JPEG, BMP)"
    )
    
    if uploaded_file is not None:
        try:
            # Load and display original image
            original_image = Image.open(uploaded_file).convert('RGB')
            original_array = np.array(original_image)
            
            st.session_state.original_image = original_image
            
            col1, col2 = st.columns(2)
            with col1:
                st.image(original_image, caption=t["orig_caption"], use_column_width=True)
                
                # Download button for original
                create_download_button(
                    original_image, 
                    "original_image.png", 
                    t["download_original"]
                )
            
            # Tool selection
            tool_option = st.selectbox(
                t["img_tool"],
                t["img_tool_opts"],
                help="Choose between matrix transformations or convolution filters"
            )
            
            # MATRIX TRANSFORMATIONS
            if tool_option == t["img_tool_opts"][0]:
                st.subheader("🔧 Matrix Transformation Settings")
                
                transform_type = st.selectbox(
                    t["transform_label"],
                    t["transform_opts"]
                )
                
                # Store parameters
                params = {}
                
                if transform_type in ["Translation", "Translasi"]:
                    col1, col2 = st.columns(2)
                    with col1:
                        tx = st.slider(t["translation_tx"], -300, 300, 50)
                        params["Translation X"] = tx
                    with col2:
                        ty = st.slider(t["translation_ty"], -300, 300, 30)
                        params["Translation Y"] = ty
                    
                    M = translation_matrix(tx, ty)
                
                elif transform_type in ["Scaling", "Skala"]:
                    col1, col2 = st.columns(2)
                    with col1:
                        sx = st.slider(t["scaling_sx"], 0.1, 5.0, 1.2, 0.1)
                        params["Scale X"] = sx
                    with col2:
                        sy = st.slider(t["scaling_sy"], 0.1, 5.0, 1.2, 0.1)
                        params["Scale Y"] = sy
                    
                    M = scaling_matrix(sx, sy)
                
                elif transform_type in ["Rotation", "Rotasi"]:
                    angle = st.slider(t["rotation_ang"], -180, 180, 45)
                    params["Rotation Angle"] = f"{angle}°"
                    
                    # Get image center
                    h, w = original_array.shape[:2]
                    M = rotation_matrix(angle, w/2, h/2)
                
                elif transform_type in ["Shearing", "Shearing"]:
                    col1, col2 = st.columns(2)
                    with col1:
                        shx = st.slider(t["shear_x"], -1.0, 1.0, 0.3, 0.1)
                        params["Shear X"] = shx
                    with col2:
                        shy = st.slider(t["shear_y"], -1.0, 1.0, 0.0, 0.1)
                        params["Shear Y"] = shy
                    
                    M = shearing_matrix(shx, shy)
                
                else:  # Reflection
                    axis = st.selectbox(
                        t["reflection_axis"],
                        t["reflection_opts"]
                    )
                    params["Reflection Axis"] = axis
                    
                    M = reflection_matrix(axis)
                
                # Apply transformation button
                if st.button(t["btn_apply"], type="primary"):
                    with st.spinner("Applying transformation..."):
                        try:
                            transformed_array = apply_affine_transform(original_array, M)
                            transformed_image = Image.fromarray(transformed_array)
                            st.session_state.processed_image = transformed_image
                            st.session_state.transformation_params = params
                            
                            with col2:
                                st.image(
                                    transformed_image, 
                                    caption=t["transformed_caption"],
                                    use_column_width=True
                                )
                                
                                # Download button for processed image
                                create_download_button(
                                    transformed_image,
                                    "transformed_image.png",
                                    t["download_processed"]
                                )
                            
                            # Show matrix
                            st.subheader("📐 Transformation Matrix")
                            st.code(f"""
                            [[{M[0,0]:.3f}, {M[0,1]:.3f}, {M[0,2]:.3f}],
                             [{M[1,0]:.3f}, {M[1,1]:.3f}, {M[1,2]:.3f}]]
                            """)
                            
                        except Exception as e:
                            st.error(f"Error applying transformation: {str(e)}")
            
            # CONVOLUTION FILTERS
            else:
                st.subheader("🎨 Convolution Filter Settings")
                
                # Filter selection
                filter_name = st.selectbox(
                    t["conv_filter"],
                    t["conv_opts"]
                )
                
                # Kernel size selection
                kernel_size = st.selectbox(
                    t["kernel_size"],
                    [3, 5],
                    format_func=lambda x: f"{x}x{x}"
                )
                
                params = {
                    "Filter": filter_name,
                    "Kernel Size": f"{kernel_size}x{kernel_size}"
                }
                
                # Get kernel
                kernel = get_convolution_kernel(filter_name, kernel_size)
                
                # Apply filter button
                if st.button(t["btn_apply"], type="primary"):
                    with st.spinner(f"Applying {filter_name} filter..."):
                        try:
                            # Convert PIL to OpenCV format
                            if original_array.dtype != np.uint8:
                                original_array = original_array.astype(np.uint8)
                            
                            # Apply convolution
                            filtered_array = cv2.filter2D(original_array, -1, kernel)
                            filtered_image = Image.fromarray(filtered_array)
                            st.session_state.processed_image = filtered_image
                            st.session_state.filter_params = params
                            
                            with col2:
                                st.image(
                                    filtered_image,
                                    caption=t["filtered_caption"],
                                    use_column_width=True
                                )
                                
                                # Download button
                                create_download_button(
                                    filtered_image,
                                    "filtered_image.png",
                                    t["download_processed"]
                                )
                            
                            # Show kernel
                            st.subheader("🔢 Convolution Kernel")
                            st.write(kernel)
                            
                        except Exception as e:
                            st.error(f"Error applying filter: {str(e)}")
        
        except Exception as e:
            st.error(f"Error loading image: {str(e)}")
    
    else:
        st.info("👆 Please upload an image to get started!")

# ================== BACKGROUND REMOVAL PAGE ==================
elif page == t["nav"][2]:
    st.header(t["bg_header"])
    
    # File uploader
    bg_file = st.file_uploader(
        t["bg_upload"],
        type=['png', 'jpg', 'jpeg'],
        key="bg_removal_uploader"
    )
    
    if bg_file is not None:
        try:
            # Load image
            bg_image = Image.open(bg_file).convert('RGB')
            bg_array = np.array(bg_image)
            h, w = bg_array.shape[:2]
            
            st.session_state.original_image = bg_image
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(bg_image, caption=t["orig_caption"], use_column_width=True)
            
            with col2:
                st.subheader("🎯 Select Region of Interest (ROI)")
                
                # Default ROI values (centered)
                default_x = max(0, int(w * 0.1))
                default_y = max(0, int(h * 0.1))
                default_w = min(int(w * 0.8), w - default_x)
                default_h = min(int(h * 0.8), h - default_y)
                
                # ROI sliders
                x = st.slider(t["bg_x"], 0, w-1, default_x, key="roi_x")
                y = st.slider(t["bg_y"], 0, h-1, default_y, key="roi_y")
                roi_w = st.slider(t["bg_w"], 10, w-x, default_w, key="roi_w")
                roi_h = st.slider(t["bg_h"], 10, h-y, default_h, key="roi_h")
                
                # Show ROI preview
                preview_img = bg_array.copy()
                cv2.rectangle(preview_img, (x, y), (x+roi_w, y+roi_h), (0, 255, 0), 3)
                st.image(preview_img, caption=t["roi_preview"], use_column_width=True)
            
            # Remove background button
            if st.button(t["bg_btn"], type="primary"):
                with st.spinner("Removing background..."):
                    try:
                        result_array = remove_background_grabcut(bg_array, x, y, roi_w, roi_h)
                        
                        if result_array is not None:
                            result_image = Image.fromarray(result_array)
                            st.session_state.processed_image = result_image
                            st.session_state.bg_removed_image = result_image
                            
                            st.success("✅ Background removed successfully!")
                            
                            # Display result
                            st.subheader("📸 Result")
                            result_col1, result_col2 = st.columns(2)
                            
                            with result_col1:
                                st.image(bg_image, caption="Original", use_column_width=True)
                            
                            with result_col2:
                                st.image(result_image, caption=t["bg_removed_caption"], use_column_width=True)
                            
                            # Save button
                            if st.button(t["bg_save"]):
                                if result_image.mode == 'RGBA':
                                    result_image = result_image.convert('RGB')
                                
                                buf = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
                                result_image.save(buf.name)
                                
                                with open(buf.name, 'rb') as f:
                                    st.download_button(
                                        label="💾 Download Result",
                                        data=f,
                                        file_name="background_removed.png",
                                        mime="image/png"
                                    )
                                
                                os.unlink(buf.name)
                    
                    except Exception as e:
                        st.error(f"Error removing background: {str(e)}")
        
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
    
    else:
        st.info("👆 Upload an image to remove its background!")

# ================== TEAM PAGE ==================
elif page == t["nav"][3]:
    st.title(t["team_title"])
    st.markdown(f"<p style='text-align: center;'>{t['team_subtitle']}</p>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Team members data
    team_members = [
        {
            "name": "ELIZABETH KURNIAWAN",
            "sid": "04202400001",
            "role": "Team Leader",
            "group": "5",
            "contribution": "Project Manager, Geometric Transformations Module",
            "image_path": "images/Elizabeth.jpg"
        },
        {
            "name": "REGINA VINTA AMANULLAH",
            "sid": "04202400133",
            "role": "Member",
            "group": "5",
            "contribution": "Image Filtering Module, UI/UX Design",
            "image_path": "images/Regina.jpg"
        },
        {
            "name": "BILL CHRISTIAN",
            "sid": "04202400058",
            "role": "Member",
            "group": "5",
            "contribution": "Background Removal Module, Image Upload & Download",
            "image_path": "images/Bill.jpg"
        },
        {
            "name": "PUTRI LASRIDA MALAU",
            "sid": "04202400132",
            "role": "Member",
            "group": "5",
            "contribution": "Histogram Module, Image Processing Functions",
            "image_path": "images/Putri.jpg"
        }
    ]
    
    # Display team members in a grid
    cols = st.columns(2)
    
    for idx, member in enumerate(team_members):
        with cols[idx % 2]:
            with st.container():
                # Create card-like container
                st.markdown(f"""
                <div style='padding: 20px; border-radius: 10px; 
                    background-color: {'#f0f2f6' if theme != '🌙 Dark Mode' else '#262730'}; 
                    margin: 10px 0;'>
                """, unsafe_allow_html=True)
                
                # Display member image or placeholder
                try:
                    member_img = safe_display_image(member["image_path"], (200, 200))
                    st.image(member_img, use_column_width=True)
                except:
                    # Placeholder if image not found
                    st.image(Image.new('RGB', (200, 200), color='gray'), use_column_width=True)
                
                # Member info
                st.markdown(f"### {member['name']}")
                st.markdown(f"**{t['team_sid']}** {member['sid']}")
                st.markdown(f"**{t['team_role']}** {member['role']}")
                st.markdown(f"**{t['team_group']}** {member['group']}")
                st.markdown(f"**{t['team_contribution']}** {member['contribution']}")
                
                st.markdown("</div>", unsafe_allow_html=True)

# ================== REPORT PAGE ==================
elif page == t["nav"][4]:
    st.header(t["report_header"])
    
    # Check if we have images to report
    if st.session_state.original_image is None or st.session_state.processed_image is None:
        st.warning(t["report_error"])
        
        # Show what's missing
        col1, col2 = st.columns(2)
        with col1:
            st.write("Original Image:")
            st.write("✅ Available" if st.session_state.original_image else "❌ Not available")
        
        with col2:
            st.write("Processed Image:")
            st.write("✅ Available" if st.session_state.processed_image else "❌ Not available")
        
        st.info("Please go to Image Processing or Background Removal section first to process an image.")
    
    else:
        # Display preview
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(
                st.session_state.original_image,
                caption="Original Image",
                use_column_width=True
            )
        
        with col2:
            st.image(
                st.session_state.processed_image,
                caption="Processed Image",
                use_column_width=True
            )
        
        # Report title input
        report_title = st.text_input(
            t["report_title"],
            value=t["report_default"]
        )
        
        # Collect parameters for report
        report_params = {}
        
        if st.session_state.transformation_params:
            report_params.update(st.session_state.transformation_params)
            report_params["Transformation Type"] = "Matrix Transformation"
        
        if st.session_state.filter_params:
            report_params.update(st.session_state.filter_params)
            report_params["Filter Type"] = "Convolution Filter"
        
        # Add general info
        report_params["Language"] = lang
        report_params["Theme"] = theme
        report_params["Original Image Size"] = f"{st.session_state.original_image.size[0]}x{st.session_state.original_image.size[1]}"
        
        # Generate PDF button
        if st.button(t["report_btn"], type="primary"):
            with st.spinner("Generating PDF report..."):
                try:
                    pdf_path = generate_pdf_report(
                        report_title,
                        st.session_state.original_image,
                        st.session_state.processed_image,
                        report_params
                    )
                    
                    if pdf_path:
                        # Read PDF file
                        with open(pdf_path, 'rb') as f:
                            pdf_bytes = f.read()
                        
                        # Create download button
                        st.download_button(
                            label=t["report_download"],
                            data=pdf_bytes,
                            file_name=f"{report_title.replace(' ', '_')}.pdf",
                            mime="application/pdf"
                        )
                        
                        st.success(t["report_success"])
                        
                        # Clean up temporary file
                        os.unlink(pdf_path)
                    
                except Exception as e:
                    st.error(f"Error generating report: {str(e)}")

# ================== FOOTER ==================
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 App Info")
st.sidebar.info("**Matrix Image Processing App**\n\nVersion 1.0.0\n\nGroup 5 - Computer Vision Class")

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Current Settings")
st.sidebar.write(f"**Language:** {lang}")
st.sidebar.write(f"**Theme:** {theme}")
st.sidebar.write(f"**Page:** {page}")

# Add a reset button in sidebar
if st.sidebar.button("🔄 Reset Session"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

# Add CSS for better styling
st.markdown("""
<style>
    .stButton > button {
        width: 100%;
        margin-top: 10px;
    }
    .stDownloadButton > button {
        width: 100%;
        margin-top: 10px;
    }
    div[data-testid="stImage"] {
        text-align: center;
    }
    .stAlert {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)
