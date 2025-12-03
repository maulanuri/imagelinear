import streamlit as st
import numpy as np
from PIL import Image, ImageEnhance
import io
import cv2  # pip install opencv-python
import zipfile
import json
import pandas as pd  # pip install pandas
from reportlab.pdfgen import canvas  # pip install reportlab

st.set_page_config(page_title="Matrix Image Processing", layout="wide")

# =========================================================
# BASE CSS (layout ala paste.txt)
# =========================================================

BASE_CSS = """
<style>
body {
  background: radial-gradient(circle at top left, #ecfdf5 0, #bbf7d0 25%, #ecfdf5 60%, #ffffff 100%);
}
.main .block-container {
  padding-top: 0.8rem;
  max-width: 1200px;
}
.hero-card {
  background: linear-gradient(135deg, #ecfdf5, #d1fae5);
  border-radius: 0.8rem;
  border: 1px solid rgba(16,185,129,0.25);
  padding: 0.9rem 1.1rem;
  box-shadow: 0 2px 8px rgba(16,185,129,0.12);
}
.decorative-divider {
  height: 1px;
  margin: 0.6rem 0 1.0rem 0;
  background: linear-gradient(to right, transparent, #6ee7b7, transparent);
}
.main-card {
  background: #ffffff;
  border-radius: 0.8rem;
  padding: 1.0rem 1.1rem 1.2rem 1.1rem;
  border: 1px solid rgba(148,163,184,0.45);
  box-shadow: 0 4px 18px rgba(15,118,110,0.20);
}
.upload-card {
  border-radius: 0.7rem;
  border: 1px dashed rgba(148,163,184,0.70);
  padding: 0.6rem 0.75rem;
  background: rgba(249,250,251,0.85);
}
.helper-text {
  font-size: 0.76rem;
  color: #6b7280;
  margin-top: 0.05rem;
}
.summary-badge {
  display: inline-flex;
  align-items: center;
  gap: 0.25rem;
  padding: 0.20rem 0.55rem;
  border-radius: 999px;
  background: rgba(16,185,129,0.08);
  border: 1px solid rgba(16,185,129,0.25);
  font-size: 0.78rem;
  color: #065f46;
  margin-right: 0.35rem;
}
.summary-dot {
  width: 0.42rem;
  height: 0.42rem;
  border-radius: 999px;
  background: linear-gradient(135deg, #10b981, #22c55e);
}
</style>
"""

DARK_CSS = """
<style>
body {
  background: radial-gradient(circle at top left, #020617 0, #0f172a 45%, #020617 100%);
}
.main-card, .hero-card {
  background: linear-gradient(135deg, #020617, #0f172a);
  border-color: rgba(148,163,184,0.65);
  box-shadow: 0 8px 24px rgba(0,0,0,0.6);
}
.upload-card {
  background: rgba(15,23,42,0.95);
  border-color: rgba(148,163,184,0.80);
}
.summary-badge {
  background: rgba(22,163,74,0.2);
  border-color: rgba(34,197,94,0.7);
  color: #bbf7d0;
}
.helper-text {
  color: #9ca3af;
}
</style>
"""

# =========================================================
# MULTI-LANGUAGE TEXTS (FULL DICTIONARY)
# =========================================================

LANG_TEXT = {
    "en": {
        "app_title": "Matrix Transformations in Image Processing – Single Page App",
        "home_title": "📘 Home / Introduction",
        "overview": "Overview",
        "conv_title": "Convolution & Filtering",
        "download_center": "Download Center",
        "team_title": "👥 Team Members + Photo Editing Controls (click to open)",
        "team_heading": "Team Members",
        "team_how_title": "How the App Works (Briefly)",
        "home_intro": (
            "This app demonstrates matrix operations and convolution for geometric "
            "transformations, filtering, and special features such as background "
            "removal and edge detection."
        ),
        "overview_text": (
            "Transformations such as translation, scaling, rotation, shearing, and reflection "
            "can be represented with a 3×3 homogeneous matrix. Each pixel is mapped to a new "
            "position by multiplying its coordinate vector by the transformation matrix."
        ),
        "conv_text": (
            "Convolution uses a small kernel that slides over the image. Blur and sharpen "
            "can be built using simple kernels computed by manual convolution."
        ),
        "team_how_text": (
            "- Upload images in the team members section.\n"
            "- Choose a transformation or filter and adjust parameters.\n"
            "- The app computes the corresponding matrix or convolution and shows the result.\n"
            "- Use Undo/Redo and save multiple results to download them as a ZIP."
        ),
        "num_members": "Number of members",
        "member_label": "Member",
        "role_placeholder": "Write role / contribution here.",
        "edit_member_prompt": "Select a member photo to edit",
        "need_member_image": "Please upload at least one member photo to use the editing controls.",
        "tools_title": "🎨 Photo Editing Controls",
        "controls_title": "Controls",
        "controls_hint": "Select an operation and adjust parameters to see the effect.",
        "operation_label": "Operation",
        "current_image": "Current Image",
        "preview_image": "Transformed Preview",
        "preview_hint": "Adjust parameters to see a preview.",
        "btn_apply": "Apply",
        "btn_save": "Save result",
        "btn_download": "Download PNG",
        "undo": "Undo",
        "redo": "Redo",
        "saved_results": "Saved results",
        "btn_download_zip": "Download all results as ZIP",
        "op_translation": "Translation",
        "op_scaling": "Scaling",
        "op_rotation": "Rotation",
        "op_shearing": "Shearing",
        "op_reflection": "Reflection",
        "op_blur": "Blur (Convolution)",
        "op_sharpen": "Sharpen (Convolution)",
        "op_hsv": "Background Removal (HSV)",
        "op_grabcut": "Background Removal (GrabCut)",
        "op_gray": "Grayscale",
        "op_edge": "Edge Detection",
        "op_invert": "Invert Color",
        "translation_params": "Translation Parameters",
        "scaling_params": "Scaling Parameters",
        "rotation_params": "Rotation Parameters",
        "shearing_params": "Shearing Parameters",
        "reflection_params": "Reflection Parameters",
        "blur_params": "Blur Parameters",
        "sharpen_params": "Sharpen Filter",
        "hsv_params": "HSV Threshold for Background",
        "grabcut_params": "GrabCut Parameters",
        "edge_params": "Edge Detection (Canny)",
        "tx_label": "tx (pixels)",
        "ty_label": "ty (pixels)",
        "scale_x": "Scale X",
        "scale_y": "Scale Y",
        "angle_label": "Angle (degrees)",
        "shear_x": "Shear X",
        "shear_y": "Shear Y",
        "axis_label": "Reflection Axis",
        "axis_horizontal": "Horizontal",
        "axis_vertical": "Vertical",
        "axis_both": "Both",
        "kernel_size": "Kernel size (odd)",
        "h_min": "H min",
        "s_min": "S min",
        "v_min": "V min",
        "h_max": "H max",
        "s_max": "S max",
        "v_max": "V max",
        "rect_scale": "Foreground rectangle scale",
        "iterations": "Iterations",
        "low_thresh": "Low threshold",
        "high_thresh": "High threshold",
        "dl_current_png": "Download current image (PNG)",
        "dl_current_jpg": "Download current image (JPG)",
        "dl_meta_json": "Download metadata (JSON)",
        "dl_meta_csv": "Download metadata (CSV)",
        "dl_report_pdf": "Download report (PDF)",
        "report_title": "Image Processing Report",
        "meta_width": "Width",
        "meta_height": "Height",
        "meta_mode": "Mode",
        "meta_saved": "Saved results",
        "lang_en": "EN",
        "lang_id": "ID",
        "lang_zh": "中文",
        "lang_ja": "日本語",
        "lang_ko": "한국어",
        "lang_ar": "العربية",
        "top_language": "Language",
        "top_dark_mode": "Dark mode",
    },
    "id": {
        "app_title": "Transformasi Matriks pada Pengolahan Citra – Aplikasi Satu Halaman",
        "home_title": "📘 Beranda / Pendahuluan",
        "overview": "Ikhtisar",
        "conv_title": "Konvolusi & Filtering",
        "download_center": "Pusat Unduhan",
        "team_title": "👥 Anggota Tim + Kontrol Edit Foto (klik untuk buka)",
        "team_heading": "Anggota Tim",
        "team_how_title": "Cara Kerja Aplikasi (Singkat)",
        "home_intro": (
            "Aplikasi ini mendemonstrasikan operasi matriks dan konvolusi "
            "untuk transformasi geometri, filtering, dan fitur khusus seperti "
            "penghapusan background dan deteksi tepi."
        ),
        "overview_text": (
            "Transformasi seperti translasi, skala, rotasi, shearing, dan refleksi "
            "dapat direpresentasikan dengan matriks 3×3 (koordinat homogen). "
            "Setiap piksel dipetakan ke posisi baru dengan mengalikan vektor koordinat "
            "dengan matriks transformasi."
        ),
        "conv_text": (
            "Konvolusi menggunakan kernel kecil yang digeser ke seluruh citra. "
            "Blur dan sharpen dapat dibangun dengan kernel sederhana yang dihitung secara manual."
        ),
        "team_how_text": (
            "- Pengguna mengunggah gambar di bagian anggota tim.\n"
            "- Pengguna memilih transformasi / filter dan mengatur parameter.\n"
            "- Aplikasi membangun matriks / operasi citra lalu menampilkan hasilnya.\n"
            "- Fitur Undo/Redo dan simpan hasil tersedia, lalu bisa diunduh sebagai ZIP."
        ),
        "num_members": "Jumlah anggota",
        "member_label": "Anggota",
        "role_placeholder": "Isi peran / kontribusi di sini.",
        "edit_member_prompt": "Pilih foto anggota untuk diedit",
        "need_member_image": "Silakan upload minimal satu foto anggota tim untuk menggunakan kontrol edit.",
        "tools_title": "🎨 Kontrol Edit Foto",
        "controls_title": "Kontrol",
        "controls_hint": "Pilih operasi dan atur parameter untuk melihat efeknya.",
        "operation_label": "Operasi",
        "current_image": "Gambar Saat Ini",
        "preview_image": "Pratinjau Hasil",
        "preview_hint": "Atur parameter untuk melihat pratinjau.",
        "btn_apply": "Terapkan",
        "btn_save": "Simpan hasil",
        "btn_download": "Unduh PNG",
        "undo": "Undo",
        "redo": "Redo",
        "saved_results": "Jumlah hasil tersimpan",
        "btn_download_zip": "Unduh semua hasil sebagai ZIP",
        "op_translation": "Translasi",
        "op_scaling": "Skala",
        "op_rotation": "Rotasi",
        "op_shearing": "Shearing",
        "op_reflection": "Refleksi",
        "op_blur": "Blur (Konvolusi)",
        "op_sharpen": "Sharpen (Konvolusi)",
        "op_hsv": "Hapus Background (HSV)",
        "op_grabcut": "Hapus Background (GrabCut)",
        "op_gray": "Grayscale",
        "op_edge": "Deteksi Tepi",
        "op_invert": "Invert Warna",
        "translation_params": "Parameter Translasi",
        "scaling_params": "Parameter Skala",
        "rotation_params": "Parameter Rotasi",
        "shearing_params": "Parameter Shearing",
        "reflection_params": "Parameter Refleksi",
        "blur_params": "Parameter Blur",
        "sharpen_params": "Filter Sharpen",
        "hsv_params": "Threshold HSV untuk Background",
        "grabcut_params": "Parameter GrabCut",
        "edge_params": "Deteksi Tepi (Canny)",
        "tx_label": "tx (piksel)",
        "ty_label": "ty (piksel)",
        "scale_x": "Skala X",
        "scale_y": "Skala Y",
        "angle_label": "Sudut (derajat)",
        "shear_x": "Shear X",
        "shear_y": "Shear Y",
        "axis_label": "Sumbu Refleksi",
        "axis_horizontal": "Horizontal",
        "axis_vertical": "Vertikal",
        "axis_both": "Keduanya",
        "kernel_size": "Ukuran kernel (ganjil)",
        "h_min": "H min",
        "s_min": "S min",
        "v_min": "V min",
        "h_max": "H max",
        "s_max": "S max",
        "v_max": "V max",
        "rect_scale": "Skala kotak foreground",
        "iterations": "Iterasi",
        "low_thresh": "Ambang bawah",
        "high_thresh": "Ambang atas",
        "dl_current_png": "Unduh gambar saat ini (PNG)",
        "dl_current_jpg": "Unduh gambar saat ini (JPG)",
        "dl_meta_json": "Unduh metadata (JSON)",
        "dl_meta_csv": "Unduh metadata (CSV)",
        "dl_report_pdf": "Unduh laporan (PDF)",
        "report_title": "Laporan Pengolahan Citra",
        "meta_width": "Lebar",
        "meta_height": "Tinggi",
        "meta_mode": "Mode",
        "meta_saved": "Jumlah hasil tersimpan",
        "lang_en": "EN",
        "lang_id": "ID",
        "lang_zh": "中文",
        "lang_ja": "日本語",
        "lang_ko": "한국어",
        "lang_ar": "العربية",
        "top_language": "Bahasa",
        "top_dark_mode": "Mode gelap",
    },
    "zh": {
        "app_title": "图像处理中的矩阵变换 – 单页应用",
        "home_title": "📘 主页 / 简介",
        "overview": "概览",
        "conv_title": "卷积与滤波",
        "download_center": "下载中心",
        "team_title": "👥 团队成员 + 照片编辑控制（点击展开）",
        "team_heading": "团队成员",
        "team_how_title": "应用工作原理（简述）",
        "home_intro": "本应用演示矩阵运算和卷积用于几何变换、滤波以及背景移除和边缘检测。",
        "overview_text": "平移、缩放、旋转、剪切和反射等变换可用3×3齐次矩阵表示。",
        "conv_text": "卷积使用小内核在图像上滑动，可实现模糊和锐化等滤波效果。",
        "team_how_text": (
            "- 在团队成员部分上传图像。\n"
            "- 选择变换或滤波器并调整参数。\n"
            "- 应用计算对应矩阵或卷积并显示结果。\n"
            "- 使用撤销/重做并保存多个结果下载为 ZIP。"
        ),
        "num_members": "成员数",
        "member_label": "成员",
        "role_placeholder": "填写成员角色/贡献。",
        "edit_member_prompt": "选择要编辑的成员照片",
        "need_member_image": "请至少上传一张成员照片以使用编辑功能。",
        "tools_title": "🎨 照片编辑控制",
        "controls_title": "控制面板",
        "controls_hint": "选择操作并调整参数查看效果。",
        "operation_label": "操作",
        "current_image": "当前图像",
        "preview_image": "变换预览",
        "preview_hint": "调整参数以查看预览。",
        "btn_apply": "应用",
        "btn_save": "保存结果",
        "btn_download": "下载 PNG",
        "undo": "撤销",
        "redo": "重做",
        "saved_results": "已保存结果",
        "btn_download_zip": "下载所有结果为 ZIP",
        "op_translation": "平移",
        "op_scaling": "缩放",
        "op_rotation": "旋转",
        "op_shearing": "剪切",
        "op_reflection": "反射",
        "op_blur": "模糊（卷积）",
        "op_sharpen": "锐化（卷积）",
        "op_hsv": "背景移除（HSV）",
        "op_grabcut": "背景移除（GrabCut）",
        "op_gray": "灰度",
        "op_edge": "边缘检测",
        "op_invert": "颜色反转",
        "translation_params": "平移参数",
        "scaling_params": "缩放参数",
        "rotation_params": "旋转参数",
        "shearing_params": "剪切参数",
        "reflection_params": "反射参数",
        "blur_params": "模糊参数",
        "sharpen_params": "锐化滤波器",
        "hsv_params": "HSV 背景阈值",
        "grabcut_params": "GrabCut 参数",
        "edge_params": "边缘检测（Canny）",
        "tx_label": "tx（像素）",
        "ty_label": "ty（像素）",
        "scale_x": "缩放 X",
        "scale_y": "缩放 Y",
        "angle_label": "角度（度）",
        "shear_x": "剪切 X",
        "shear_y": "剪切 Y",
        "axis_label": "反射轴",
        "axis_horizontal": "水平",
        "axis_vertical": "垂直",
        "axis_both": "两者",
        "kernel_size": "内核大小（奇数）",
        "h_min": "H 最小值",
        "s_min": "S 最小值",
        "v_min": "V 最小值",
        "h_max": "H 最大值",
        "s_max": "S 最大值",
        "v_max": "V 最大值",
        "rect_scale": "前景矩形比例",
        "iterations": "迭代次数",
        "low_thresh": "低阈值",
        "high_thresh": "高阈值",
        "dl_current_png": "下载当前图像 (PNG)",
        "dl_current_jpg": "下载当前图像 (JPG)",
        "dl_meta_json": "下载元数据 (JSON)",
        "dl_meta_csv": "下载元数据 (CSV)",
        "dl_report_pdf": "下载报告 (PDF)",
        "report_title": "图像处理报告",
        "meta_width": "宽度",
        "meta_height": "高度",
        "meta_mode": "模式",
        "meta_saved": "已保存结果数",
        "lang_en": "EN",
        "lang_id": "ID",
        "lang_zh": "中文",
        "lang_ja": "日本語",
        "lang_ko": "한국어",
        "lang_ar": "العربية",
        "top_language": "语言",
        "top_dark_mode": "深色模式",
    },
    "ja": {
        "app_title": "画像処理における行列変換 – シングルページアプリ",
        "home_title": "📘 ホーム / イントロダクション",
        "overview": "概要",
        "conv_title": "畳み込みとフィルタリング",
        "download_center": "ダウンロードセンター",
        "team_title": "👥 チームメンバー + 写真編集コントロール（クリックで展開）",
        "team_heading": "チームメンバー",
        "team_how_title": "アプリの動作原理（簡潔）",
        "home_intro": "このアプリは行列演算と畳み込みによる幾何学的変換、フィルタリング、背景除去やエッジ検出を示します。",
        "overview_text": "平行移動、拡大縮小、回転、シアー、反射などの変換は 3×3 の斉次行列で表せます。",
        "conv_text": "畳み込みは小さなカーネルを画像全体にスライドさせる操作で、ぼかしやシャープ化を実現します。",
        "team_how_text": (
            "- チームメンバーセクションで画像をアップロード。\n"
            "- 変換またはフィルタを選択しパラメータを調整。\n"
            "- アプリが対応する行列または畳み込みを計算し結果を表示。\n"
            "- 元に戻す/やり直しを使用して複数結果を保存し ZIP でダウンロード。"
        ),
        "num_members": "メンバー数",
        "member_label": "メンバー",
        "role_placeholder": "役割 / 貢献内容を入力してください。",
        "edit_member_prompt": "編集するメンバー写真を選択",
        "need_member_image": "編集コントロールを使うにはメンバー写真を少なくとも 1 枚アップロードしてください。",
        "tools_title": "🎨 写真編集コントロール",
        "controls_title": "コントロール",
        "controls_hint": "操作を選択し、パラメータを調整して効果を確認してください。",
        "operation_label": "操作",
        "current_image": "現在の画像",
        "preview_image": "変換プレビュー",
        "preview_hint": "パラメータを調整してプレビューを確認してください。",
        "btn_apply": "適用",
        "btn_save": "結果を保存",
        "btn_download": "PNG をダウンロード",
        "undo": "元に戻す",
        "redo": "やり直し",
        "saved_results": "保存済み結果",
        "btn_download_zip": "すべての結果を ZIP でダウンロード",
        "op_translation": "平行移動",
        "op_scaling": "拡大縮小",
        "op_rotation": "回転",
        "op_shearing": "シアー",
        "op_reflection": "反射",
        "op_blur": "ぼかし（畳み込み）",
        "op_sharpen": "シャープ化（畳み込み）",
        "op_hsv": "背景除去（HSV）",
        "op_grabcut": "背景除去（GrabCut）",
        "op_gray": "グレースケール",
        "op_edge": "エッジ検出",
        "op_invert": "色反転",
        "translation_params": "平行移動パラメータ",
        "scaling_params": "拡大縮小パラメータ",
        "rotation_params": "回転パラメータ",
        "shearing_params": "シアーパラメータ",
        "reflection_params": "反射パラメータ",
        "blur_params": "ぼかしパラメータ",
        "sharpen_params": "シャープ化フィルタ",
        "hsv_params": "HSV 背景しきい値",
        "grabcut_params": "GrabCut パラメータ",
        "edge_params": "エッジ検出（Canny）",
        "tx_label": "tx（ピクセル）",
        "ty_label": "ty（ピクセル）",
        "scale_x": "拡大縮小 X",
        "scale_y": "拡大縮小 Y",
        "angle_label": "角度（度）",
        "shear_x": "シアー X",
        "shear_y": "シアー Y",
        "axis_label": "反射軸",
        "axis_horizontal": "水平",
        "axis_vertical": "垂直",
        "axis_both": "両方",
        "kernel_size": "カーネルサイズ（奇数）",
        "h_min": "H 最小値",
        "s_min": "S 最小値",
        "v_min": "V 最小値",
        "h_max": "H 最大値",
        "s_max": "S 最大値",
        "v_max": "V 最大値",
        "rect_scale": "前景矩形スケール",
        "iterations": "反復回数",
        "low_thresh": "下限しきい値",
        "high_thresh": "上限しきい値",
        "dl_current_png": "現在の画像をダウンロード (PNG)",
        "dl_current_jpg": "現在の画像をダウンロード (JPG)",
        "dl_meta_json": "メタデータをダウンロード (JSON)",
        "dl_meta_csv": "メタデータをダウンロード (CSV)",
        "dl_report_pdf": "レポートをダウンロード (PDF)",
        "report_title": "画像処理レポート",
        "meta_width": "幅",
        "meta_height": "高さ",
        "meta_mode": "モード",
        "meta_saved": "保存済み結果数",
        "lang_en": "EN",
        "lang_id": "ID",
        "lang_zh": "中文",
        "lang_ja": "日本語",
        "lang_ko": "한국어",
        "lang_ar": "العربية",
        "top_language": "言語",
        "top_dark_mode": "ダークモード",
    },
    "ko": {
        "app_title": "이미지 처리에서의 행렬 변환 – 단일 페이지 앱",
        "home_title": "📘 홈 / 소개",
        "overview": "개요",
        "conv_title": "컨볼루션 및 필터링",
        "download_center": "다운로드 센터",
        "team_title": "👥 팀 멤버 + 사진 편집 컨트롤 (클릭하여 열기)",
        "team_heading": "팀 멤버",
        "team_how_title": "앱 작동 원리 (간략)",
        "home_intro": "이 앱은 행렬 연산과 컨볼루션을 이용한 기하학적 변환, 필터링, 배경 제거 및 에지 검출을 보여줍니다.",
        "overview_text": "병진, 크기 조정, 회전, 전단, 반사와 같은 변환은 3×3 균질 행렬로 표현할 수 있습니다.",
        "conv_text": "컨볼루션은 작은 커널을 이미지 전체에 슬라이드하여 블러와 샤프닝 필터를 구현합니다.",
        "team_how_text": (
            "- 팀 멤버 섹션에서 이미지를 업로드합니다.\n"
            "- 변환 또는 필터를 선택하고 매개변수를 조정합니다.\n"
            "- 앱이 해당 행렬 또는 컨볼루션을 계산하여 결과를 표시합니다.\n"
            "- 실행 취소/다시 실행 및 결과 저장 후 ZIP으로 다운로드할 수 있습니다."
        ),
        "num_members": "멤버 수",
        "member_label": "멤버",
        "role_placeholder": "역할 / 기여 내용을 입력하세요.",
        "edit_member_prompt": "편집할 멤버 사진 선택",
        "need_member_image": "편집 컨트롤을 사용하려면 멤버 사진을 최소 1장 업로드하세요.",
        "tools_title": "🎨 사진 편집 컨트롤",
        "controls_title": "컨트롤",
        "controls_hint": "작업을 선택하고 매개변수를 조정하여 효과를 확인하세요.",
        "operation_label": "작업",
        "current_image": "현재 이미지",
        "preview_image": "변환 미리보기",
        "preview_hint": "매개변수를 조정하여 미리보기를 확인하세요.",
        "btn_apply": "적용",
        "btn_save": "결과 저장",
        "btn_download": "PNG 다운로드",
        "undo": "실행 취소",
        "redo": "다시 실행",
        "saved_results": "저장된 결과",
        "btn_download_zip": "모든 결과를 ZIP으로 다운로드",
        "op_translation": "병진",
        "op_scaling": "크기 조정",
        "op_rotation": "회전",
        "op_shearing": "전단",
        "op_reflection": "반사",
        "op_blur": "블러 (컨볼루션)",
        "op_sharpen": "샤프닝 (컨볼루션)",
        "op_hsv": "배경 제거 (HSV)",
        "op_grabcut": "배경 제거 (GrabCut)",
        "op_gray": "그레이스케일",
        "op_edge": "에지 검출",
        "op_invert": "색상 반전",
        "translation_params": "병진 매개변수",
        "scaling_params": "크기 조정 매개변수",
        "rotation_params": "회전 매개변수",
        "shearing_params": "전단 매개변수",
        "reflection_params": "반사 매개변수",
        "blur_params": "블러 매개변수",
        "sharpen_params": "샤프닝 필터",
        "hsv_params": "HSV 배경 임계값",
        "grabcut_params": "GrabCut 매개변수",
        "edge_params": "에지 검출 (Canny)",
        "tx_label": "tx (픽셀)",
        "ty_label": "ty (픽셀)",
        "scale_x": "크기 조정 X",
        "scale_y": "크기 조정 Y",
        "angle_label": "각도 (도)",
        "shear_x": "전단 X",
        "shear_y": "전단 Y",
        "axis_label": "반사 축",
        "axis_horizontal": "수평",
        "axis_vertical": "수직",
        "axis_both": "둘 다",
        "kernel_size": "커널 크기 (홀수)",
        "h_min": "H 최소",
        "s_min": "S 최소",
        "v_min": "V 최소",
        "h_max": "H 최대",
        "s_max": "S 최대",
        "v_max": "V 최대",
        "rect_scale": "전경 사각형 스케일",
        "iterations": "반복 횟수",
        "low_thresh": "하한 임계값",
        "high_thresh": "상한 임계값",
        "dl_current_png": "현재 이미지 다운로드 (PNG)",
        "dl_current_jpg": "현재 이미지 다운로드 (JPG)",
        "dl_meta_json": "메타데이터 다운로드 (JSON)",
        "dl_meta_csv": "메타데이터 다운로드 (CSV)",
        "dl_report_pdf": "리포트 다운로드 (PDF)",
        "report_title": "이미지 처리 리포트",
        "meta_width": "너비",
        "meta_height": "높이",
        "meta_mode": "모드",
        "meta_saved": "저장된 결과 수",
        "lang_en": "EN",
        "lang_id": "ID",
        "lang_zh": "中文",
        "lang_ja": "日本語",
        "lang_ko": "한국어",
        "lang_ar": "العربية",
        "top_language": "언어",
        "top_dark_mode": "다크 모드",
    },
    "ar": {
        "app_title": "تحويلات المصفوفات في معالجة الصور – تطبيق صفحة واحدة",
        "home_title": "📘 الرئيسية / المقدمة",
        "overview": "نظرة عامة",
        "conv_title": "الالتفاف والترشيح",
        "download_center": "مركز التنزيل",
        "team_title": "👥 أعضاء الفريق + أدوات تحرير الصور (انقر للفتح)",
        "team_heading": "أعضاء الفريق",
        "team_how_title": "كيفية عمل التطبيق (موجز)",
        "home_intro": "يعرض هذا التطبيق عمليات المصفوفات والالتفاف للتحويلات الهندسية، وترشيح الصور، وإزالة الخلفية، وكشف الحواف.",
        "overview_text": "يمكن تمثيل التحويلات مثل الإزاحة والتكبير والتدوير والإمالة والانعكاس بمصفوفة متجانسة 3×3.",
        "conv_text": "يستخدم الالتفاف نواة صغيرة تنزلق عبر الصورة لتحقيق تأثيرات مثل التمويه والشحذ.",
        "team_how_text": (
            "- قم برفع صور الأعضاء في قسم الفريق.\n"
            "- اختر التحويل أو المرشح واضبط المعاملات.\n"
            "- يحسب التطبيق المصفوفة أو الالتفاف المقابل ويعرض النتيجة.\n"
            "- استخدم التراجع/الإعادة وحفظ نتائج متعددة لتنزيلها كـ ZIP."
        ),
        "num_members": "عدد الأعضاء",
        "member_label": "عضو",
        "role_placeholder": "اكتب الدور / المساهمة هنا.",
        "edit_member_prompt": "اختر صورة عضو لتحريرها",
        "need_member_image": "يرجى رفع صورة عضو واحدة على الأقل لاستخدام أدوات التحرير.",
        "tools_title": "🎨 أدوات تحرير الصور",
        "controls_title": "لوحة التحكم",
        "controls_hint": "اختر العملية واضبط المعاملات لرؤية التأثير.",
        "operation_label": "العملية",
        "current_image": "الصورة الحالية",
        "preview_image": "معاينة التحويل",
        "preview_hint": "اضبط المعاملات لرؤية المعاينة.",
        "btn_apply": "تطبيق",
        "btn_save": "حفظ النتيجة",
        "btn_download": "تنزيل PNG",
        "undo": "تراجع",
        "redo": "إعادة",
        "saved_results": "النتائج المحفوظة",
        "btn_download_zip": "تنزيل كل النتائج كملف ZIP",
        "op_translation": "إزاحة",
        "op_scaling": "تكبير/تصغير",
        "op_rotation": "دوران",
        "op_shearing": "إمالة",
        "op_reflection": "انعكاس",
        "op_blur": "تمويه (التفاف)",
        "op_sharpen": "شحذ (التفاف)",
        "op_hsv": "إزالة الخلفية (HSV)",
        "op_grabcut": "إزالة الخلفية (GrabCut)",
        "op_gray": "تدرج رمادي",
        "op_edge": "كشف الحواف",
        "op_invert": "عكس الألوان",
        "translation_params": "معاملات الإزاحة",
        "scaling_params": "معاملات التكبير/التصغير",
        "rotation_params": "معاملات الدوران",
        "shearing_params": "معاملات الإمالة",
        "reflection_params": "معاملات الانعكاس",
        "blur_params": "معاملات التمويه",
        "sharpen_params": "مرشح الشحذ",
        "hsv_params": "عتبة HSV للخلفية",
        "grabcut_params": "معاملات GrabCut",
        "edge_params": "كشف الحواف (Canny)",
        "tx_label": "tx (بكسل)",
        "ty_label": "ty (بكسل)",
        "scale_x": "مقياس X",
        "scale_y": "مقياس Y",
        "angle_label": "الزاوية (درجة)",
        "shear_x": "إمالة X",
        "shear_y": "إمالة Y",
        "axis_label": "محور الانعكاس",
        "axis_horizontal": "أفقي",
        "axis_vertical": "عمودي",
        "axis_both": "كلاهما",
        "kernel_size": "حجم النواة (فردي)",
        "h_min": "H الحد الأدنى",
        "s_min": "S الحد الأدنى",
        "v_min": "V الحد الأدنى",
        "h_max": "H الحد الأقصى",
        "s_max": "S الحد الأقصى",
        "v_max": "V الحد الأقصى",
        "rect_scale": "مقياس مستطيل المقدمة",
        "iterations": "عدد التكرارات",
        "low_thresh": "العتبة الدنيا",
        "high_thresh": "العتبة العليا",
        "dl_current_png": "تنزيل الصورة الحالية (PNG)",
        "dl_current_jpg": "تنزيل الصورة الحالية (JPG)",
        "dl_meta_json": "تنزيل البيانات الوصفية (JSON)",
        "dl_meta_csv": "تنزيل البيانات الوصفية (CSV)",
        "dl_report_pdf": "تنزيل التقرير (PDF)",
        "report_title": "تقرير معالجة الصور",
        "meta_width": "العرض",
        "meta_height": "الارتفاع",
        "meta_mode": "الوضع",
        "meta_saved": "عدد النتائج المحفوظة",
        "lang_en": "EN",
        "lang_id": "ID",
        "lang_zh": "中文",
        "lang_ja": "日本語",
        "lang_ko": "한국어",
        "lang_ar": "العربية",
        "top_language": "اللغة",
        "top_dark_mode": "الوضع الداكن",
    },
}

def t(lang: str, key: str) -> str:
    if lang not in LANG_TEXT:
        lang = "en"
    return LANG_TEXT[lang].get(key, LANG_TEXT["en"].get(key, key))


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
# EXTRA EDIT FEATURES
# =========================================================

def flip_image(pil_img: Image.Image, mode: str = "horizontal") -> Image.Image:
    arr = np.array(pil_img.convert("RGBA"))
    if mode == "horizontal":
        arr_flipped = np.flip(arr, axis=1)
    else:
        arr_flipped = np.flip(arr, axis=0)
    return Image.fromarray(arr_flipped)

def adjust_brightness(pil_img: Image.Image, factor: float) -> Image.Image:
    enhancer = ImageEnhance.Brightness(pil_img)
    return enhancer.enhance(factor)

def adjust_contrast(pil_img: Image.Image, factor: float) -> Image.Image:
    enhancer = ImageEnhance.Contrast(pil_img)
    return enhancer.enhance(factor)

def crop_image(pil_img: Image.Image, left: int, top: int, right: int, bottom: int) -> Image.Image:
    w, h = pil_img.size
    left = max(0, min(left, w - 1))
    top = max(0, min(top, h - 1))
    right = max(left + 1, min(right, w))
    bottom = max(top + 1, min(bottom, h))
    return pil_img.crop((left, top, right, bottom))


# =========================================================
# STATE
# =========================================================

def init_state():
    if "history" not in st.session_state:
        st.session_state.history = []
    if "redo_stack" not in st.session_state:
        st.session_state.redo_stack = []
    if "current_image" not in st.session_state:
        st.session_state.current_image = None
    if "saved_results" not in st.session_state:
        st.session_state.saved_results = []
    if "lang_code" not in st.session_state:
        st.session_state.lang_code = "id"
    if "last_member" not in st.session_state:
        st.session_state.last_member = None
    if "dark_mode" not in st.session_state:
        st.session_state.dark_mode = False

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
# AFFINE TRANSFORMS
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
    else:
        R = np.array([[-1, 0, 0],
                      [0, -1, 0],
                      [0,  0, 1]], dtype=float)
    T2 = get_translation_matrix(cx, cy)
    return T2 @ R @ T1


# =========================================================
# CONVOLUTION
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
    blurred_rgb = np.stack([blurred_gray] * 3, axis=-1)
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
    sharp_rgb = np.stack([sharp_gray] * 3, axis=-1)
    out = np.dstack([sharp_rgb, alpha])
    return out


# =========================================================
# SPECIAL FILTERS
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
# TOP BAR (DARK MODE + LANGUAGE)
# =========================================================

def top_bar_and_theme():
    st.markdown(BASE_CSS, unsafe_allow_html=True)
    if st.session_state.get("dark_mode", False):
        st.markdown(DARK_CSS, unsafe_allow_html=True)

    col_left, col_right = st.columns([3, 3])
    with col_left:
        dm = st.toggle("🌙 " + t(st.session_state.lang_code, "top_dark_mode"),
                       value=st.session_state["dark_mode"])
        st.session_state["dark_mode"] = dm
    with col_right:
        lang_options = ["en", "id", "zh", "ja", "ko", "ar"]
        lang_labels = [LANG_TEXT["en"]["lang_en"],
                       LANG_TEXT["id"]["lang_id"],
                       LANG_TEXT["zh"]["lang_zh"],
                       LANG_TEXT["ja"]["lang_ja"],
                       LANG_TEXT["ko"]["lang_ko"],
                       LANG_TEXT["ar"]["lang_ar"]]
        idx_now = lang_options.index(st.session_state.lang_code)
        choice = st.radio(
            t(st.session_state.lang_code, "top_language"),
            options=list(range(len(lang_options))),
            format_func=lambda i: lang_labels[i],
            horizontal=True,
            index=idx_now,
        )
        st.session_state.lang_code = lang_options[choice]


# =========================================================
# MAIN APP
# =========================================================

def main():
    init_state()
    top_bar_and_theme()
    lang = st.session_state.lang_code

    # Hero card
    st.markdown(
        f"""
        <div class='hero-card'>
          <h4 style="margin-top:0; margin-bottom:0.3rem; color:#047857;">
            🧮 {t(lang, "app_title")}
          </h4>
          <p style="margin:0; font-size:0.9rem; color:#065f46;">
            {t(lang, "home_intro")}
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div class='decorative-divider'></div>", unsafe_allow_html=True)
    st.markdown("<div class='main-card'>", unsafe_allow_html=True)

    # Intro / teori
    with st.expander(t(lang, "home_title"), expanded=True):
        st.subheader(t(lang, "overview"))
        st.write(t(lang, "overview_text"))
        st.subheader(t(lang, "conv_title"))
        st.write(t(lang, "conv_text"))

    # Anggota tim + kontrol
    with st.expander(t(lang, "team_title"), expanded=True):
        st.subheader(t(lang, "team_heading"))
        num_members = st.number_input(t(lang, "num_members"), 1, 12, 4)
        members_data = []
        for i in range(int(num_members)):
            st.markdown(f"**{t(lang, 'member_label')} {i+1}**")
            col_form = st.columns([2, 2, 2])
            with col_form[0]:
                name = st.text_input(f"Name {i+1}", key=f"name_{i+1}")
            with col_form[1]:
                role = st.text_input(f"Role {i+1}", key=f"role_{i+1}",
                                     placeholder=t(lang, "role_placeholder"))
            with col_form[2]:
                photo_file = st.file_uploader(
                    f"Photo {i+1}", type=["png", "jpg", "jpeg"], key=f"photo_{i+1}"
                )
            members_data.append((name, role, photo_file))
            st.markdown("---")

        cols = st.columns(2)
        member_images = []

        for i, (name, role, photo_file) in enumerate(members_data):
            if not name and not role and photo_file is None:
                continue
            with cols[i % 2]:
                label_name = name or f"{t(lang, 'member_label')} {i+1}"
                st.markdown(f"**{label_name}**")
                img_obj = None
                if photo_file is not None:
                    img_obj = Image.open(photo_file)
                    st.image(img_obj, width=200)
                st.write(role or t(lang, "role_placeholder"))
                if img_obj is not None:
                    member_images.append((label_name, img_obj))

        st.markdown("---")
        st.subheader(t(lang, "team_how_title"))
        st.write(t(lang, "team_how_text"))

        st.markdown("---")
        st.subheader(t(lang, "tools_title"))

        if not member_images:
            st.info(t(lang, "need_member_image"))
            st.markdown("</div>", unsafe_allow_html=True)
            return

        names_list = [m[0] for m in member_images]
        selected_name = st.selectbox(t(lang, "edit_member_prompt"), names_list, key="member_select")

        for nm, img_obj in member_images:
            if nm == selected_name:
                base_img = img_obj
                break

        if st.session_state.current_image is None:
            st.session_state.current_image = base_img.copy()
            st.session_state.history.clear()
            st.session_state.redo_stack.clear()
            st.session_state.last_member = selected_name
        else:
            if st.session_state.last_member != selected_name:
                st.session_state.last_member = selected_name
                st.session_state.current_image = base_img.copy()
                st.session_state.history.clear()
                st.session_state.redo_stack.clear()

        current_img = st.session_state.current_image
        img_arr = pil_to_array(current_img)
        h, w, _ = img_arr.shape
        cx, cy = w / 2, h / 2

        st.markdown(f"### {t(lang, 'controls_title')}")
        st.markdown(f"> {t(lang, 'controls_hint')}")

        col_left, col_right = st.columns([1.5, 2])

        with col_left:
            tool = st.selectbox(
                t(lang, "operation_label"),
                [
                    t(lang, "op_translation"),
                    t(lang, "op_scaling"),
                    t(lang, "op_rotation"),
                    t(lang, "op_shearing"),
                    t(lang, "op_reflection"),
                    t(lang, "op_blur"),
                    t(lang, "op_sharpen"),
                    "Flip Horizontal",
                    "Flip Vertical",
                    "Brightness",
                    "Contrast",
                    "Crop",
                    t(lang, "op_hsv"),
                    t(lang, "op_grabcut"),
                    t(lang, "op_gray"),
                    t(lang, "op_edge"),
                    t(lang, "op_invert"),
                ],
                key="tool_select",
            )

            transformed_img = None

            if tool == t(lang, "op_translation"):
                st.markdown(f"**{t(lang, 'translation_params')}**")
                tx = st.slider(t(lang, "tx_label"), -200, 200, 0, key="tx")
                ty = st.slider(t(lang, "ty_label"), -200, 200, 0, key="ty")
                M = get_translation_matrix(tx, ty)
                out = apply_affine_transform(img_arr, M)
                transformed_img = array_to_pil(out)

            elif tool == t(lang, "op_scaling"):
                st.markdown(f"**{t(lang, 'scaling_params')}**")
                sx = st.slider(t(lang, "scale_x"), 0.1, 3.0, 1.0, key="sx")
                sy = st.slider(t(lang, "scale_y"), 0.1, 3.0, 1.0, key="sy")
                M = get_scaling_matrix(sx, sy, cx, cy)
                out = apply_affine_transform(img_arr, M)
                transformed_img = array_to_pil(out)

            elif tool == t(lang, "op_rotation"):
                st.markdown(f"**{t(lang, 'rotation_params')}**")
                angle = st.slider(t(lang, "angle_label"), -180, 180, 0, key="angle")
                M = get_rotation_matrix(angle, cx, cy)
                out = apply_affine_transform(img_arr, M)
                transformed_img = array_to_pil(out)

            elif tool == t(lang, "op_shearing"):
                st.markdown(f"**{t(lang, 'shearing_params')}**")
                shx = st.slider(t(lang, "shear_x"), -1.0, 1.0, 0.0, key="shx")
                shy = st.slider(t(lang, "shear_y"), -1.0, 1.0, 0.0, key="shy")
                M = get_shearing_matrix(shx, shy, cx, cy)
                out = apply_affine_transform(img_arr, M)
                transformed_img = array_to_pil(out)

            elif tool == t(lang, "op_reflection"):
                st.markdown(f"**{t(lang, 'reflection_params')}**")
                axis = st.selectbox(
                    t(lang, "axis_label"),
                    [
                        t(lang, "axis_horizontal"),
                        t(lang, "axis_vertical"),
                        t(lang, "axis_both"),
                    ],
                    key="axis",
                )
                axis_map = {
                    t(lang, "axis_horizontal"): "Horizontal",
                    t(lang, "axis_vertical"): "Vertical",
                    t(lang, "axis_both"): "Both",
                }
                axis_internal = axis_map[axis]
                M = get_reflection_matrix(axis_internal, cx, cy)
                out = apply_affine_transform(img_arr, M)
                transformed_img = array_to_pil(out)

            elif tool == t(lang, "op_blur"):
                st.markdown(f"**{t(lang, 'blur_params')}**")
                k = st.slider(t(lang, "kernel_size"), 1, 9, 3, step=2, key="k_blur")
                out = blur_filter(add_alpha_channel(img_arr), kernel_size=k)
                transformed_img = array_to_pil(out)

            elif tool == t(lang, "op_sharpen"):
                st.markdown(f"**{t(lang, 'sharpen_params')}**")
                out = sharpen_filter(add_alpha_channel(img_arr))
                transformed_img = array_to_pil(out)

            elif tool == "Flip Horizontal":
                transformed_img = flip_image(current_img, mode="horizontal")

            elif tool == "Flip Vertical":
                transformed_img = flip_image(current_img, mode="vertical")

            elif tool == "Brightness":
                factor = st.slider("Brightness factor", 0.1, 3.0, 1.0, key="bright")
                transformed_img = adjust_brightness(current_img, factor)

            elif tool == "Contrast":
                factor = st.slider("Contrast factor", 0.1, 3.0, 1.0, key="contrast")
                transformed_img = adjust_contrast(current_img, factor)

            elif tool == "Crop":
                w_img, h_img = current_img.size
                st.write(f"{w_img} x {h_img} px")
                left = st.number_input("Left", 0, w_img - 1, 0, key="crop_left")
                top = st.number_input("Top", 0, h_img - 1, 0, key="crop_top")
                right = st.number_input("Right", 1, w_img, w_img, key="crop_right")
                bottom = st.number_input("Bottom", 1, h_img, h_img, key="crop_bottom")
                transformed_img = crop_image(current_img, left, top, right, bottom)

            elif tool == t(lang, "op_hsv"):
                st.markdown(f"**{t(lang, 'hsv_params')}**")
                h_min = st.slider(t(lang, "h_min"), 0, 180, 0, key="hmin")
                s_min = st.slider(t(lang, "s_min"), 0, 255, 0, key="smin")
                v_min = st.slider(t(lang, "v_min"), 0, 255, 200, key="vmin")
                h_max = st.slider(t(lang, "h_max"), 0, 180, 180, key="hmax")
                s_max = st.slider(t(lang, "s_max"), 0, 255, 25, key="smax")
                v_max = st.slider(t(lang, "v_max"), 0, 255, 255, key="vmax")
                transformed_img = remove_background_hsv(
                    current_img,
                    lower_hsv=(h_min, s_min, v_min),
                    upper_hsv=(h_max, s_max, v_max),
                )

            elif tool == t(lang, "op_grabcut"):
                st.markdown(f"**{t(lang, 'grabcut_params')}**")
                rect_scale = st.slider(t(lang, "rect_scale"), 0.5, 1.0, 0.9, key="rect")
                iters = st.slider(t(lang, "iterations"), 1, 10, 5, key="iters")
                transformed_img = remove_background_grabcut(
                    current_img, rect_scale=rect_scale, iters=iters
                )

            elif tool == t(lang, "op_gray"):
                transformed_img = grayscale_filter(current_img)

            elif tool == t(lang, "op_edge"):
                st.markdown(f"**{t(lang, 'edge_params')}**")
                low = st.slider(t(lang, "low_thresh"), 0, 255, 100, key="low")
                high = st.slider(t(lang, "high_thresh"), 0, 255, 200, key="high")
                transformed_img = edge_detection(current_img, low, high)

            elif tool == t(lang, "op_invert"):
                transformed_img = invert_colors(current_img)

        with col_right:
            col_img1, col_img2 = st.columns(2)
            with col_img1:
                st.markdown(f"**{t(lang, 'current_image')}**")
                st.image(current_img, use_container_width=True)
            with col_img2:
                st.markdown(f"**{t(lang, 'preview_image')}**")
                if transformed_img is not None:
                    st.image(transformed_img, use_container_width=True)
                else:
                    st.info(t(lang, "preview_hint"))

            if transformed_img is not None:
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    if st.button(t(lang, "btn_apply")):
                        push_history(transformed_img)
                with col_b:
                    if st.button(t(lang, "btn_save")):
                        default_name = "result_" + str(len(st.session_state.saved_results) + 1)
                        save_current_result(default_name)
                with col_c:
                    buf = io.BytesIO()
                    transformed_img.save(buf, format="PNG")
                    byte_im = buf.getvalue()
                    st.download_button(
                        label=t(lang, "btn_download"),
                        data=byte_im,
                        file_name="transformed.png",
                        mime="image/png",
                    )

        st.markdown("---")
        col_u, col_r, col_z = st.columns([1, 1, 2])
        with col_u:
            if st.button(t(lang, "undo")):
                undo()
        with col_r:
            if st.button(t(lang, "redo")):
                redo()
        with col_z:
            st.write(f"{t(lang, 'saved_results')}: {len(st.session_state.saved_results)}")
            if st.session_state.saved_results:
                zip_buffer = make_zip_from_results(st.session_state.saved_results)
                st.download_button(
                    label=t(lang, "btn_download_zip"),
                    data=zip_buffer,
                    file_name="results.zip",
                    mime="application/zip",
                )

        # Download Center
        st.markdown("---")
        st.subheader(t(lang, "download_center"))

        if st.session_state.current_image is not None:
            buf_png = io.BytesIO()
            st.session_state.current_image.save(buf_png, format="PNG")
            st.download_button(
                t(lang, "dl_current_png"),
                data=buf_png.getvalue(),
                file_name="current_image.png",
                mime="image/png",
            )

            buf_jpg = io.BytesIO()
            rgb_img = st.session_state.current_image.convert("RGB")
            rgb_img.save(buf_jpg, format="JPEG")
            st.download_button(
                t(lang, "dl_current_jpg"),
                data=buf_jpg.getvalue(),
                file_name="current_image.jpg",
                mime="image/jpeg",
            )

            meta = {
                "width": st.session_state.current_image.width,
                "height": st.session_state.current_image.height,
                "mode": st.session_state.current_image.mode,
                "num_saved_results": len(st.session_state.saved_results),
            }
            meta_json = json.dumps(meta, indent=2)
            st.download_button(
                t(lang, "dl_meta_json"),
                data=meta_json,
                file_name="image_metadata.json",
                mime="application/json",
            )

            df_meta = pd.DataFrame([meta])
            csv_meta = df_meta.to_csv(index=False).encode("utf-8")
            st.download_button(
                t(lang, "dl_meta_csv"),
                data=csv_meta,
                file_name="image_metadata.csv",
                mime="text/csv",
            )

            pdf_buf = io.BytesIO()
            c = canvas.Canvas(pdf_buf)
            c.drawString(50, 800, t(lang, "report_title"))
            c.drawString(50, 780, f"{t(lang, 'meta_width')}: {meta['width']}")
            c.drawString(50, 760, f"{t(lang, 'meta_height')}: {meta['height']}")
            c.drawString(50, 740, f"{t(lang, 'meta_mode')}: {meta['mode']}")
            c.drawString(50, 720, f"{t(lang, 'meta_saved')}: {meta['num_saved_results']}")
            c.showPage()
            c.save()
            pdf_buf.seek(0)
            st.download_button(
                t(lang, "dl_report_pdf"),
                data=pdf_buf,
                file_name="report.pdf",
                mime="application/pdf",
            )

    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
