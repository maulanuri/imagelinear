import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats  # make sure scipy is installed: pip install scipy

# =======================
# PAGE CONFIG
# =======================
st.set_page_config(
    page_title="Survey Analysis X and Y",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =======================
# LANGUAGE DICTIONARY
# =======================
LANGS = {
    "en": "English",
    "id": "Bahasa Indonesia",
    "zh": "中文 (Chinese)",
    "ja": "日本語 (Japanese)",
}

TEXT = {
    "title": {
        "en": "Digital Payment Survey Analysis: X and Y",
        "id": "Analisis Survei Pembayaran Digital: X dan Y",
        "zh": "数字支付问卷分析：X 和 Y",
        "ja": "デジタル決済アンケート分析：X と Y",
    },
    "subtitle": {
        "en": "Explore descriptive statistics and associations between X and Y based on your survey data.",
        "id": "Jelajahi statistik deskriptif dan hubungan antara X dan Y berdasarkan data survei Anda.",
        "zh": "基于您的问卷数据，探索 X 与 Y 之间的描述性统计和关联。",
        "ja": "アンケートデータに基づいて，X と Y の記述統計と関連性を分析します。",
    },
    "start_button": {
        "en": "Start Analysis",
        "id": "Mulai Analisis",
        "zh": "开始分析",
        "ja": "分析を開始",
    },
    "upload_header": {
        "en": "Upload Data",
        "id": "Unggah Data",
        "zh": "上传数据",
        "ja": "データをアップロード",
    },
    "upload_info": {
        "en": "Please upload survei.csv from the sidebar.",
        "id": "Silakan unggah file survei.csv dari sidebar.",
        "zh": "请从侧边栏上传 survei.csv 文件。",
        "ja": "サイドバーから survei.csv ファイルをアップロードしてください。",
    },
    "data_preview": {
        "en": "Data Preview",
        "id": "Pratinjau Data",
        "zh": "数据预览",
        "ja": "データプレビュー",
    },
    "available_columns": {
        "en": "Available columns:",
        "id": "Kolom yang tersedia:",
        "zh": "可用列：",
        "ja": "利用可能な列：",
    },
    "select_xy_title": {
        "en": "Select items for X and Y (Likert scale)",
        "id": "Pilih item untuk X dan Y (skala Likert)",
        "zh": "选择 X 和 Y 的题项（李克特量表）",
        "ja": "X と Y の項目を選択（リッカート尺度）",
    },
    "select_x_label": {
        "en": "Select items for variable X (e.g., financial discipline statements)",
        "id": "Pilih item untuk variabel X (misalnya pernyataan kedisiplinan finansial)",
        "zh": "选择变量 X 的题项（例如财务纪律相关题项）",
        "ja": "変数 X の項目を選択（例：金融規律に関する設問）",
    },
    "select_y_label": {
        "en": "Select items for variable Y (e.g., digital payment behavior/consumption statements)",
        "id": "Pilih item untuk variabel Y (misalnya perilaku/konsumsi pembayaran digital)",
        "zh": "选择变量 Y 的题项（例如数字支付行为/消费相关题项）",
        "ja": "変数 Y の項目を選択（例：デジタル決済行動・消費に関する設問）",
    },
    "likert_note": {
        "en": "Note: Make sure the selected items are Likert-scale questions that can be converted to numeric values (1–5).",
        "id": "Catatan: Pastikan item yang dipilih adalah pertanyaan skala Likert yang dapat diubah ke nilai numerik (1–5).",
        "zh": "注意：请确保所选题项为可转换为数值（1–5）的李克特量表题。",
        "ja": "注意：選択した項目が，数値（1〜5）に変換可能なリッカート尺度の設問であることを確認してください。",
    },
    "desc_header": {
        "en": "A. Descriptive Statistics",
        "id": "A. Statistik Deskriptif",
        "zh": "A. 描述性统计",
        "ja": "A. 記述統計",
    },
    "numeric_select_label": {
        "en": "Select numeric columns for descriptive statistics",
        "id": "Pilih kolom numerik untuk statistik deskriptif",
        "zh": "选择用于描述性统计的数值列",
        "ja": "記述統計に用いる数値列を選択",
    },
    "cat_select_label": {
        "en": "Select categorical columns for frequency tables",
        "id": "Pilih kolom kategorik untuk tabel frekuensi",
        "zh": "选择分类列用于频数表",
        "ja": "度数表に用いるカテゴリ列を選択",
    },
    "a1_header": {
        "en": "A.1 Statistics for each item / numeric variable",
        "id": "A.1 Statistik untuk setiap item / variabel numerik",
        "zh": "A.1 各题项/数值变量统计",
        "ja": "A.1 各項目・数値変数の統計量",
    },
    "numeric_warning": {
        "en": "Select at least one numeric column to view statistics.",
        "id": "Pilih minimal satu kolom numerik untuk melihat statistik.",
        "zh": "请至少选择一列数值变量以查看统计结果。",
        "ja": "統計量を表示するには，少なくとも1つの数値列を選択してください。",
    },
    "a2_header": {
        "en": "A.2 Frequency & Percentage Tables",
        "id": "A.2 Tabel Frekuensi & Persentase",
        "zh": "A.2 频数与百分比表",
        "ja": "A.2 度数・百分率表",
    },
    "cat_warning": {
        "en": "Select at least one categorical column to create frequency tables.",
        "id": "Pilih minimal satu kolom kategorik untuk membuat tabel frekuensi.",
        "zh": "请至少选择一列分类变量以生成频数表。",
        "ja": "度数表を作成するには，少なくとも1つのカテゴリ列を選択してください。",
    },
    "a3_header": {
        "en": "A.3 Histogram & Boxplot (Optional)",
        "id": "A.3 Histogram & Boxplot (Opsional)",
        "zh": "A.3 直方图与箱线图（可选）",
        "ja": "A.3 ヒストグラムと箱ひげ図（任意）",
    },
    "a3_warning": {
        "en": "Select at least one numeric column for histogram & boxplot.",
        "id": "Pilih minimal satu kolom numerik untuk histogram & boxplot.",
        "zh": "请至少选择一列数值变量以绘制直方图和箱线图。",
        "ja": "ヒストグラムと箱ひげ図を作成するには，少なくとも1つの数値列を選択してください。",
    },
    "b_header": {
        "en": "B. Association Analysis between X_total and Y_total",
        "id": "B. Analisis Hubungan antara X_total dan Y_total",
        "zh": "B. X_total 与 Y_total 的关联分析",
        "ja": "B. X_total と Y_total の関連分析",
    },
    "assoc_info": {
        "en": "For association analysis, first select items for X and Y so that X_total and Y_total can be computed.",
        "id": "Untuk analisis hubungan, pertama pilih item untuk X dan Y sehingga X_total dan Y_total dapat dihitung.",
        "zh": "要进行关联分析，请先为 X 和 Y 选择题项以计算 X_total 和 Y_total。",
        "ja": "関連分析を行うには，まず X と Y の項目を選択し，X_total と Y_total を算出してください。",
    },
}

# =======================
# SESSION STATE: LANGUAGE & COVER
# =======================
if "lang" not in st.session_state:
    st.session_state.lang = "id"

if "show_app" not in st.session_state:
    st.session_state.show_app = False

# =======================
# COVER PAGE
# =======================
with st.sidebar:
    st.markdown("### Language / Bahasa")
    lang_label = st.radio(
        label="",
        options=list(LANGS.keys()),
        format_func=lambda x: LANGS[x],
        index=list(LANGS.keys()).index(st.session_state.lang),
        horizontal=False,
    )
    st.session_state.lang = lang_label

# Nice centered cover
if not st.session_state.show_app:
    col_left, col_center, col_right = st.columns([1, 2, 1])

    with col_center:
        st.markdown(
            """
            <div style="text-align:center; padding:40px; border-radius:16px;
                        background: linear-gradient(135deg, #1f77b4, #ff7f0e);
                        color:white; box-shadow: 0 4px 15px rgba(0,0,0,0.25);">
                <h1 style="margin-bottom:0.4em;">📊 Survey Analysis X & Y</h1>
                <p style="font-size:1.0rem; margin-bottom:1.5em;">
                    Analyze digital payment survey data with descriptive statistics and association tests.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<br/>", unsafe_allow_html=True)

        # Title + subtitle in selected language
        st.markdown(f"### {TEXT['title'][st.session_state.lang]}")
        st.write(TEXT["subtitle"][st.session_state.lang])

        st.markdown("---")

        st.write("Select language from the sidebar, then click the button below to start.")
        if st.button(TEXT["start_button"][st.session_state.lang]):
            st.session_state.show_app = True

    st.stop()

# =======================
# MAIN APP CONTENT (AFTER COVER)
# =======================

# Title in current language
st.title(TEXT["title"][st.session_state.lang])

# 1. Upload / read data
st.sidebar.header(TEXT["upload_header"][st.session_state.lang])
uploaded_file = st.sidebar.file_uploader("Upload survei.csv file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    st.info(TEXT["upload_info"][st.session_state.lang])
    st.stop()

st.subheader(TEXT["data_preview"][st.session_state.lang])
st.dataframe(df.head())

st.write(TEXT["available_columns"][st.session_state.lang])
st.write(list(df.columns))

# 2. Select X and Y items
st.markdown(f"### {TEXT['select_xy_title'][st.session_state.lang]}")

likert_cols = [
    c for c in df.columns
    if "10." in c or "11." in c or "12." in c
    or "13." in c or "14." in c or "15." in c or "16." in c
    or "17." in c or "18." in c or "19." in c or "20." in c
    or "22." in c or "23." in c
]

if not likert_cols:
    likert_cols = [c for c in df.columns if "=" in str(df[c].iloc[0])]

cols_x = st.multiselect(
    TEXT["select_x_label"][st.session_state.lang],
    options=likert_cols
)

cols_y = st.multiselect(
    TEXT["select_y_label"][st.session_state.lang],
    options=likert_cols
)

st.markdown(TEXT["likert_note"][st.session_state.lang])

# Helper: Likert to numeric
def likert_to_num(df_sub):
    out = df_sub.copy()
    for c in out.columns:
        out[c] = out[c].astype(str).str.extract(r"(\d+)").astype(float)
    return out

if cols_x:
    x_numeric = likert_to_num(df[cols_x])
    df["X_total"] = x_numeric.sum(axis=1, min_count=1)
else:
    x_numeric = None

if cols_y:
    y_numeric = likert_to_num(df[cols_y])
    df["Y_total"] = y_numeric.sum(axis=1, min_count=1)
else:
    y_numeric = None

st.markdown("---")

# 3. Descriptive statistics
st.header(TEXT["desc_header"][st.session_state.lang])

numeric_candidates = ["2. Age (numeric)"]
if "X_total" in df.columns:
    numeric_candidates.append("X_total")
if "Y_total" in df.columns:
    numeric_candidates.append("Y_total")

numeric_cols = st.multiselect(
    TEXT["numeric_select_label"][st.session_state.lang],
    options=list(df.columns),
    default=[c for c in numeric_candidates if c in df.columns]
)

cat_cols_default = [
    "1. Gender",
    "3. Education Level",
    "4. Employment Status",
    "5. Average Monthly Income",
    "6. How often did you use digital payment methods (e-wallet, mobile banking, QRIS, etc.) in the past week?",
    "8. What do you primarily use digital payments for?",
]
cat_cols = st.multiselect(
    TEXT["cat_select_label"][st.session_state.lang],
    options=list(df.columns),
    default=[c for c in cat_cols_default if c in df.columns]
)

# 3.1
st.subheader(TEXT["a1_header"][st.session_state.lang])

if numeric_cols:
    for col in numeric_cols:
        seri = pd.to_numeric(df[col], errors="coerce").dropna()
        if seri.empty:
            st.warning(f"Column {col} has no valid numeric data.")
            continue

        mean_val = seri.mean()
        median_val = seri.median()
        mode_val = seri.mode()
        min_val = seri.min()
        max_val = seri.max()
        std_val = seri.std()

        st.markdown(f"#### Statistics for: {col}")
        stats_rows = [
            ("Mean", mean_val),
            ("Median", median_val),
            ("Minimum", min_val),
            ("Maximum", max_val),
            ("Std Dev", std_val),
        ] + [(f"Mode {i+1}", v) for i, v in enumerate(mode_val.values)]

        stats_df = pd.DataFrame(stats_rows, columns=["Statistic", "Value"])
        st.dataframe(stats_df)

        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(x="Statistic", y="Value", data=stats_df, ax=ax, palette="viridis")
        ax.set_title(f"Statistics for {col}")
        ax.set_xlabel("")
        ax.set_ylabel("Value")
        plt.xticks(rotation=30)
        st.pyplot(fig)
else:
    st.info(TEXT["numeric_warning"][st.session_state.lang])

# 3.2
st.subheader(TEXT["a2_header"][st.session_state.lang])

if cat_cols:
    for col in cat_cols:
        st.markdown(f"#### Frequency table: {col}")
        freq = df[col].value_counts(dropna=False)
        percent = df[col].value_counts(normalize=True, dropna=False) * 100

        freq_table = pd.DataFrame({
            "Category": freq.index.astype(str),
            "Frequency": freq.values,
            "Percentage": percent.values.round(2)
        })

        st.dataframe(freq_table)

        fig2, ax2 = plt.subplots(figsize=(7, 4))
        sns.barplot(x="Category", y="Frequency", data=freq_table, ax=ax2, palette="magma")
        ax2.set_title(f"Frequency of {col}")
        ax2.set_xlabel("")
        ax2.set_ylabel("Frequency")
        plt.xticks(rotation=30, ha="right")
        st.pyplot(fig2)
else:
    st.info(TEXT["cat_warning"][st.session_state.lang])

# 3.3
st.subheader(TEXT["a3_header"][st.session_state.lang])

if numeric_cols:
    pilihan_plot_col = st.selectbox(
        "Select one numeric column for histogram and boxplot",
        options=numeric_cols
    )

    seri_plot = pd.to_numeric(df[pilihan_plot_col], errors="coerce").dropna()

    if not seri_plot.empty:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"**Histogram – {pilihan_plot_col}**")
            fig_h, ax_h = plt.subplots(figsize=(5, 4))
            sns.histplot(seri_plot, kde=True, bins=10, ax=ax_h, color="skyblue")
            ax_h.set_xlabel(pilihan_plot_col)
            ax_h.set_ylabel("Frequency")
            st.pyplot(fig_h)

        with col2:
            st.markdown(f"**Boxplot – {pilihan_plot_col}**")
            fig_b, ax_b = plt.subplots(figsize=(3, 4))
            sns.boxplot(y=seri_plot, ax=ax_b, color="orange")
            ax_b.set_ylabel(pilihan_plot_col)
            st.pyplot(fig_b)
    else:
        st.warning(f"Column {pilihan_plot_col} has no valid numeric data.")
else:
    st.info(TEXT["a3_warning"][st.session_state.lang])

st.markdown("---")

# 4. Association Analysis X and Y
st.header(TEXT["b_header"][st.session_state.lang])

if ("X_total" in df.columns) and ("Y_total" in df.columns):
    st.write("X_total and Y_total have been computed from the items you selected above.")

    method = st.radio(
        "Select association method:",
        ("Pearson Correlation (numeric)", "Spearman Rank Correlation (numeric)", "Chi-square Test (categorical)")
    )

    valid = df[["X_total", "Y_total"]].dropna()

    if valid.empty:
        st.warning("No complete data available for X_total and Y_total.")
    else:
        if "Pearson" in method:
            st.markdown(
                "- Use when variables are numeric (e.g., Likert-scale totals such as X_total and Y_total).\n"
                "- Reports: correlation coefficient (r), p-value, and interpretation (positive/negative, weak/moderate/strong)."
            )

            r, p = stats.pearsonr(valid["X_total"], valid["Y_total"])
            st.subheader("Pearson Correlation")
            st.write(f"r = {r:.3f}")
            st.write(f"p-value = {p:.4f}")

            if r > 0:
                direction = "positive"
            elif r < 0:
                direction = "negative"
            else:
                direction = "no correlation"

            if abs(r) < 0.3:
                strength = "weak"
            elif abs(r) < 0.5:
                strength = "moderate"
            else:
                strength = "strong"

            st.write(f"Interpretation: {direction} correlation with {strength} strength.")

            fig_scatter, ax_scatter = plt.subplots(figsize=(5, 4))
            sns.regplot(x="X_total", y="Y_total", data=valid, ax=ax_scatter, scatter_kws={"alpha": 0.7})
            ax_scatter.set_title("Scatter Plot X_total vs Y_total")
            st.pyplot(fig_scatter)

        elif "Spearman" in method:
            st.markdown(
                "- Use when data are not normally distributed or are ordinal/ranked.\n"
                "- Reports: Spearman correlation coefficient (rho), p-value, and interpretation (positive/negative, weak/moderate/strong)."
            )

            r, p = stats.spearmanr(valid["X_total"], valid["Y_total"])
            st.subheader("Spearman Rank Correlation")
            st.write(f"rho = {r:.3f}")
            st.write(f"p-value = {p:.4f}")

            if r > 0:
                direction = "positive"
            elif r < 0:
                direction = "negative"
            else:
                direction = "no correlation"

            if abs(r) < 0.3:
                strength = "weak"
            elif abs(r) < 0.5:
                strength = "moderate"
            else:
                strength = "strong"

            st.write(f"Interpretation: {direction} correlation with {strength} strength.")

        else:
            st.markdown(
                "- Use when X and Y are categorical variables.\n"
                "- Here, X_total and Y_total are first grouped into categories (e.g., low/medium/high).\n"
                "- Reports: chi-square value, degrees of freedom (df), p-value, and interpretation of association significance."
            )

            st.subheader("Chi-square Test")

            bins = st.slider(
                "Number of categories (bins) to convert X_total and Y_total into categorical variables",
                2, 5, 3
            )

            valid["X_cat"] = pd.qcut(valid["X_total"], q=bins, duplicates="drop")
            valid["Y_cat"] = pd.qcut(valid["Y_total"], q=bins, duplicates="drop")

            ctab = pd.crosstab(valid["X_cat"], valid["Y_cat"])
            st.write("Crosstab:")
            st.dataframe(ctab)

            chi2, p, dof, expected = stats.chi2_contingency(ctab)
            st.write(f"Chi-square = {chi2:.3f}")
            st.write(f"df = {dof}")
            st.write(f"p-value = {p:.4f}")

            st.write("Interpretation: if p-value < 0.05, there is a statistically significant association between X_cat and Y_cat.")
else:
    st.info(TEXT["assoc_info"][st.session_state.lang])
