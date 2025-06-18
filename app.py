import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

st.set_page_config(page_title="Phân tích dữ liệu cân", layout="wide")

st.title("Ứng dụng Phân tích Dữ liệu Cân - Histogram, Control Chart,...")

# 1. Upload file CSV
uploaded_file = st.file_uploader("Chọn file CSV chứa dữ liệu cân", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    # Xóa cột rác nếu có (cột Unnamed)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    st.write("### Dữ liệu tải lên")
    st.dataframe(df.head(10))

    # Lấy 3 cột cuối làm mặc định
    cols = df.columns[-3:].tolist()
    st.write(f"**3 cột cuối được chọn để phân tích:** {cols}")

    # Cho phép chọn cột (nếu muốn)
    selected_cols = st.multiselect("Chọn các cột cân để phân tích", options=df.columns.tolist(), default=cols)

    # # 2. Nhập giới hạn USL, LSL cho từng cột đã chọn
    # st.write("### Nhập giới hạn USL và LSL cho từng cột")
    # limits = {}

    # if analysis_mode == "Từng cân riêng biệt":
    #     st.write("### Nhập giới hạn USL và LSL cho từng cân")
    #     for col in selected_cols:
    #         with st.expander(f"Giới hạn cho {col}", expanded=True):
    #         lsl = st.number_input(f"LSL cho {col}", value=float(df[col].min()), key=f"{col}_lsl")
    #         usl = st.number_input(f"USL cho {col}", value=float(df[col].max()), key=f"{col}_usl")
    #         limits[col] = {'LSL': lsl, 'USL': usl}
    # else:
    #     st.write("### Nhập giới hạn USL và LSL chung cho cả 3 cân")
    #     lsl_common = st.number_input("LSL chung", value=min(df[selected_cols].min()), key="common_lsl")
    #     usl_common = st.number_input("USL chung", value=max(df[selected_cols].max()), key="common_usl")
    #     for col in selected_cols:
    #         limits[col] = {'LSL': lsl_common, 'USL': usl_common}


    # 3. Chọn biểu đồ muốn vẽ
    st.write("### Chọn biểu đồ muốn hiển thị")
    chart_options = ['Histogram', 'Control Chart (X̄)', 'Individuals Chart', 'Range Chart (R)']
    selected_charts = st.multiselect("Loại biểu đồ", options=chart_options, default=['Histogram'])

    # 4. Chọn phân tích riêng từng cột hay tổng hợp
    analysis_mode = st.radio("Chọn chế độ phân tích:", options=["Từng cân riêng biệt", "Phân tích chung cả 3 cân"], index=0)

    # 2. Nhập giới hạn USL, LSL cho từng cột đã chọn
    st.write("### Nhập giới hạn USL và LSL cho từng cột")
    limits = {}

    if analysis_mode == "Từng cân riêng biệt":
        st.write("### Nhập giới hạn USL và LSL cho từng cân")
        for col in selected_cols:
            with st.expander(f"Giới hạn cho {col}", expanded=True):
                lsl = st.number_input(f"LSL cho {col}", value=float(df[col].min()), key=f"{col}_lsl")
                usl = st.number_input(f"USL cho {col}", value=float(df[col].max()), key=f"{col}_usl")
                limits[col] = {'LSL': lsl, 'USL': usl}
    else:
        st.write("### Nhập giới hạn USL và LSL chung cho cả 3 cân")
        lsl_common = st.number_input("LSL chung", value=min(df[selected_cols].min()), key="common_lsl")
        usl_common = st.number_input("USL chung", value=max(df[selected_cols].max()), key="common_usl")
        for col in selected_cols:
            limits[col] = {'LSL': lsl_common, 'USL': usl_common}

    if st.button("Bắt đầu phân tích"):
        st.write("### Kết quả phân tích")

        def calc_sigma_level(yield_ratio):
            # Tính z-score tương ứng yield (two-sided)
            return norm.ppf(yield_ratio + (1 - yield_ratio)/2)

        def plot_histogram(data, col, LSL, USL):
            mu = data.mean()
            sigma = data.std()
            within_limits = data[(data >= LSL) & (data <= USL)]
            yield_ratio = len(within_limits) / len(data)
            sigma_level = calc_sigma_level(yield_ratio)

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.hist(data, bins=20, density=True, alpha=0.6, color='skyblue', edgecolor='black')
            x = np.linspace(min(data), max(data), 1000)
            ax.plot(x, norm.pdf(x, mu, sigma), 'r--', label='Phân phối chuẩn')
            ax.axvline(LSL, color='orange', linestyle='--', label='LSL')
            ax.axvline(USL, color='green', linestyle='--', label='USL')
            ax.set_title(f"Histogram + Normal Distribution - {col}")
            ax.legend()
            ax.grid(True)

            # Thông số
            text = f"μ = {mu:.2f}\nσ = {sigma:.2f}\nYield = {yield_ratio:.2%}\nSigma Level = {sigma_level:.2f}"
            plt.gcf().text(0.75, 0.75, text, fontsize=10, bbox=dict(facecolor='white', alpha=0.6))

            st.pyplot(fig)

        def plot_individual_chart(data, col):
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.plot(data.values, marker='o', linestyle='-', color='blue')
            ax.set_title(f'Individuals Chart - {col}')
            ax.set_xlabel('Sample Index')
            ax.set_ylabel('Giá trị')
            ax.grid(True)
            st.pyplot(fig)

        def plot_range_chart(data, col):
            if len(data) < 2:
                st.warning(f"Không đủ dữ liệu để vẽ Range Chart cho {col}")
                return
            # Tính biên độ giữa các mẫu kế tiếp
            ranges = data.diff().abs().dropna()
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.plot(ranges.values, marker='o', linestyle='-', color='purple')
            ax.set_title(f'Range Chart (R) - {col}')
            ax.set_xlabel('Sample Index')
            ax.set_ylabel('Range (Biên độ)')
            ax.grid(True)
            st.pyplot(fig)


        def plot_xbar_chart(data, col, subgroup_size=5):
            # Tính trung bình nhóm con size=subgroup_size
            if len(data) < subgroup_size:
                st.warning(f"Không đủ dữ liệu để vẽ X̄ Chart cho {col}")
                return
            groups = data.groupby(data.index // subgroup_size)
            xbar = groups.mean()
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.plot(xbar.values, marker='o', linestyle='-', color='green')
            ax.set_title(f'X̄ Chart (Mean Control Chart) - {col} (nhóm size={subgroup_size})')
            ax.set_xlabel('Nhóm mẫu')
            ax.set_ylabel('Trung bình')
            ax.grid(True)
            st.pyplot(fig)

        if analysis_mode == "Từng cân riêng biệt":
            for col in selected_cols:
                st.markdown(f"---\n## Phân tích cho cân: **{col}**")
                data = df[col].dropna()
                LSL = limits[col]['LSL']
                USL = limits[col]['USL']

                if 'Histogram' in selected_charts:
                    plot_histogram(data, col, LSL, USL)
                if 'Individuals Chart' in selected_charts:
                    plot_individual_chart(data, col)
                if 'Range Chart (R)' in selected_charts:
                    plot_range_chart(data, col)
                if 'Control Chart (X̄)' in selected_charts:
                    plot_xbar_chart(data, col)
        else:
            # Phân tích chung cả 3 cân: gộp dữ liệu
            st.markdown(f"---\n## Phân tích chung cả 3 cân")
            combined_data = pd.DataFrame()
            for col in selected_cols:
                combined_data[col] = df[col].dropna().reset_index(drop=True)
            combined_data = combined_data.dropna()

            # Gộp dữ liệu thành 1 cột để vẽ histogram chung
            all_data = combined_data.values.flatten()

            # Dùng trung bình LSL, USL của 3 cột
            avg_LSL = np.mean([limits[col]['LSL'] for col in selected_cols])
            avg_USL = np.mean([limits[col]['USL'] for col in selected_cols])

            if 'Histogram' in selected_charts:
                plot_histogram(pd.Series(all_data), "3 cân chung", avg_LSL, avg_USL)
            if 'Individuals Chart' in selected_charts:
                plot_individual_chart(pd.Series(all_data), "3 cân chung")
            if 'Range Chart (R)' in selected_charts:
                plot_range_chart(pd.Series(all_data), "3 cân chung")
            if 'Control Chart (X̄)' in selected_charts:
                plot_xbar_chart(pd.Series(all_data), "3 cân chung")
            


else:
    st.info("Vui lòng tải lên file CSV chứa dữ liệu cân để bắt đầu phân tích.")
