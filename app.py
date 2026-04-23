import streamlit as st
import pandas as pd
import numpy as np
import io
import base64
from src.preprocessing import preprocess_input
from src.predict import predict

# 页面配置
st.set_page_config(
    page_title="失眠亚型预测系统",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 微调（可选）
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .stButton>button { width: 100%; border-radius: 8px; }
    .prediction-box { padding: 20px; border-radius: 10px; background-color: #e9f5ff; border-left: 5px solid #2196F3; }
</style>
""", unsafe_allow_html=True)

# 侧边栏
with st.sidebar:
    st.title("🧠 失眠亚型分类器")
    st.markdown("""
    **基于静息态心电功能连接特征**  
    使用嵌套交叉验证训练的机器学习模型，实现四亚型自动判别。
    """)
    st.markdown("---")
    st.markdown("### 使用步骤")
    st.markdown("1. 上传包含心电特征的 Excel/CSV 文件")
    st.markdown("2. 系统自动预处理并对齐特征")
    st.markdown("3. 查看预测概率与亚型标签")
    st.markdown("4. 下载带预测结果的数据表")
    st.markdown("---")
    st.info("注意：输入文件应包含与训练完全一致的列（顺序可不同）。", icon="ℹ️")

# 主界面
st.title("失眠症患者亚型预测平台")
st.caption("Insomnia Subtype Classification via Resting-State EEG Functional Connectivity")

# 文件上传
uploaded_file = st.file_uploader(
    "📤 上传待预测数据（.xlsx 或 .csv）",
    type=["xlsx", "csv"],
    accept_multiple_files=False
)

if uploaded_file is not None:
    # 读取数据
    try:
        if uploaded_file.name.endswith('.csv'):
            df_raw = pd.read_csv(uploaded_file)
        else:
            df_raw = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"文件读取失败：{e}")
        st.stop()

    st.success(f"✅ 成功读取 {len(df_raw)} 行 × {df_raw.shape[1]} 列")

    # 展示原始数据预览
    with st.expander("🔍 查看原始数据预览（前5行）", expanded=False):
        st.dataframe(df_raw.head(), use_container_width=True)

    # 执行预处理与预测
    with st.spinner("正在进行特征对齐、标准化与模型推理..."):
        try:
            X_processed, row_ids = preprocess_input(df_raw)
            pred, proba, class_names = predict(X_processed)
        except ValueError as ve:
            st.error(str(ve))
            st.stop()
        except Exception as e:
            st.error(f"预测过程出错：{e}")
            st.stop()

    # 构建结果 DataFrame
    result_df = pd.DataFrame({
        'Sample_ID': row_ids,
        'Predicted_Subtype': [class_names[p] for p in pred],
        'Predicted_Label': pred
    })
    # 追加各类别概率
    for i, name in enumerate(class_names):
        result_df[f'Prob_{name}'] = np.round(proba[:, i], 4)

    # 结果展示
    st.markdown("---")
    st.subheader("📊 预测结果")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.dataframe(
            result_df.style.background_gradient(
                subset=[f'Prob_{c}' for c in class_names],
                cmap='YlGnBu'
            ),
            use_container_width=True,
            height=400
        )

    with col2:
        st.markdown("#### 亚型分布统计")
        subtype_counts = result_df['Predicted_Subtype'].value_counts().reset_index()
        subtype_counts.columns = ['Subtype', 'Count']
        st.bar_chart(subtype_counts.set_index('Subtype'))

        st.markdown("#### 最高置信度样本")
        result_df['Max_Prob'] = result_df[[f'Prob_{c}' for c in class_names]].max(axis=1)
        top_confident = result_df.nlargest(3, 'Max_Prob')[['Sample_ID', 'Predicted_Subtype', 'Max_Prob']]
        st.table(top_confident)

    # 单样本详细解释（可折叠）
    st.markdown("---")
    st.subheader("🔬 单样本概率详情")
    selected_idx = st.selectbox("选择样本查看详情", result_df['Sample_ID'].tolist())
    selected_row = result_df[result_df['Sample_ID'] == selected_idx].iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    for i, name in enumerate(class_names):
        prob = selected_row[f'Prob_{name}']
        with [c1, c2, c3, c4][i]:
            st.metric(label=name, value=f"{prob:.1%}")

    # 下载按钮
    st.markdown("---")
    st.subheader("💾 导出结果")

    # Excel 下载
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        result_df.to_excel(writer, index=False, sheet_name='Predictions')
    output.seek(0)

    st.download_button(
        label="📥 下载预测结果 (.xlsx)",
        data=output,
        file_name="insomnia_subtype_predictions.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    # CSV 下载
    csv = result_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 下载预测结果 (.csv)",
        data=csv,
        file_name="insomnia_subtype_predictions.csv",
        mime="text/csv"
    )

else:
    # 未上传文件时的占位
    st.info("👈 请在左侧上传数据文件以开始预测。", icon="⬆️")
    st.markdown("""
    ### 输入文件格式要求
    | 要求 | 说明 |
    |------|------|
    | 格式 | `.xlsx` 或 `.csv` |
    | 特征列 | 必须与训练数据特征完全一致（列名或列顺序至少匹配一项） |
    | 目标列 | 可选。如果最后一列是标签，系统会自动忽略 |
    | 缺失值 | 系统自动填充为 0 |
    """)