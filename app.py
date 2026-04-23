import streamlit as st
import pandas as pd
import numpy as np
import io
import os
import joblib
import json
import matplotlib.pyplot as plt
import shap

from src.preprocessing import preprocess_input
from src.predict import predict

# ==================== 加载模型与元数据 ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PIPELINE = joblib.load(os.path.join(BASE_DIR, 'model', 'model.pkl'))
SCALER = PIPELINE.named_steps['scaler']
MODEL = PIPELINE.named_steps['model']

COEF = MODEL.coef_

with open(os.path.join(BASE_DIR, 'model', 'feature_names.json'), 'r', encoding='utf-8') as f:
    FEATURE_NAMES = json.load(f)

with open(os.path.join(BASE_DIR, 'model', 'class_labels.json'), 'r', encoding='utf-8') as f:
    CLASS_INFO = json.load(f)

train_stats_path = os.path.join(BASE_DIR, 'model', 'train_stats.json')
TRAIN_STATS = pd.read_json(train_stats_path) if os.path.exists(train_stats_path) else None

background_path = os.path.join(BASE_DIR, 'model', 'background.npy')
BACKGROUND = np.load(background_path) if os.path.exists(background_path) else None

# ==================== 页面配置 ====================
st.set_page_config(
    page_title="失眠亚型预测系统",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .stButton>button { width: 100%; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ==================== 侧边栏 ====================
with st.sidebar:
    st.title("🧠 失眠亚型分类器")
    st.markdown("""
    **基于静息态心电功能连接特征**  
    使用嵌套交叉验证训练的 Logistic Regression 模型，实现四亚型自动判别。
    """)
    st.markdown("---")
    st.markdown("### 使用步骤")
    st.markdown("1. 上传包含心电特征的 Excel/CSV 文件")
    st.markdown("2. 系统自动预处理并对齐特征")
    st.markdown("3. 查看预测概率与亚型标签")
    st.markdown("4. 下载带预测结果的数据表")
    st.markdown("---")
    
    example_path = os.path.join(BASE_DIR, 'model', 'example_input.xlsx')
    if os.path.exists(example_path):
        with open(example_path, 'rb') as f:
            st.download_button(
                label="📥 下载示例输入模板",
                data=f,
                file_name="example_input.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    
    st.markdown("---")
    st.info(f"模型特征数：{len(FEATURE_NAMES)}", icon="ℹ️")

# ==================== 主界面 ====================
st.title("失眠症患者亚型预测平台")
st.caption("Insomnia Subtype Classification via Resting-State EEG Functional Connectivity")

uploaded_file = st.file_uploader(
    "📤 上传待预测数据（.xlsx 或 .csv）",
    type=["xlsx", "csv"],
    accept_multiple_files=False
)

if uploaded_file is not None:
    # -------------------- 读取与预测 --------------------
    try:
        if uploaded_file.name.endswith('.csv'):
            df_raw = pd.read_csv(uploaded_file)
        else:
            df_raw = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"文件读取失败：{e}")
        st.stop()

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

    # -------------------- 构建结果表 --------------------
    result_df = pd.DataFrame({
        'Sample_ID': row_ids,
        'Predicted_Subtype': [class_names[p] for p in pred],
        'Predicted_Label': pred
    })
    for i, name in enumerate(class_names):
        result_df[f'Prob_{name}'] = np.round(proba[:, i], 4)

    # 用于显示的列（隐藏 Sample_ID 和 Max_Prob）
    display_cols = [c for c in result_df.columns if c not in ['Sample_ID', 'Max_Prob']]

    # -------------------- 并排布局：表格 + 单样本解析 --------------------
    st.markdown("---")
    
    col_table, col_right = st.columns([1.5, 2])
    
    # ===== 左侧：预测结果总表 =====
    with col_table:
        st.markdown("#### 📊 预测结果")
        
        format_dict = {f'Prob_{c}': "{:.2%}" for c in class_names}
        try:
            styled = result_df[display_cols].style.background_gradient(
                subset=[f'Prob_{c}' for c in class_names],
                cmap='YlGnBu',
                vmin=0, vmax=1
            ).format(format_dict)
            st.dataframe(
                styled,
                use_container_width=True,
                height=min(400, 35 * len(result_df) + 50)
            )
        except Exception:
            st.dataframe(result_df[display_cols], use_container_width=True)
    
    # ===== 右侧：单样本解析 =====
    with col_right:
        st.markdown("#### 🔬 单样本解析")
        
        # 横跨第二、三栏的下拉选择器
        selected_idx = st.selectbox("选择样本查看详情", result_df['Sample_ID'].tolist())
        
        # 预计算公共变量
        sample_pos = row_ids.index(selected_idx)
        pred_label = int(result_df[result_df['Sample_ID'] == selected_idx]['Predicted_Label'].values[0])
        x_raw = X_processed[sample_pos].copy()
        x_std = SCALER.transform(X_processed[sample_pos:sample_pos+1])[0]
        
        # SHAP 预计算
        sv = None
        if BACKGROUND is not None:
            try:
                shap_explainer = shap.Explainer(MODEL, BACKGROUND)
                sv = shap_explainer(x_std.reshape(1, -1))
            except Exception:
                pass
        
        # 漂移预计算
        z_scores = None
        if TRAIN_STATS is not None:
            means = TRAIN_STATS['mean'].values
            stds = TRAIN_STATS['std'].values
            z_scores = (x_raw - means) / stds
        
        # 内部分两列：SHAP + 漂移
        c_shap, c_drift = st.columns(2)
        
        with c_shap:
            st.markdown(f"**{class_names[pred_label]}**")
            st.caption("SHAP 特征贡献")
            
            if sv is not None:
                exp = shap.Explanation(
                    values=sv.values[0, :, pred_label],
                    base_values=sv.base_values[0, pred_label],
                    data=x_std,
                    feature_names=FEATURE_NAMES
                )
                fig = plt.figure(figsize=(5, 8))
                shap.plots.waterfall(exp, max_display=len(FEATURE_NAMES), show=False)
                fig = plt.gcf()
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.warning("未找到背景数据")
        
        with c_drift:
            st.markdown(f"**{class_names[pred_label]}**")
            st.caption("数据漂移检测")
            
            if z_scores is not None:
                fig, ax = plt.subplots(figsize=(5, 8))
                colors = ['#d62728' if abs(z) > 2 else '#ff7f0e' if abs(z) > 1 else '#2ca02c' 
                          for z in z_scores]
                ax.barh(
                    range(len(FEATURE_NAMES)),
                    z_scores,
                    color=colors,
                    alpha=0.7,
                    edgecolor='none'
                )
                ax.set_yticks(range(len(FEATURE_NAMES)))
                ax.set_yticklabels(FEATURE_NAMES, fontsize=8)
                ax.invert_yaxis()
                ax.set_xlabel("Z-score", fontsize=9)
                ax.axvline(x=0, color='black', linewidth=0.8)
                ax.axvline(x=2, color='red', linestyle='--', alpha=0.5)
                ax.axvline(x=-2, color='red', linestyle='--', alpha=0.5)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
                
                abnormal = np.where(np.abs(z_scores) > 2)[0]
                if len(abnormal) > 0:
                    st.warning(f"⚠️ {len(abnormal)} 个特征偏离 >2σ")
                else:
                    st.success("✅ 分布正常")
            else:
                st.info("无训练统计")

    # -------------------- 下方明细表（可折叠） --------------------
    st.markdown("---")
    if sv is not None:
        with st.expander("📋 查看 SHAP 数值明细与模型系数"):
            shap_df = pd.DataFrame({
                'Feature': FEATURE_NAMES,
                'Raw_Value': np.round(x_raw, 3),
                'Std_Value': np.round(x_std, 3),
                'SHAP_Value': np.round(sv.values[0, :, pred_label], 4),
                'Direction': ['🔴 推动' if v > 0 else '🔵 抑制' 
                              for v in sv.values[0, :, pred_label]]
            }).sort_values('SHAP_Value', key=abs, ascending=False)
            st.dataframe(shap_df, use_container_width=True, hide_index=True)

            coef_df = pd.DataFrame({
                'Feature': FEATURE_NAMES,
                'Coefficient': np.round(COEF[pred_label], 4),
                'Abs_Coef': np.round(np.abs(COEF[pred_label]), 4)
            }).sort_values('Abs_Coef', ascending=False)
            st.dataframe(coef_df, use_container_width=True, hide_index=True)

    if z_scores is not None:
        with st.expander("📐 查看数据漂移明细"):
            drift_df = pd.DataFrame({
                'Feature': FEATURE_NAMES,
                'Train_Mean': np.round(TRAIN_STATS['mean'].values, 3),
                'Train_Std': np.round(TRAIN_STATS['std'].values, 3),
                'Current_Value': np.round(x_raw, 3),
                'Z_Score': np.round(z_scores, 2)
            }).sort_values('Z_Score', key=abs, ascending=False)
            st.dataframe(drift_df, use_container_width=True, hide_index=True)

    # -------------------- 导出 --------------------
    st.markdown("---")
    st.subheader("💾 导出结果")

    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
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
    with col_dl2:
        csv = result_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 下载预测结果 (.csv)",
            data=csv,
            file_name="insomnia_subtype_predictions.csv",
            mime="text/csv"
        )

else:
    st.info("👈 请在左侧上传数据文件，或下载示例模板以开始预测。", icon="⬆️")
    st.markdown("""
    ### 输入文件格式要求
    | 要求 | 说明 |
    |------|------|
    | 格式 | `.xlsx` 或 `.csv` |
    | 特征列 | 必须与训练数据特征完全一致（共 **11** 列） |
    | 目标列 | 可选。若最后一列为标签，系统会自动忽略 |
    | 缺失值 | 系统自动填充为 0 |
    | 异常值 | 无穷大(Inf) 会被替换为 0 |
    """)
