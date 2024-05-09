from langchain.agents import AgentType
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_community.chat_models import ChatOpenAI
import streamlit as st
from ydata_profiling import ProfileReport
import pandas as pd
import os
import numpy as np
from streamlit_pandas_profiling import st_profile_report
from sklearn.model_selection import train_test_split
from ebm_agent import get_agent
from shap_agent import shap_get_agent
from interpret.glassbox import ExplainableBoostingClassifier
from langchain.memory import ConversationBufferMemory
from langchain.schema import messages
from langchain.callbacks.base import BaseCallbackHandler
from dataset_description import get_data_description
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import seaborn as sns


y_axis_description = "The y-axis depicts contributions in log-odds towards the outcome, that is the probability that the person makes over 50K a year."

def clear_submit():
    """
    Clear the Submit Button State
    Returns:

    """
    st.session_state["submit"] = False

# 文件格式支持
file_formats = {
    "csv": pd.read_csv,
    "xls": pd.read_excel,
    "xlsx": pd.read_excel,
    "xlsm": pd.read_excel,
    "xlsb": pd.read_excel,
}

def main():
    st.set_page_config(page_title="InterpretableGPT", page_icon="🦜")
    #col1, col2, col3 = st.columns([0.2, 0.5, 0.3])
    # 使用Google Fonts
    google_fonts = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Lobster&display=swap');

    .custom-google-font1 {
        font-family: 'Lobster', cursive;
        font-size: 56px;
        color: #FFFFFF;
    }
    .custom-google-font2 {
        font-family: 'Lobster', cursive;
        font-size: 36px;
        color: #7E7EB1;
    }
    </style>

    <h1 class='custom-google-font1'>TabularPrism</h1>
    <h2 class='custom-google-font2'>An Interpretable llm data analyst</h2>
    """

    st.markdown(google_fonts, unsafe_allow_html=True)

    setup_api_key()
    uploaded_file = st.file_uploader("upload data file", type=list(file_formats.keys()))
    rag_files = st.file_uploader("upload knowledge documents",
                                      accept_multiple_files=True,
                                      type=["pdf", "text", "txt"],
                                      key="a")

    if uploaded_file:
        df = load_data(uploaded_file)
        llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo", openai_api_key=st.session_state['api_key'], streaming=True)
        data_description = get_data_description(llm, df)
        data_description = st.sidebar.text_area(label="Data Description", value=data_description, height=300, max_chars=8000, help="You can edit the data description")
        if df is not None and st.sidebar.button('Exploratory Data Analysis ', type="primary"):
            display_data_profile(df)

        selected_model = st.selectbox("Select XAI model", ["EBM", "SHAP"], index=None)
        if selected_model == "EBM":
            model, df1 = ebm_train(df)
            #st.sidebar.text("EBM has been trained on data")
            handle_chat_interaction(df1, model, data_description)
        elif selected_model == "SHAP":
            model, X_train = XGBoost_train(df)
            #st.sidebar.text("XGBoost has been trained on data")
            #SHAP_explainer(model, X_train)
            shap_chat_interaction(X_train, model, data_description)
            #功能待完成

def setup_api_key():
    if 'api_key' not in st.session_state:
        st.session_state['api_key'] = os.getenv("OPENAI_API_KEY")
    #modified_key = st.sidebar.text_input("OpenAI API Key:old_key:", type="password", value=st.session_state['api_key'])
    #if modified_key:
    #    st.session_state['api_key'] = modified_key

@st.cache_data(ttl=7200)
def load_data(uploaded_file):
    ext = os.path.splitext(uploaded_file.name)[1][1:].lower()
    if ext in file_formats:
        return file_formats[ext](uploaded_file)
    else:
        st.error(f"不支持的文件格式: {ext}")
        return None

def display_data_profile(df):
    pr = ProfileReport(df, title="数据分析报告")
    with st.sidebar.expander("Data Profile Report", expanded=True):
        st_profile_report(pr)

@st.cache_data
def ebm_train(df):
    # 数据分割
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    seed = 42
    np.random.seed(seed)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=seed)

    # 设置 EBM 参数以提高模型性能
    ebm = ExplainableBoostingClassifier(
        interactions=15,  # 允许更多交互
        max_bins=256,    # 提高离散化 bins 数量
        max_interaction_bins=64,  # 增加交互 bins 数量
        max_rounds=800  # 增加训练轮数
    )

    # 模型训练
    ebm.fit(X_train, y_train)

    # 模型评估
    y_pred = ebm.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    confusion = confusion_matrix(y_test, y_pred)

    # 在前端显示测试结果
    st.sidebar.write("EBM has been successfully trained. ✔️")
    st.sidebar.write("Accuracy:", report['accuracy'])
    st.sidebar.write("Classification Report:")
    st.sidebar.json(report)

    st.sidebar.write("Confusion metrix:")
    #st.sidebar.write(confusion)
    # 可视化混淆矩阵
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted 0', 'Predicted 1'],
                yticklabels=['Actual 0', 'Actual 1'])
    st.sidebar.pyplot()

    return ebm, X

def handle_chat_interaction(df, ebm, data_description):
    if 'messages' not in st.session_state:
        st.session_state['messages'] = [{"role": "assistant", "content": "How can I help you?"}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input("Ask questions about data："):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        if st.session_state['api_key']:
            response = process_prompt_with_agent(prompt, df, ebm, data_description)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.chat_message("assistant").write(response)
            manage_feedback(df, ebm)


def process_prompt_with_agent(prompt, df, ebm, data_description):
    llm = ChatOpenAI(temperature=0, model="gpt-4-turbo", openai_api_key=st.session_state['api_key'], streaming=True)
    eag = get_agent(llm, ebm, df=df, dataset_description=data_description, y_axis_description=y_axis_description)
    st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
    return eag.run(prompt, callbacks=[st_cb])


def manage_feedback(df, ebm):
    feedback = st.sidebar.radio("Does this answer meet your expectations?", (':+1: Yes', ':shit: No'), index=None, key='feedback')
    if feedback == ':shit: No':
        feedback_details = st.sidebar.text_area("请提供反馈，帮助我们改进解释：", key='feedback_details')
        if st.sidebar.button("根据反馈重新生成解释", key='feedback_button'):
            if feedback_details:
                new_explanation = process_prompt_with_agent(feedback_details, df, ebm)
                st.session_state.messages.append({"role": "assistant", "content": new_explanation})
                st.chat_message("assistant").write("根据反馈调整后的解释结果:", new_explanation)
                st.session_state['feedback_details'] = ''  # Reset feedback details after processing
                st.experimental_rerun()  # Optional: trigger a rerun to refresh the state
            else:
                st.sidebar.error("请输入反馈详情。")

@st.cache_data
def XGBoost_train(df):
    label_encoder = LabelEncoder()
    df['income'] = label_encoder.fit_transform(df['income'])

    # 为每个分类特征独立实例化 LabelEncoder
    categorical_columns = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race',
                           'gender', 'native-country']
    encoders = {col: LabelEncoder() for col in categorical_columns}
    for column in categorical_columns:
        df[column] = encoders[column].fit_transform(df[column])

    # 分离特征和目标变量
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # 创建 DMatrix
    dtrain = xgb.DMatrix(data=X_train, label=y_train)
    dtest = xgb.DMatrix(data=X_test, label=y_test)

    params = {
        #'objective': 'multi:softprob',  # 多分类任务
        #'num_class': len(np.unique(y)),  # 类别数量
        'objective': 'binary:logistic',  # 二分类任务目标
        'eval_metric': 'logloss',  # 评估指标
        'max_depth': 7,  # 树的最大深度
        'eta': 0.1,  # 学习率
    }
    model = xgb.train(params, dtrain, num_boost_round=100)
    # 进行预测
    y_pred = model.predict(dtest)
    y_pred_binary = [1 if pred > 0.5 else 0 for pred in y_pred]

    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred_binary)

    # 显示测试结果
    st.sidebar.write("XGBoost has been successfully trained. ✔️")
    st.sidebar.write("Accuracy:", accuracy)
    #st.sidebar.write("Classification Report:")
    #st.sidebar.write(classification_report(y_test, y_pred_binary))

    # 计算并显示混淆矩阵
    cm = confusion_matrix(y_test, y_pred_binary)
    st.sidebar.write("Confusion Matrix:")
    #st.sidebar.write(pd.DataFrame(cm, columns=['Predicted 0', 'Predicted 1'], index=['Actual 0', 'Actual 1']))

    # 可视化混淆矩阵
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted 0', 'Predicted 1'],
                yticklabels=['Actual 0', 'Actual 1'])
    st.sidebar.pyplot()
    return model, X_train

def SHAP_explainer(model, X_train):
    # 使用 TreeExplainer 生成 SHAP 值
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)

    # 打印数据和 SHAP 值形状以调试维度不匹配的问题
    st.write(f"SHAP values shape: {shap_values.shape}")
    st.write(f"X_train shape: {X_train.shape}")

    # 确保 SHAP 值的行数与 X_train 一致
    if shap_values.shape[0] != X_train.shape[0] or shap_values.shape[1] != X_train.shape[1]:
        st.write("Error: SHAP values do not match input data dimensions.")
        return

    # SHAP 总结图
    st.write("SHAP Summary Plot:")
    shap.summary_plot(shap_values, X_train, show=False)
    st.pyplot(bbox_inches='tight')

    # SHAP 特征重要性条形图
    st.write("SHAP Feature Importance Bar Plot:")
    shap.summary_plot(shap_values, X_train, plot_type="bar", show=False)
    st.pyplot(bbox_inches='tight')

    # SHAP 依赖图
    st.write("SHAP Dependence Plots for each feature:")
    for feature_name in X_train.columns:
        try:
            # 确保传入的特征名称与数据一致
            if feature_name not in X_train.columns:
                st.write(f"Feature {feature_name} not found in input data")
                continue

            # 绘制依赖图
            shap.dependence_plot(feature_name, shap_values, X_train, show=False)
            st.pyplot(bbox_inches='tight')
        except Exception as e:
            st.write(f"Error plotting dependence plot for feature {feature_name}: {e}")

    # SHAP 力量图
    st.write("SHAP Force Plot for the first sample:")
    shap.initjs()  # 初始化 JavaScript 可视化
    force_plot = shap.force_plot(explainer.expected_value, shap_values[0, :], X_train.iloc[0, :], matplotlib=True)
    st.pyplot(force_plot)

def shap_chat_interaction(df, model, data_description):
    if 'messages' not in st.session_state:
        st.session_state['messages'] = [{"role": "assistant", "content": "How can I help you?"}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input("Ask questions about data："):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        if st.session_state['api_key']:
            response = process_prompt_with_shap_agent(prompt, df, model, data_description)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.chat_message("assistant").write(response)
            shap_manage_feedback(df, model)


def process_prompt_with_shap_agent(prompt, df, model, data_description):
    llm = ChatOpenAI(temperature=0, model="gpt-4-turbo", openai_api_key=st.session_state['api_key'], streaming=True)
    eag = shap_get_agent(llm, model, df=df, dataset_description=data_description, y_axis_description=y_axis_description)
    st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
    return eag.run(prompt, callbacks=[st_cb])


def shap_manage_feedback(df, model):
    feedback = st.sidebar.radio("Does this answer meet your expectations?", (':+1: Yes', ':shit: No'), index=None, key='feedback')
    if feedback == ':shit: No':
        feedback_details = st.sidebar.text_area("Please provide feedback to improve:", key='feedback_details')


if __name__ == "__main__":
    main()




