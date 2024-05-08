import streamlit as st
import streamlit.components.v1 as components

import shap
import numpy as np
import matplotlib.pyplot as plt
from langchain.agents import Tool
from langchain_experimental.tools.python.tool import PythonREPLTool
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import StrOutputParser
import xgboost as xgb

def get_tools(model, X_train):
    html_path = 'charts/altair_chart.html'
    # 执行python命令的工具
    python_tool = PythonREPLTool()
    python_tool_desc = (
        "Use this tool when you need to execute python commands to obtain data or plot charts. "
        "Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`. "
        "You can keep reusing this tool until you get the final answer. "
        "For chart plotting, the altair library is mandatory. "
        "Instead of displaying your chart directly, save it using the `chart.save('charts/altair_chart.html')` "
        "After you create a chart, you must call the chart_display_tools to display the chart."
    )
    python_tool.description = python_tool_desc

    #展示图表的工具
    def chart_display(html_path='charts/altair_chart.html'):
        """
        Load and display a saved Altair chart from an HTML file.

        Parameters:
        - html_path (str): The file path of the HTML chart to be loaded and displayed.
        """
        try:
            # 读取 HTML 文件内容
            with open(html_path, 'r', encoding='utf-8') as f:
                chart_html = f.read()

            # 使用 Streamlit components 嵌入 HTML 文件内容
            components.html(chart_html, height=500)

        except FileNotFoundError:
            st.error(f"The chart HTML file could not be found at {html_path}. Please ensure the path is correct.")
        except Exception as e:
            st.error(f"An error occurred while displaying the chart: {e}")

    chart_display_tool = Tool(
        name='chart_display_tool',
        func=chart_display,
        description="Use this tool after you calling python_tool to create a chart."
    )

    # 解释器初始化
    explainer = shap.TreeExplainer(model)

    feature_names = X_train.columns.tolist()

    # 得到local explain的图工具
    def local_exp(input):
        # 将输入字符串转换为 Python 数据结构（如列表或 numpy 数组）
        sample = eval(input)

        # 确保 sample 为二维数组
        if isinstance(sample, list):
            # 转换为 numpy 数组
            sample = np.array(sample)

        # 如果只有一个样本，则将其变成二维的 `[1, n_features]`
        if sample.ndim == 1:
            sample = sample.reshape(1, -1)

        # 调用 explainer 并计算 SHAP 值
        shap_values = explainer.shap_values(sample)
        expected_value = explainer.expected_value

        # 使用 matplotlib 绘制 force plot 并保存
        if sample.ndim == 2 and sample.shape[0] > 0:
            # 使用 shap.force_plot 直接返回可视化对象
            shap.initjs()  # 初始化 JavaScript 环境以支持在网页中嵌入 SHAP 图
            force_plot = shap.force_plot(expected_value, shap_values[0], sample[0],
                                                feature_names=X_train.columns.tolist(), matplotlib=True)
            st.pyplot(force_plot)

    # 工具配置
    explanation_tool = Tool(
        name='Explanation_tool',
        func=local_exp,
        description="Use this tool when you need to get the contribution of each feature to the prediction outcome of a sample. "
                    "Input should be a list of feature values. It will return a visualization of the feature contribution."
    )

    def forecast_tool(input_str, model, explainer, feature_names):
        try:
            # 将输入字符串转换为实际的输入数据
            input_data = eval(input_str)
            # 确保输入是二维数组（即使是一个样本）
            if isinstance(input_data[0], list):
                input_data = np.array(input_data)
            else:
                input_data = np.array([input_data])

            # 使用 XGBoost 的 DMatrix 进行数据转换
            dmatrix = xgb.DMatrix(input_data, feature_names=feature_names)

            # 获取模型的预测结果
            predictions = model.predict(dmatrix)

            # 判断预测结果维度以确保处理多分类和二分类任务
            if predictions.ndim == 2 and predictions.shape[1] > 1:  # 多分类
                predicted_class = np.argmax(predictions, axis=1)
                probabilities = predictions
            else:  # 二分类
                predicted_class = (predictions > 0.5).astype(int)
                probabilities = predictions

            # 使用 SHAP 获取每个预测的解释
            shap_values = explainer.shap_values(dmatrix)

            # 构建输出结果
            results = []
            for idx, prediction in enumerate(predicted_class):
                result = {
                    'prediction': prediction,
                    'probability': probabilities[idx],
                    'shap_values': shap_values[idx]
                }
                results.append(result)

                # 使用 force_plot 并嵌入 Streamlit
                shap.initjs()  # 初始化 JavaScript 环境以支持在网页中嵌入 SHAP 图
                force_plot = shap.force_plot(explainer.expected_value, shap_values[idx], input_data[idx],
                                                  feature_names=feature_names,  matplotlib=True)
                st.pyplot(force_plot)

            return results
        except Exception as e:
            return str(e)

    forecast_description = "Use this tool to predict the outcomes based on input features. Input should be a list of feature values or a list of lists for multiple samples."
    forecast = Tool(
        name='Forecast',
        func=lambda input: forecast_tool(input, model, explainer, feature_names),
        description=forecast_description
    )

    # SHAP 总结图工具
    def shap_summary(input):
        try:
            # 绘制总结图
            shap_values = explainer.shap_values(X_train)
            shap.summary_plot(shap_values, X_train, show=False)
            st.pyplot(bbox_inches='tight')
        except Exception as e:
            return str(e)

    shap_summary_tool = Tool(
        name='SHAP_Summary',
        func=shap_summary,
        description="Generate and display the SHAP summary plot. Use this tool when users need global explanations or pay attention to all the features."
    )

    # SHAP 条形图工具
    def shap_bar(input):
        try:
            # 绘制条形图
            shap_values = explainer.shap_values(X_train)
            shap.summary_plot(shap_values, X_train, plot_type='bar', show=False)
            st.pyplot(bbox_inches='tight')
        except Exception as e:
            return str(e)

    shap_bar_tool = Tool(
        name='SHAP_Bar',
        func=shap_bar,
        description="Generate and display the SHAP bar plot. Use this tool when users need global explanations or pay attention to all the features."
    )

    # SHAP 依赖图工具
    def shap_dependence(input):
        try:
            # 绘制单个特征的依赖图
            feature_name = input.strip()
            shap_values = explainer.shap_values(X_train)
            shap.dependence_plot(feature_name, shap_values, X_train, show=False)
            st.pyplot(bbox_inches='tight')
        except Exception as e:
            return str(e)

    shap_dependence_tool = Tool(
        name='SHAP_Dependence',
        func=shap_dependence,
        description="Generate and display the SHAP dependence plot for a specified feature. Input should be the feature name.Use this tool when users pay attention to a specified feature."
    )

    # 直接回答的工具
    def final(input):
        return input

    Final_answer = Tool(
        name='Final_answer',
        func=final,
        description="Use this if you want to respond directly to the human. Input should be what you want to respond and it will return the same. After using this tool, you must leave the results returned by this tool intact as Final Answer instead of continuing with other actions."
    )

    # 任务分解的工具
    llm_gpt4 = ChatOpenAI(model_name="gpt-4", temperature=0)
    prompt = PromptTemplate.from_template(
        """
        You are a researcher working with an interpretable deep model, and your goal is to break down abstract tasks into specific work steps and processes. The task involves performing various operations related to training and predicting samples. The model you use to predict is XGBoost and the model you use to get feature impoetance is SHAP. Here are the specific operations you can perform:

        1. Execute Python code: You can use Python code to perform various functions, including plotting and viewing data in a DataFrame. The DataFrame contains the collection of samples to be predicted.

        2. Obtain predictions for specific samples: You can get the prediction results for one or more samples.

        3. Analyze the contribution of each feature to the prediction results: You can determine the contribution of each feature to the prediction results for a specific sample.

        4. Plot the contribution of each feature for different feature values: You can create visualizations that show the contribution of each feature to the results for different values of the feature.

        Additionally, the prompt includes basic operations required for interpretable analysis.

        Please choose the appropriate operation based on your specific needs and follow the prompts. Keep it short enough to include just one sentence for each step. You only need to give the overall process and steps, without writing down the specific operations required for a certain step and the meaning of each step.

        The task is {query}
        """
    )
    runnable = prompt | llm_gpt4 | StrOutputParser()

    def decomposition(query):
        return runnable.invoke({"query": query})

    decomposition_desc = "This tool must be used first when you receive a task at the beginning and attempt to complete it. You'll get step-by-step instructions for completing the task. You can refer to this guidance without blindly following it."

    Task_decom_tool = Tool(
        name='Task_decomposition',
        func=decomposition,
        description=decomposition_desc
    )

    # 获得所有的工具的列表
    tools = []
    tools.append(python_tool)
    tools.append(explanation_tool)
    tools.append(Final_answer)
    tools.append(forecast)
    tools.append(Task_decom_tool)
    tools.append(chart_display_tool)
    tools.append(shap_summary_tool)
    tools.append(shap_bar_tool)
    tools.append(shap_dependence_tool)

    return tools
