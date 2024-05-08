import lime
import lime.lime_tabular
import numpy as np
from langchain.chat_models import ChatOpenAI
from langchain.tools import Tool
from langchain_experimental.tools.python.tool import PythonREPLTool
import numpy as np
from langchain.prompts import  PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import StrOutputParser


def get_tools(model, X_train, feature_names, class_names):
    # 初始化 LIME 解释器
    explainer = lime.lime_tabular.LimeTabularExplainer(
        X_train,
        feature_names=feature_names,
        class_names=class_names,
        mode='classification'  # 或 'regression', 取决于您的模型类型
    )

    # LIME 解释生成工具
    def local_exp(input):
        input = eval(input)
        exp = explainer.explain_instance(np.array(input), model.predict_proba, num_features=len(feature_names))
        explanation = exp.as_list()
        ans = "The feature contributions for the prediction are as follows:\n"
        for feature, effect in explanation:
            ans += f"{feature} : {effect:.4f}\n"
        return ans

    explanation_tool = Tool(
        name='LIME_Explanation',
        func=local_exp,
        description="Use this tool to get the contribution of each feature to the prediction outcome of a sample."
    )

    # 预测工具
    def predict_sample(input, model):
        try:
            # 安全地评估输入数据
            sample = eval(input)  # 将输入字符串转换为实际的 Python 对象
            if isinstance(sample, list):
                sample = np.array([sample])  # 转换为 NumPy 数组以符合模型的输入要求
            elif isinstance(sample, dict):
                # 如果输入是字典形式，假设字典按特征名称排列
                sample = np.array([[sample[name] for name in feature_names]])
            # 进行预测
            prediction = model.predict(sample)
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(sample)
                proba_str = ", ".join(f"{class_names[i]}: {prob:.2%}" for i, prob in enumerate(probabilities[0]))
                return f"Prediction: {class_names[prediction[0]]}, Probabilities: {proba_str}"
            else:
                return f"Prediction: {prediction[0]}"
        except Exception as e:
            return f"Error: {str(e)}"

    prediction_tool = Tool(
        name='Prediction',
        func=lambda input: predict_sample(input, model),  # 使用 Lambda 将模型传递给函数
        description="Predict the outcome for a given sample. Input should be a list of feature values or a feature dictionary."
    )

    # Python 执行工具，直接回答工具等
    python_tool = PythonREPLTool()

    python_tool_desc = (
        "Use this tool when you need to execute python commands to obtain data or plot charts. Input should be a valid python command."
        " If you want to see the output of a value, you should print it out with `print(...)`.You can keep reusing this tool until you get the final answer."
        "For chart plotting, the altair library is mandatory. Instead of saving your chart, display it using the `chart.display()` function rather than `chart.show()`."
        "You can also add some required interactivity to the chart. When you update or modify the chart, you must make modifications on the original chart,"
        " which means that the existing parts of the original chart cannot be changed."
    )

    python_tool.description = python_tool_desc

    # 直接回答的工具
    def final(input):
        return input

    Final_answer = Tool(
        name='Final_answer',
        func=final,
        description="Use this if you want to respond directly to the human. Input should be what you want to respond and it will return the same. After using this tool, you must leave the results returned by this tool intact as Final Answer instead of continuing with other actions."
    )

    # 任务分解的工具
    llm_gpt4 = ChatOpenAI(model_name="gpt-4-turbo", temperature=0)
    prompt = PromptTemplate.from_template(
        """
        You are a researcher working with an interpretable deep model, and your goal is to break down abstract tasks into specific work steps and processes. The task involves performing various operations related to training and predicting samples. The model has been trained. Here are the specific operations you can perform:

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

    # 打包所有工具
    tools = [
        explanation_tool,
        prediction_tool,
        Task_decom_tool,
        python_tool,
        Final_answer
    ]
    return tools
