from langchain.agents import Tool
from langchain_experimental.tools.python.tool import PythonREPLTool
import numpy as np
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import StrOutputParser
import streamlit as st
import streamlit.components.v1 as components

def get_tools(ebm):
    #样本预测的工具
    forecast_desc = (
        "use this tool when you need to predict the probabilities given a series of feature values in a Generalized Additive Model."
        "It will return the prediction result and probability. Your final answer must begin with the results returned and your thoughts about it,"
        " plus a conclusion based on the combination of the results returned and the description of the data set. To use the tool you must input "
        "a list consisting of the value of each feature. The value must be provided in question. For example, the question is: if a person's  Name is Jobs"
        ", height is 175cm, weight is 65kg, what is the prediction? you need to provide this tool with [' Jobs', 175, 65]. Remember to put quotes around the strings in the list."
        "And there must be a space before the strings within the quotation marks. Also the order of elements should be consistent with the order of features in the data you see. "
    )

    def forecast(input):
        sample = eval(input)
        result = ebm.predict(sample)
        probability = ebm.predict_proba(sample)
        if isinstance(sample[0], list):
            ans = "The prediction results for each row from top to bottom are as follows:\n"
            for i, s in enumerate(sample) :
                probe = np.array(probability[i], dtype=float)
                probe = max(probe[0],probe[1])
                ans += f"    The prediction result of row {i} in the your input is {result[i]}, and the probability that the result comes to fruition is {probe}\n"
            return ans
        else:
            probability = np.array(probability, dtype=float)
            probability = max(probability[0][0],probability[0][1])
            return f"The prediction result is {result[0]}，and the probability that the result comes to fruition is {probability}"

    forecast_tool = Tool(
        name='Forecast',
        func=forecast,
        description=forecast_desc
    )

    #执行python命令的工具
    python_tool = PythonREPLTool()

    python_tool_desc = (
        "Use this tool when you need to execute python commands to obtain data or plot charts. "
        "Input should be a valid python command."
        " If you want to see the output of a value, you should print it out with `print(...)`."
        "You can keep reusing this tool until you get the final answer."
        "For chart plotting, the altair library is mandatory. "
        "Instead of displaying your chart directly, save it using the `chart.save('charts/altair_chart.html')` "
        "After you create a chart, you must call the chart_display_tools to display the chart."
        "You can also add some required interactivity to the chart. "
        "When you update or modify the chart, you must make modifications on the original chart,"
        " which means that the existing parts of the original chart cannot be changed."
    )

    python_tool.description = python_tool_desc

    # 展示图表的工具
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
            components.html(chart_html, height=500, width=800, scrolling=True)

        except FileNotFoundError:
            st.error(f"The chart HTML file could not be found at {html_path}. Please ensure the path is correct.")
        except Exception as e:
            st.error(f"An error occurred while displaying the chart: {e}")

    chart_display_tool = Tool(
        name='chart_display_tool',
        func=chart_display,
        description="Use this tool after you calling python_tool to create a chart."
    )

    #得到local explain的图工具
    desc = (
        "use this tool When you need to get the contribution of each feature to the prediction outcome of a sample. You must input one or more lists consisting of the value of each feature."
        "Be sure that the order of elements is consistent with the order of features in the data you see. It will return the score of each feature. A positive score indicates a positive contribution to the prediction result"
        ", otherwise, it is a negative contribution. The larger the absolute value of the score, the greater the degree of contribution."
    )

    def local_exp(input):
        sample = eval(input)
        explanation = ebm.explain_local(sample)
        if isinstance(sample[0], np.ndarray) or isinstance(sample[0], list):
            ans = f"The feature contribution to prediction results of features in each sample from top to bottom are as follows(The order of features corresponding to these scores is:{explanation.data(0)['names']}):\n"
            for i, s in enumerate(sample) :
                scores = [f'{x:.4f}' for x in explanation.data(i)['scores']]
                ans += f"     the contribution scores of row {i} in your input is {scores}\n"
            return ans
        else:
            explanation = ebm.explain_local(sample)
            scores = [f'{x:.4f}' for x in explanation.data(0)['scores']]
            return f"the contribution scores is {scores}, and the order of features corresponding to these scores is:{explanation.data(0)['names']}"

    explanation_tool = Tool(
        name='Explanation_tool',
        func=local_exp,
        description=desc
    )

    #直接回答的工具
    def final(input):
        return input

    Final_answer = Tool(
        name='Final_answer',
        func=final,
        description="Use this if you want to respond directly to the human. Input should be what you want to respond and it will return the same. After using this tool, you must leave the results returned by this tool intact as Final Answer instead of continuing with other actions."
    )

    #任务分解的工具
    llm_gpt4 = ChatOpenAI(model_name= "gpt-4",temperature=0)
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
        return runnable.invoke({"query":query})

    decomposition_desc = "This tool must be used first when you receive a task at the beginning and attempt to complete it. You'll get step-by-step instructions for completing the task. You can refer to this guidance without blindly following it."

    Task_decom_tool = Tool(
        name='Task_decomposition',
        func=decomposition,
        description=decomposition_desc
    )

    #获得所有的工具的列表
    tools=[]
    tools.append(python_tool)
    tools.append(forecast_tool)
    tools.append(Final_answer)
    tools.append(explanation_tool)
    tools.append(Task_decom_tool)
    tools.append(chart_display_tool)

    return tools