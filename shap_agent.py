import shap
import numpy as np
from langchain.agents import ZeroShotAgent
from langchain.agents import  AgentExecutor
from langchain.chains.llm import LLMChain
from langchain.memory import ConversationBufferMemory
from shap_tool import get_tools
from prompt import suffix_no_df, suffix_with_df, shap_get_prefix

def shap_get_agent(llm, model, df=None, dataset_description=None, y_axis_description=None):
    # 初始化 SHAP 解释器
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df)  # 这里假设 df 已经是训练数据或者相应的特征数据

    # 使用 SHAP 计算全局特征重要性
    global_importance = np.abs(shap_values).mean(axis=0)
    feature_importance = shap_feature_importances_to_text(global_importance, df.columns)

    # 获取 prompt 的 prefix 部分
    prefix = shap_get_prefix(feature_importance, dataset_description, y_axis_description)

    # 获取工具
    tools = get_tools(model, df)
    python_tool = tools[0]

    # 获取 prompt 的 suffix 部分
    if df is not None:
        python_tool.python_repl.locals = {"df": df, "shap_values": shap_values, "explainer": explainer}
        input_variables = ["input", "chat_history", "agent_scratchpad", "df_head"]
        suffix = suffix_with_df
    else:
        python_tool.python_repl.locals = {"shap_values": shap_values, "explainer": explainer}
        input_variables = ["input", "chat_history", "agent_scratchpad"]
        suffix = suffix_no_df

    prompt = ZeroShotAgent.create_prompt(
        tools,
        prefix=prefix,
        suffix=suffix,
        input_variables=input_variables,
    )

    memory = ConversationBufferMemory(memory_key="chat_history")

    if 'df_head' in input_variables:
        prompt = prompt.partial(df_head=str(df.head().to_markdown()))

    llm_chain = LLMChain(llm=llm, prompt=prompt)
    agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
    return AgentExecutor.from_agent_and_tools(
        agent=agent, tools=tools, verbose=True, memory=memory, handle_parsing_errors=True
    )


def shap_feature_importances_to_text(importances, feature_names):
    # 将特征重要性格式化为文本
    feature_importance_text = ""
    for importance, name in sorted(zip(importances, feature_names), reverse=True):
        feature_importance_text += f"{name}: {importance:.2f}\n"
    return feature_importance_text
