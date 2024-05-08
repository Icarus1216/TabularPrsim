import lime
import lime.lime_tabular
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from langchain.agents import ZeroShotAgent
from langchain.agents import  AgentExecutor
from langchain.chains.llm import LLMChain
from langchain.memory import ConversationBufferMemory
from lime_tools import get_tools
from prompt import suffix_no_df, suffix_with_df, lime_get_prefix

def feature_importances_to_text(lime_explanation):
    feature_importances = ""
    # LIME 的 explain_instance 方法返回的解释是一个列表，其中每个元素都是（特征名称, 特征对预测的影响）的形式
    for feature, importance in lime_explanation:
        feature_importances += f"{feature}: {importance:.2f}\n"
    return feature_importances


def get_agent(llm, explainer, model, dataset, sample_index, df=None, dataset_description=None, y_axis_description=None):
    try:
        # 获取 LIME 解释
        exp = explainer.explain_instance(data_row=dataset.X_test[sample_index],
                                         predict_fn=model.predict_proba if dataset.is_classification else model.predict)
        local_explanation = exp.as_list()
        feature_importance = feature_importances_to_text(local_explanation)

        # 获取 prompt 的 prefix 部分
        prefix = lime_get_prefix(feature_importance, dataset_description, y_axis_description)

        # 获取工具
        tools = get_tools(model)
        python_tool = tools[0]

        # 获取 prompt 的 suffix 部分
        if df is not None:
            python_tool.python_repl.locals = {"df": df, "ft_graph": local_explanation}
            input_variables = ["input", "chat_history", "agent_scratchpad", "df_head"]
            suffix = suffix_with_df
        else:
            python_tool.python_repl.locals = {"ft_graph": local_explanation}
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
    except Exception as e:
        print(f"An error occurred: {e}")
        return None



def build_prompt(explanation):
    if not explanation:
        return "No explanation available or an error occurred."
    text = "Model explanation:\n"
    for feature, effect in explanation:
        text += f"{feature} : {effect}\n"
    return text
