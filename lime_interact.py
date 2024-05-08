from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_community.chat_models import ChatOpenAI
import streamlit as st
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import lime_agent

def train_random_forest(df):
    # 准备数据
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    seed = 42
    np.random.seed(seed)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    # 训练模型
    model = RandomForestClassifier(n_estimators=100, random_state=seed)
    model.fit(X_train, y_train)
    return model, X_train, X_test

def lime_handle_chat_interaction(df, model, X_train, feature_names, class_names):
    if 'messages' not in st.session_state:
        st.session_state['messages'] = [{"role": "assistant", "content": "How can I help you?"}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input("请输入要解释的数据行索引（例如：5）:"):
        idx = 0
        if prompt.isdigit() and 0 <= int(prompt) < len(df):
            idx = int(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        if st.session_state['api_key']:
            response = process_prompt_with_agent_lime(prompt, df, model, X_train, feature_names, class_names, idx)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.chat_message("assistant").write(response)

def process_prompt_with_agent_lime(prompt, df, model, X_train, feature_names, class_names, index):
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo", openai_api_key=st.session_state['api_key'], streaming=True)
    eag = lime_agent.get_agent(llm, model, X_train, feature_names, class_names, df=df, index=index)
    st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
    return eag.run(prompt, callbacks=[st_cb])