#构建一个agent用以对上传的数据文件进行分析，并生成对应的数据描述
from langchain_core.output_parsers import StrOutputParser
from langchain.agents import ZeroShotAgent
from langchain.chains.llm import LLMChain
def get_data_description(llm, df):

    features = df.columns.tolist()
    n_samples, n_features = df.shape

    # 构造描述请求
    prompt = f"The dataset contains {n_samples} samples and {n_features} features, which are: {', '.join(features)}. "
    prompt += """Please provide a detailed description of this dataset, including detailed description of each features and the prediction goals(usually shown in the last column of the file.Besides, you shoule give suggestions for possible machine learning tasks such as classification or regression, according to whether the prediction goal is discrete.
    Here is an exemplified output:
    Feature Descriptions:
        age - The age of the passenger.
        workclass: The type of work this person does.
        fnlwgt: The serial number in Census database.
        education: The educational level of the person.
        education-num: The education time of the person.
        marital-status: Marital status of the person.
        occupation: The person's occupation.
        relationship: The social role this person occupies.
        race:The race of the person, black and white;
        sex: The person's gender.
        capital.gain: The person’s investment income.
        capital.loss: The person's investment losses.
        hours.per.week: The person's weekly working hours.
        native.country:The person's nationality."""

    # 获取描述
    ai_data_description = llm.invoke(prompt)
    output_parser = StrOutputParser()
    data_description = output_parser.invoke(ai_data_description)
    return data_description


