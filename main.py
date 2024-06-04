from langchain.agents import AgentType
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI
import streamlit as st
import pandas as pd
import os


def clear_submit():
    """
    Clear the Submit Button State
    Returns:
    """
    st.session_state["submit"] = False


@st.cache_data(ttl="2h")
def load_data():
    file_path = "data/out.csv"
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        st.error(f"File not found: {file_path}")
        return None


st.set_page_config(page_title="LangChain: Chat with pandas DataFrame", page_icon="🦜")
st.title("🦜 LangChain OpenAI: Chat with the NFT Dataset")
st.subheader(
    "This app is a WIP prototype and there may be bugs.  Try asking questions about the NFT dataset.  Last Updated: June 2024"
)

df = load_data()

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("OpenAI API Key not found in environment variables.")
    st.stop()

correct_password = os.getenv("PASSWORD")
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    password = st.sidebar.text_input("Password", type="password")
    if st.sidebar.button("Submit"):
        if password == correct_password:
            st.session_state["authenticated"] = True
        else:
            st.sidebar.error("Incorrect password. Please try again.")
    st.stop()

if "messages" not in st.session_state or st.sidebar.button(
    "Clear conversation history"
):
    st.session_state["messages"] = [
        {"role": "assistant", "content": "How can I help you?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input(placeholder="What is this data about?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

    llm = ChatOpenAI(
        temperature=0,
        model="gpt-3.5-turbo-0613",
        openai_api_key=openai_api_key,
        streaming=True,
    )

    pandas_df_agent = create_pandas_dataframe_agent(
        llm,
        df,
        verbose=True,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        handle_parsing_errors=True,
    )

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = pandas_df_agent.run(st.session_state.messages, callbacks=[st_cb])
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)
