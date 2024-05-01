import streamlit as st
import uuid
import sys

import kendra_chat_bedrock_claudev2 as bedrock_claudev2

USER_ICON = "images/user-icon.png"
ICON = "images/dataset.png"
AI_ICON = "images/dataset.png"
MAX_HISTORY_LENGTH = 5
PROVIDER_MAP = {
    "bedrock_claude": "Bedrock Claude",
    "bedrock_claudev2": "Bedrock Claude V2",
    'llama2' : 'Llama 2',
    'falcon40b': 'Falcon 40B'
}

st.set_page_config(
    page_title = "Classify Right Chatbot",
    page_icon = "💻"
)

# Check user type
if 'user_type' not in st.session_state:
    st.session_state['user_type'] = ""

#function to read a properties file and create environment variables
def read_properties_file(filename):
    import os
    import re
    with open(filename, 'r') as f:
        for line in f:
            m = re.match(r'^\s*(\w+)\s*=\s*(.*)\s*$', line)
            if m:
                os.environ[m.group(1)] = m.group(2)


# Check if the user ID is already stored in the session state
if 'user_id' in st.session_state:
    user_id = st.session_state['user_id']

# If the user ID is not yet stored in the session state, generate a random UUID
else:
    user_id = str(uuid.uuid4())
    st.session_state['user_id'] = user_id

st.session_state['llm_app1'] = bedrock_claudev2
st.session_state['llm_chain1'] = bedrock_claudev2.build_chain()

# if 'llm_chain1' not in st.session_state:
#     if (len(sys.argv) > 1):
#         if (sys.argv[1] == 'bedrock_claudev2'):
#             st.session_state['llm_app1'] = bedrock_claudev2
#             st.session_state['llm_chain1'] = bedrock_claudev2.build_chain()
#         elif (sys.argv[1] == 'llama2'):
#             st.session_state['llm_app1'] = llama2
#             st.session_state['llm_chain1'] = llama2.build_chain()
#         elif (sys.argv[1] == 'falcon'):
#             st.session_state['llm_app1'] = falcon40b
#             st.session_state['llm_chain1'] = falcon40b.build_chain()
#         else:
#             raise Exception("Unsupported LLM: ", sys.argv[1])
#     else:
#         raise Exception("Usage: streamlit run app.py <bedrock_claude|bedrock_claudev2|llama2|falcon40b>")

if 'chat_history1' not in st.session_state:
    st.session_state['chat_history1'] = []
    
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

if "chats1" not in st.session_state:
    st.session_state.chats1 = [
        {
            'id': 0,
            'question': '',
            'answer': ''
        }
    ]

if "questions1" not in st.session_state:
    st.session_state.questions1 = []

if "answers1" not in st.session_state:
    st.session_state.answers1 = []

if "input1" not in st.session_state:
    st.session_state.input1 = ""

# CN - 21 Jan 2024
st.session_state['llm_app'] = st.session_state['llm_app1']
st.session_state['llm_chain'] = st.session_state['llm_chain1']
st.session_state['chat_history'] = st.session_state['chat_history1']
st.session_state.chats = st.session_state.chats1        
st.session_state.questions = st.session_state.questions1
st.session_state.answers = st.session_state.answers1
st.session_state.input = st.session_state.input1

# chat_provider = sys.argv[1]
chat_provider = "bedrock_claudev2"

st.markdown("""
        <style>
               .block-container {
                    padding-top: 32px;
                    padding-bottom: 32px;
                    padding-left: 0;
                    padding-right: 0;
                }
                .element-container img {
                    background-color: #000000;
                }

                .main-header {
                    font-size: 25px;
                }
        </style>
        """, unsafe_allow_html=True)

# def write_logo():
#     col1, col2, col3 = st.columns([5, 1, 5])
#     with col2:
#         st.image(AI_ICON, use_column_width='always') 


def write_top_bar():
    col1, col2, col3 = st.columns([3,8,2])
    with col1:
        st.image(ICON,use_column_width=False,width=150)
    with col2:
        # selected_provider = sys.argv[1]
        selected_provider = "bedrock_claudev2"

        if selected_provider in PROVIDER_MAP:
            provider = PROVIDER_MAP[selected_provider]
        else:
            provider = selected_provider.capitalize()
        header = f"Classify Right Chatbot"
        powered_by = f"Powered by Amazon Kendra and {provider}"
        st.title(header)
        st.caption(powered_by)
    with col3:
        clear = st.button("Clear Chat")
    return clear

if st.session_state.user_type == "":
    st.write("")
else:
    clear = write_top_bar()

    if clear:
        st.session_state.questions = []
        st.session_state.answers = []
        st.session_state.input = ""
        st.session_state["chat_history"] = []
        st.session_state.questions1 = []
        st.session_state.answers1 = []
        st.session_state.input1 = ""
        st.session_state["chat_history1"] = []

    
def handle_input():
    input = st.session_state.input
    question_with_id = {
        'question': input,
        'id': len(st.session_state.questions)
    }
    st.session_state.questions.append(question_with_id)
    
    # Managing Chat History
    chat_history = st.session_state["chat_history"]
    if len(chat_history) == MAX_HISTORY_LENGTH:
        chat_history = chat_history[:-1]

    llm_chain = st.session_state['llm_chain']
    chain = st.session_state['llm_app']
    result = chain.run_chain(llm_chain, input, chat_history)
    answer = result['answer']
    chat_history.append((input, answer))
    
    document_list = []
    if 'source_documents' in result:
        for d in result['source_documents']:
            if not (d.metadata['source'] in document_list):
                document_list.append((d.metadata['source']))

    st.session_state.answers.append({
        'answer': result,
        'sources': document_list,
        'id': len(st.session_state.questions)
    })
    st.session_state.input = ""
    # CN - Added 21 Jan 2024
    st.session_state['llm_app1'] = st.session_state['llm_app']
    st.session_state['llm_chain1'] = st.session_state['llm_chain']
    st.session_state['chat_history1'] = st.session_state['chat_history']
    st.session_state.chats1 = st.session_state.chats        
    st.session_state.questions1 = st.session_state.questions
    st.session_state.answers1 = st.session_state.answers
    st.session_state.input1 = st.session_state.input    
    

def write_user_message(md):
    col1, col2 = st.columns([1,12])
    
    with col1:
        st.image(USER_ICON, use_column_width='always')
    with col2:
        st.warning(md['question'])


def render_result(result):
    answer, sources = st.tabs(['Answer', 'Sources'])
    with answer:
        render_answer(result['answer'])
    with sources:
        if 'source_documents' in result:
            render_sources(result['source_documents'])
        else:
            render_sources([])

def render_answer(answer):
    col1, col2 = st.columns([1,12])
    with col1:
        st.image(AI_ICON, use_column_width='always')
    with col2:
        st.info(answer['answer'])

def render_sources(sources):
    col1, col2 = st.columns([1,12])
    with col2:
        with st.expander("Sources"):
            for s in sources:
                st.write(s)

    
#Each answer will have context of the question asked in order to associate the provided feedback with the respective question
def write_chat_message(md, q):
    chat = st.container()
    with chat:
        render_answer(md['answer'])
        render_sources(md['sources'])
    

with st.container():
    for (q, a) in zip(st.session_state.questions, st.session_state.answers):
        write_user_message(q)
        write_chat_message(a, q)


st.markdown("")
if st.session_state.user_type == "":
    st.error("Please proceed to login...")
else:
    input = st.text_input("You are talking to Classify Right AI. Input your question below...", key="input", on_change=handle_input)
    st.markdown("---")
    with st.expander("Session State"):
        st.write(st.session_state)