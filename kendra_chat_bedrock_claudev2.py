# from aws_langchain.kendra import AmazonKendraRetriever #custom library
from langchain.retrievers import AmazonKendraRetriever
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.llms.bedrock import Bedrock
from langchain.chains.llm import LLMChain
# from langchain_openai import ChatOpenAI
import sys
import os
import boto3

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

MAX_HISTORY_LENGTH = 5

# AWS User Credentials
AWS_ACCESS_KEY_ID=os.environ["AWS_ACCESS_KEY_ID"]
AWS_SECRET_ACCESS_KEY=os.environ["AWS_SECRET_ACCESS_KEY"]
AWS_REGION="ap-southeast-1"

def build_chain():
  AWS_ACCESS_KEY_ID=os.environ["AWS_ACCESS_KEY_ID"]
  AWS_SECRET_ACCESS_KEY=os.environ["AWS_SECRET_ACCESS_KEY"]
  GT_API_KEY=os.environ["GT_API_KEY"]
  region = AWS_REGION
  kendra_index_id = os.environ["KENDRA_INDEX_ID"]
  boto3_bedrock = boto3.client('bedrock-runtime', aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY, region_name='us-east-1')
  boto3_kendra = boto3.client('kendra', aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY, region_name=region)

  llm = Bedrock(
      client=boto3_bedrock,
      region_name = 'us-east-1',
      model_kwargs={
          "max_tokens_to_sample":1000,
          "temperature":1,
          "top_k":250,"top_p":0.999,
          "anthropic_version":"bedrock-2023-05-31"
      },
      model_id="anthropic.claude-v2:1"
  )

  # chat = ChatOpenAI(
  #   openai_api_base="https://litellm.launchpad.tech.gov.sg",
  #   model = "claude-3-haiku",
  #   temperature=0.1,
  #   extra_body={
  #       "metadata": {
  #           "generation_name": "ishaan-generation-langchain-client",
  #           "generation_id": "langchain-client-gen-id22",
  #           "trace_id": "langchain-client-trace-id22",
  #           "trace_user_id": "langchain-client-user-id2"
  #       }
  #   }
  # )
      
  retriever = AmazonKendraRetriever(index_id=kendra_index_id,top_k=5,region_name=region,client=boto3_kendra)


  prompt_template = """
  Human: Your role is to assign a data security classification to fields. It comes in two parts: the security level which measures impact to government or national interests, and the sensitivity level which measures impact to individual or commercial interests. Combined together, they form a classification level, where "Restricted / Sensitive Normal" means the security level is Restricted and the sensitivity level is Sensitive Normal. The valid classification levels are shown below (in increasing order)
  Security: Official (Open), Official (Closed), Restricted, Confidential (Cloud Eligible), Confidential, Secret, Top Secret
  Sensitivity: Non-Sensitive, Sensitive Normal, Sensitive High
  Furthermore, if the sensitivity level is Sensitive Normal or Sensitive High, the security level can never be Official (Open)

  If the classification level is obvious or can be inferred from your domain knowledge, propose a classification level. Otherwise, prompt the user further to provide information or context which may be useful to propose the classification level, asking the user specific easy to answer questions. Assume that the user has complete information on business context and nature of fields, but no idea what a data security classification is or what information may be relevant. 

  In particular, the following factors may affect the classification level: 
  1) Account level data may have higher classification than aggregated data
  2) Deidentified data may have lower classification
  3) Frequency of data, as well as how real-time the data is, may affect classification level

  Assistant: OK, got it, I'll be a truthful AI assistant.

  Human: Here are a few documents in <documents> tags:
  <documents>
  {context}
  </documents>
  
  Based on the above documents, depending on the nature of the query below, classify the security and sensitivity level if a dataset is provided (asking follow up queries if needed), or act on the user input if a query or clarification is provided. 
  {question} 

  Assistant:
  """
  prompt = PromptTemplate(
      template=prompt_template, input_variables=["context", "question"]
  )

  condense_qa_template = """
  {chat_history}
  Human:
  Given the previous conversation and a follow up question below, rephrase the follow up question
  to be a standalone question.

  Follow Up Question: {question}
  Standalone Question:

  Assistant:"""
    
  standalone_question_prompt = PromptTemplate.from_template(condense_qa_template)


  
  qa = ConversationalRetrievalChain.from_llm(
        llm=llm, 
        retriever=retriever, 
        condense_question_prompt=standalone_question_prompt, 
        return_source_documents=True, 
        combine_docs_chain_kwargs={"prompt":prompt},
        verbose=True)

  # qa = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, qa_prompt=prompt, return_source_documents=True)
  return qa


def run_chain(chain, prompt: str, history=[]):
  return chain({"question": prompt, "chat_history": history})


if __name__ == "__main__":
  chat_history = []
  qa = build_chain()
  print(bcolors.OKBLUE + "Hello! How can I help you?" + bcolors.ENDC)
  print(bcolors.OKCYAN + "Ask a question, start a New search: or CTRL-D to exit." + bcolors.ENDC)
  print(">", end=" ", flush=True)
  for query in sys.stdin:
    if (query.strip().lower().startswith("new search:")):
      query = query.strip().lower().replace("new search:","")
      chat_history = []
    elif (len(chat_history) == MAX_HISTORY_LENGTH):
      chat_history.pop(0)
    result = run_chain(qa, query, chat_history)
    chat_history.append((query, result["answer"]))
    print(bcolors.OKGREEN + result['answer'] + bcolors.ENDC)
    if 'source_documents' in result:
      print(bcolors.OKGREEN + 'Sources:')
      for d in result['source_documents']:
        print(d.metadata['source'])
    print(bcolors.ENDC)
    print(bcolors.OKCYAN + "Ask a question, start a New search: or CTRL-D to exit." + bcolors.ENDC)
    print(">", end=" ", flush=True)
  print(bcolors.OKBLUE + "Bye" + bcolors.ENDC)
