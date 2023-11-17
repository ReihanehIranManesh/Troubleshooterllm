import openai

OPENAI_API_KEY = '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'

openai.api_key = OPENAI_API_KEY
system_template = """Use the following pieces of context to answer the users question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
{context}

Begin!
----------------
Question: {question}
Helpful Answer:"""

from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}")
]

prompt = ChatPromptTemplate.from_messages(messages)

from langchain.document_loaders import PyPDFLoader

loader = PyPDFLoader("ServiceHandbookSmall.pdf")
data = loader.load()

from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(data)

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

embedding = OpenAIEmbeddings(openai_api_key=openai.api_key)
vectorstore = Chroma.from_documents(documents=docs, embedding=embedding)

from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613", openai_api_key=openai.api_key)

while (True):
    question = input("Enter your Question: ")

    from langchain.chains import RetrievalQA

    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": prompt})

    response = qa_chain({"query": question})
    answer = response["result"]

    print(response)
    MODEL = "gpt-4-1106-preview"

    from openai import OpenAI

    client = OpenAI(
        # defaults to os.environ.get("OPENAI_API_KEY")
        api_key=OPENAI_API_KEY,
    )

    message = "Based on the given answer, search the web and youtube for related videos and show them to the user. Text:" + answer

    stream = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": message,

            }
        ],
        model=MODEL,
        stream=True
    )

    for part in stream:
        answer += part.choices[0].delta.content or ""

    print(answer)
    break
