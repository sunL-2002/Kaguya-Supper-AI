# coding=utf-8
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_community.embeddings import ZhipuAIEmbeddings
import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.chat_models import ChatZhipuAI

load_dotenv()

documents = [
    Document(
        page_content="Dogs are great companions, known for their loyalty and friendliness.",
        metadata={"source": "mammal-pets-doc"},
    ),
    Document(
        page_content="Cats are independent pets that often enjoy their own space.",
        metadata={"source": "mammal-pets-doc"},
    ),
    Document(
        page_content="Goldfish are popular pets for beginners, requiring relatively simple care.",
        metadata={"source": "fish-pets-doc"},
    ),
    Document(
        page_content="Parrots are intelligent birds capable of mimicking human speech.",
        metadata={"source": "bird-pets-doc"},
    ),
    Document(
        page_content="Rabbits are social animals that need plenty of space to hop around.",
        metadata={"source": "mammal-pets-doc"},
    ),
]

# 使用智谱AI的embeddings
embeddings = ZhipuAIEmbeddings(
    model="embedding-2",
    api_key=os.getenv("ZHIPUAI_API_KEY")
)

vectorstore = Chroma.from_documents(
    documents,
    embedding=embeddings,
)

# 返回文档
# print(vectorstore.similarity_search("cat"))
# 返回分数
# print(vectorstore.similarity_search_with_score("cat"))

# 根据与嵌入查询的相似性返回文档
# embedding = embeddings.embed_query("cat")
# vectorstore.similarity_search_by_vector(embedding)

# 检索器
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 1},
)
# retriever.batch(["cat", "shark"])

# 简易检索增强生产(RAG)
llm = ChatZhipuAI(
    model_name="GLM-4.5-Flash",
    temperature=0.8,
    api_key=os.getenv("ZHIPUAI_API_KEY")
)

message = """
Answer this question using the provided context only.

{question}

Context:
{context}
"""

prompt = ChatPromptTemplate.from_messages([("human", message)])

rag_chain = {"context": retriever, "question": RunnablePassthrough()} | prompt | llm

response = rag_chain.invoke("tell me about cats")

print(response.content)
# 输出文档中的 Cats are independent pets that often enjoy their own space.
