from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.llms import Ollama

# ChromaDB 저장 경로
CHROMA_DB_PATH = "C:/rag-project/chroma_db"

# 벡터 DB 로드
embeddings = OllamaEmbeddings(model="mxbai-embed-large")
retriever = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings).as_retriever()
qa_chain = RetrievalQA.from_chain_type(
    llm=Ollama(model="llama3"),
    chain_type="stuff",
    retriever=retriever
)

def answer_question(question):
    """사용자의 질문을 받아서 답변을 반환하는 함수"""
    if not question.strip():
        return "❌ 질문을 입력해주세요."

    response = qa_chain.invoke({"query": question})
    return response["result"]

# CLI에서 실행 가능하도록 설정
if __name__ == "__main__":
    while True:
        query = input("\n🟢 질문을 입력하세요 (종료하려면 'exit' 입력): ")
        if query.lower() == "exit":
            print("🔴 프로그램을 종료합니다.")
            break
        print("📝 답변:", answer_question(query))
