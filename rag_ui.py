import gradio as gr
from pdf_processor import process_pdf
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama

# ChromaDB 저장 경로
CHROMA_DB_PATH = "C:/rag-project/chroma_db"

# Ollama 임베딩 & 모델 설정
embeddings = OllamaEmbeddings(model="mxbai-embed-large")
vectorstore = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)
retriever = vectorstore.as_retriever()

# 프롬프트 템플릿 추가
def create_prompt(question, relevant_docs):
    # 문서 내용들을 하나로 합쳐서 context로 사용
    context = " ".join([doc.page_content for doc in relevant_docs])
    
    # 프롬프트를 한글로 응답하도록 강력히 유도
    return f"""
    아래의 문서 내용에 대한 질문을 한글로 정확하고 자연스럽게 답변해 주세요:
    
    문서 내용: {context}
    
    질문: {question}
    
    답변은 반드시 한글로 작성해야 하며, 제공된 문서 내용을 기반으로 구체적이고 정확한 답을 작성해주세요.
    """

qa_chain = RetrievalQA.from_chain_type(
    llm=Ollama(model="gemma2"),  # 여기서 모델을 gemma2-9b로 변경
    chain_type="stuff",
    retriever=retriever
)

# 📂 **PDF 업로드 및 분석 함수**
def handle_upload(pdf_file):
    if pdf_file is None:
        return "❌ PDF 파일을 업로드해주세요."
    
    process_pdf(pdf_file.name)  # PDF 처리 실행
    return "✅ PDF 분석이 완료되었습니다!"

# 📝 **Q&A 시스템 (사용자 질문에 대한 답변)**
def ask_question(question):
    if not question.strip():
        return "❌ 질문을 입력해주세요!"
    
    # 문서에서 관련된 내용을 검색
    retrieved_docs = retriever.get_relevant_documents(question)
    
    if not retrieved_docs:
        return "❌ 관련된 문서를 찾을 수 없습니다."
    
    # 한국어로 답변을 요청하는 프롬프트 생성
    prompt = create_prompt(question, retrieved_docs)
    
    # `qa_chain` 실행, 프롬프트 템플릿을 사용하여 한국어 답변 유도
    answer = qa_chain.run(prompt)
    return f"📝 답변: {answer}"

# 🎨 **Gradio UI 디자인**
with gr.Blocks() as demo:
    gr.Markdown("📄 **PDF 기반 LLaMA3 RAG Q&A 시스템**")

    with gr.Row():
        with gr.Column():
            gr.Markdown("📂 **PDF 파일 업로드**")
            upload_button = gr.File(label="Upload PDF")
            analyze_button = gr.Button("📂 PDF 분석")
            output_text = gr.Textbox(label="처리 상태", interactive=False)
            
            analyze_button.click(fn=handle_upload, inputs=[upload_button], outputs=[output_text])

        with gr.Column():
            gr.Markdown("🤖 **질문 입력 & 답변 출력**")
            question_input = gr.Textbox(label="Enter your question")
            submit_button = gr.Button("Submit")
            answer_output = gr.Textbox(label="Answer", interactive=False)
            
            submit_button.click(fn=ask_question, inputs=[question_input], outputs=[answer_output])

demo.launch()
