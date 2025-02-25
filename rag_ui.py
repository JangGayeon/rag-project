import gradio as gr
from pdf_processor import process_pdf
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.llms import Ollama

# ChromaDB 저장 경로
CHROMA_DB_PATH = "C:/rag-project/chroma_db"

# Ollama 임베딩 & 모델 설정
embeddings = OllamaEmbeddings(model="mxbai-embed-large")
vectorstore = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)
retriever = vectorstore.as_retriever()

qa_chain = RetrievalQA.from_chain_type(
    llm=Ollama(model="llama3"),
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
    
    retrieved_docs = retriever.get_relevant_documents(question)
    
    if not retrieved_docs:
        return "❌ 관련된 문서를 찾을 수 없습니다."
    
    answer = qa_chain.run(question)
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
