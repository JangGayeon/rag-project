import os
import gradio as gr
from pdf_processor import process_pdf
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama

# 파일 저장 경로
UPLOAD_DIR = 'C:/rag-project/pdf-files/'
CHROMA_DB_PATH = "C:/rag-project/chroma_db"

# Ollama 임베딩 & 모델 설정
embeddings = OllamaEmbeddings(model="mxbai-embed-large")  # 임베딩 모델
vectorstore = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)
retriever = vectorstore.as_retriever()  # 벡터 DB에서 검색을 위한 retriever

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
    llm=Ollama(model="gemma2"),  # gemma2-9b 모델 사용
    chain_type="stuff",
    retriever=retriever
)

# 📂 **PDF 업로드 및 분석 함수**
def handle_upload(file, file_type):
    if file is None:
        return "❌ 파일을 업로드해주세요."
    
    if file_type == 'pdf':
        process_pdf(file.name)  # PDF 처리 실행
    elif file_type == 'csv':
        process_csv(file.name)
    elif file_type == 'json':
        process_json(file.name)
    
    # 업로드된 PDF 목록 업데이트
    uploaded_pdfs = os.listdir(UPLOAD_DIR)
    return uploaded_pdfs

# CSV 파일 처리 함수
def process_csv(file_path):
    import pandas as pd
    df = pd.read_csv(file_path)
    return df.to_dict(orient="records")

# JSON 파일 처리 함수
def process_json(file_path):
    import json
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

# PDF 파일 삭제 기능
def delete_pdf(pdf_filename):
    pdf_path = os.path.join(UPLOAD_DIR, pdf_filename)
    if os.path.exists(pdf_path):
        os.remove(pdf_path)
        uploaded_pdfs = os.listdir(UPLOAD_DIR)  # PDF 목록을 업데이트
        return f"✅ PDF 파일 '{pdf_filename}'이 삭제되었습니다!", uploaded_pdfs
    else:
        return f"❌ 파일 '{pdf_filename}'을(를) 찾을 수 없습니다.", uploaded_pdfs

# 📝 **Q&A 시스템 (사용자 질문에 대한 답변)**
def ask_question(question, chat_history, chat_name):
    if not question.strip():
        return "❌ 질문을 입력해주세요!", chat_history

    # 문서에서 관련된 내용을 검색
    retrieved_docs = retriever.get_relevant_documents(question)
    
    if not retrieved_docs:
        return "❌ 관련된 문서를 찾을 수 없습니다.", chat_history

    # 한국어로 답변을 요청하는 프롬프트 생성
    prompt = create_prompt(question, retrieved_docs)
    
    # `qa_chain` 실행, 프롬프트 템플릿을 사용하여 한국어 답변 유도
    answer = qa_chain.run(prompt)
    
    # 대화 기록에 사용자의 질문과 Bot의 답변만 간결하게 저장
    chat_history.append((question, answer))

    return "", chat_history  # 대화 기록 업데이트 후 출력 값 리턴

# 🎨 **Gradio UI 디자인**
with gr.Blocks() as demo:
    gr.Markdown("**LLM-RAG 기반 농업용 언어모델 적용 챗봇 시스템**")

    with gr.Row():
        with gr.Column(scale=1, min_width=200):
            # 네비게이션 관련 UI
            gr.Markdown("**문서**")
            file_input = gr.File(label="Upload File")
            file_type_dropdown = gr.Dropdown(choices=["pdf", "csv", "json"], label="파일 유형 선택")
            upload_button = gr.Button("업로드")
            output_text = gr.Textbox(label="처리 상태", interactive=False)
            
            upload_button.click(fn=handle_upload, inputs=[file_input, file_type_dropdown], outputs=[output_text])

            # PDF 목록과 삭제 기능
            gr.Markdown("🗂 **업로드된 PDF 목록**")
            pdf_list = gr.Dropdown(choices=os.listdir(UPLOAD_DIR), label="Uploaded PDFs")
            delete_button = gr.Button("삭제")
            delete_output = gr.Textbox(label="삭제 상태", interactive=False)
            
            delete_button.click(fn=delete_pdf, inputs=[pdf_list], outputs=[delete_output, pdf_list])

        with gr.Column(scale=3):
            # 채팅 이름 입력란
            gr.Markdown("🗨 **채팅 이름**")
            chat_name_input = gr.Textbox(label="채팅 이름 입력", placeholder="채팅 이름을 입력하세요", interactive=True)

            # 대화 내역을 보여주는 Textbox (메신저처럼 대화가 쌓이는 영역)
            chat_output = gr.Chatbot(label="대화 내역", height=600)

            # 하단에 사용자가 질문을 입력하고, 전송하는 부분
            with gr.Row():
                with gr.Column(scale=0.75):
                    user_input = gr.Textbox(label="질문 입력", placeholder="질문을 입력하세요", interactive=True)
                with gr.Column(scale=0.25):
                    submit_btn = gr.Button("전송")

            # submit 버튼을 누르면 질문을 처리하고 답변을 반환하는 함수 연결
            submit_btn.click(fn=ask_question, inputs=[user_input, chat_output, chat_name_input], outputs=[user_input, chat_output])

demo.launch()
