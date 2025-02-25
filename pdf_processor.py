import fitz  # PyMuPDF
import os
import pytesseract
from pdf2image import convert_from_path
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# PDF에서 OCR을 사용하여 텍스트 추출
def extract_text_with_ocr(pdf_path):
    """OCR을 사용하여 PDF에서 텍스트를 추출하는 함수"""
    text = ""
    images = convert_from_path(pdf_path)
    for img in images:
        text += pytesseract.image_to_string(img, lang="kor+eng")  # 한글 + 영어 OCR
    return text

# 기존 텍스트 추출 방법 (OCR 없이)
def extract_text_from_pdf(pdf_path):
    """PDF 파일에서 텍스트를 추출하는 기본 함수"""
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text("text")
    return text if text.strip() else extract_text_with_ocr(pdf_path)  # OCR 적용

# ChromaDB 저장 경로
CHROMA_DB_PATH = "C:/rag-project/chroma_db"

# Ollama 임베딩 모델
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

def process_pdf(pdf_path):
    """PDF를 분석하고 벡터 DB에 저장하는 함수"""
    extracted_text = extract_text_from_pdf(pdf_path)
    
    if not extracted_text.strip():
        print("❌ PDF에서 텍스트를 추출할 수 없습니다.")
        return
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_text(extracted_text)

    # 벡터 DB에 저장
    vectorstore = Chroma.from_texts(texts, embeddings, persist_directory=CHROMA_DB_PATH)
    print("✅ PDF 문서가 벡터 DB에 저장되었습니다!")

# 직접 실행할 경우
if __name__ == "__main__":
    pdf_file = "C:/rag-project/pdf-files/sample_paper.pdf"  # 실제 PDF 경로 입력
    process_pdf(pdf_file)

def process_all_pdfs(directory="C:/rag-project/pdf-files/"):
    """지정된 폴더 내 모든 PDF 파일을 처리하여 벡터 DB에 저장"""
    pdf_files = [f for f in os.listdir(directory) if f.endswith(".pdf")]

    if not pdf_files:
        print("❌ PDF 파일이 없습니다.")
        return

    for pdf in pdf_files:
        pdf_path = os.path.join(directory, pdf)
        print(f"📄 Processing: {pdf_path}")
        process_pdf(pdf_path)  # 기존 `process_pdf` 함수 사용