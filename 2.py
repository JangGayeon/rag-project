import os
import sys
import PyPDF2
import ollama
import whisper
import speech_recognition as sr
import pyttsx3
from pymongo import MongoClient
from PyQt6.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel, QVBoxLayout, QTextBrowser,
    QFileDialog, QLineEdit, QHBoxLayout, QFrame
)
from PyQt6.QtGui import QFont, QTextCursor
from PyQt6.QtCore import Qt

# 🔹 MongoDB 연결
client = MongoClient("mongodb://localhost:27017/")
db = client["pdf_search"]
collection = db["documents"]
status_collection = db["status"]  # 난방 상태를 저장할 컬렉션

# 🔹 MongoDB에서 모든 문서를 오름차순(_id 기준)으로 불러오는 함수
def load_all_documents():
    all_texts = []
    for doc in collection.find().sort("_id", 1):  # 문서 순서대로 불러오기
        all_texts.append(doc['content'])
    return " ".join(all_texts)

# 🔹 AI가 학습한 내용을 저장하는 변수
ai_memory = ""

# 🔹 한 문서를 요약하는 함수
def summarize_text(text):
    prompt = f"다음 문서를 간략하게 요약해줘:\n\n{text}"
    response = ollama.chat(model="gemma", messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]

# 🔹 모든 문서를 개별적으로 요약한 후 결합하는 함수
def summarize_all_documents():
    summaries = []
    for doc in collection.find().sort("_id", 1):  # 오름차순으로 모든 문서를 가져옴
        content = doc.get('content', '')
        if content.strip():  # 내용이 비어있지 않은 경우만
            summary = summarize_text(content)  # summarize_text는 한 문서에 대한 요약을 반환
            summaries.append(summary)
    return " ".join(summaries)  # 모든 요약문을 하나로 결합

# 🔹 MongoDB에서 난방 상태를 가져오는 함수
def get_heating_status():
    status = status_collection.find_one({"device": "heating"})
    if status:
        return status["state"]  # 1이면 가동중, 0이면 가동 안됨
    return 0  # 기본적으로 가동 안됨

# 🔹 난방 상태를 변경하는 함수 (1: 가동, 0: 정지)
def set_heating_status(state):
    status_collection.update_one(
        {"device": "heating"},
        {"$set": {"state": state}},
        upsert=True
    )

# 🔹 Ollama로 문서 학습 (전체 문서가 너무 길면 개별 요약문 결합)
def train_ai_on_documents():
    global ai_memory
    print("🧠 AI가 모든 문서를 학습 중...")
    
    docs_text = load_all_documents()
    if not docs_text.strip():
        return "📂 데이터베이스에 저장된 문서가 없습니다."
    
    if len(docs_text) > 10000:
        summary_text = summarize_all_documents()
    else:
        summary_text = docs_text

    prompt = f"""
    아래는 데이터베이스에서 가져온 문서의 핵심 내용입니다:
    "{summary_text}"
    이 문서들을 분석하고, 질문이 들어오면 이 내용을 기반으로 대답하도록 학습해줘.
    """
    response = ollama.chat(model="gemma", messages=[{"role": "user", "content": prompt}])
    ai_memory = response["message"]["content"]
    return ai_memory

# 🔹 Ollama를 이용해 문서 내용을 학습한 후, 질문을 이해하고 답변 생성
def generate_humanlike_response(query):
    print("🧠 AI가 문서를 활용하여 답변 생성 중...")
    if not ai_memory:
        return "📂 AI가 아직 문서를 학습하지 않았습니다. 먼저 학습을 진행하세요."

    prompt = f"""
    사용자 질문: "{query}"
    [저장된 문서 내용]
    "{ai_memory}"
    위의 문서 내용을 기반으로 사용자의 질문에 대한 답변을 정확하고 자연스럽게 생성해줘.
    반드시 문서에서 배운 내용을 활용해서 대답해야 해.
    """
    response = ollama.chat(model="gemma", messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]

# 🔹 난방 상태에 따른 대답 생성
def control_heating_system(query):
    status = get_heating_status()  # 현재 난방 상태 가져오기
    if "난방 가동해줘" in query:
        if status == 1:
            return "현재 난방이 이미 가동 중입니다."
        else:
            set_heating_status(1)  # 난방 가동
            return "난방을 가동합니다."
    elif "난방 꺼줘" in query:
        if status == 0:
            return "현재 난방은 이미 꺼져 있습니다."
        else:
            set_heating_status(0)  # 난방 정지
            return "난방을 껐습니다."
    return "무슨 명령을 원하시는지 알 수 없습니다."

# 🔹 STT: 마이크에서 음성을 직접 입력받아 텍스트로 변환
def speech_to_text_from_mic():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("\n🎤 말하세요...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        query = recognizer.recognize_google(audio, language="ko-KR")
        print(f"🎤 인식된 음성: {query}")
        return query
    except sr.UnknownValueError:
        print("❌ 음성을 인식할 수 없습니다.")
        return None
    except sr.RequestError:
        print("❌ Google STT 서비스에 접근할 수 없습니다.")
        return None

# 🔹 TTS: 텍스트를 음성으로 변환
def text_to_speech(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# 🔹 PDF 파일을 MongoDB에 저장하는 함수
def store_pdf_in_mongo(file_path):
    if not file_path:
        return "❌ 파일이 선택되지 않았습니다."

    try:
        with open(file_path, "rb") as pdf_file:
            reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"

        if not text.strip():
            return "❌ PDF에서 텍스트를 추출할 수 없습니다."

        document = {"filename": os.path.basename(file_path), "content": text}
        collection.insert_one(document)
        return f"✅ {os.path.basename(file_path)} 저장 완료!"
    except Exception as e:
        return f"❌ 오류 발생: {str(e)}"

# 🔹 GUI (ChatGPT 다크 테마 스타일)
class VoiceSearchApp(QWidget):
    def __init__(self):
        super().__init__()
        self.ai_memory = ""
        self.initUI()
        # 초기 실행 시 AI 학습
        self.ai_memory = train_ai_on_documents()

    def initUI(self):
        self.setWindowTitle("ChatGPT 스타일 AI 검색")
        self.setGeometry(100, 100, 600, 650)

        # 다크 테마 적용을 위한 스타일시트
        self.setStyleSheet("""
            QWidget {
                background-color: #121212;
                color: #ffffff;
                font-family: 'Arial';
            }
            QLineEdit {
                background-color: #1f1f1f;
                border: 1px solid #3c3c3c;
                border-radius: 4px;
                padding: 6px;
                color: #ffffff;
            }
            QTextBrowser {
                background-color: #1f1f1f;
                border: 1px solid #3c3c3c;
                border-radius: 4px;
                color: #ffffff;
                padding: 6px;
            }
            QPushButton {
                background-color: #2d2d2d;
                border: none;
                border-radius: 4px;
                padding: 6px 12px;
                color: #ffffff;
            }
            QPushButton:hover {
                background-color: #3c3c3c;
            }
            QLabel {
                color: #ffffff;
            }
        """)

        # 상단 바 (헤더) 생성
        self.header = QFrame(self)
        self.header.setStyleSheet("background-color: #1F1F1F;")
        self.headerLayout = QHBoxLayout()
        self.header.setLayout(self.headerLayout)

        # KMS 로고(텍스트) 예시
        self.titleLabel = QLabel("KMS")
        self.titleLabel.setFont(QFont("Arial", 16))
        self.headerLayout.addWidget(self.titleLabel)

        # 안내 라벨
        self.label = QLabel("🎤 음성 또는 채팅으로 질문하세요!")
        self.label.setFont(QFont("Arial", 12))

        # 채팅창
        self.chatArea = QTextBrowser()
        self.chatArea.setFont(QFont("Arial", 11))
        self.chatArea.setOpenExternalLinks(True)

        # 음성 검색 버튼
        self.btnStart = QPushButton("🎤 음성 검색")
        self.btnStart.clicked.connect(self.start_voice_search)

        # 파일 업로드 버튼
        self.btnUpload = QPushButton("📂 PDF 업로드")
        self.btnUpload.clicked.connect(self.upload_file_to_mongo)

        # 채팅 입력창 & 버튼
        self.chatInput = QLineEdit()
        self.chatInput.setPlaceholderText("메시지를 입력하세요...")
        self.chatInput.setFont(QFont("Arial", 11))

        self.btnChat = QPushButton("➡️")
        self.btnChat.setFixedWidth(50)
        self.btnChat.clicked.connect(self.start_chat_search)

        chatLayout = QHBoxLayout()
        chatLayout.addWidget(self.chatInput)
        chatLayout.addWidget(self.btnChat)

        # 메인 레이아웃
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.header)    # 상단 바
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.chatArea)
        self.layout.addWidget(self.btnStart)
        self.layout.addWidget(self.btnUpload)
        self.layout.addLayout(chatLayout)

        self.setLayout(self.layout)

    # 🔹 PDF 파일 업로드 & AI 재학습
    def upload_file_to_mongo(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "파일 선택", "", "PDF Files (*.pdf)")
        if file_path:
            result = store_pdf_in_mongo(file_path)
            self.chatArea.append(result)
            # PDF 업로드 후, 새롭게 AI 학습
            self.ai_memory = train_ai_on_documents()

    # 🔹 음성 검색 실행
    def start_voice_search(self):
        query = speech_to_text_from_mic()
        if query:
            self.chatInput.setText(query)
            self.start_chat_search()

    # 🔹 채팅 검색(질의응답)
    def start_chat_search(self):
        user_query = self.chatInput.text().strip()
        if not user_query:
            return

        # 난방 제어 관련 명령이 있으면 처리
        response = control_heating_system(user_query)
        if response:
            self.add_chat_message("AI", response)
        else:
            # 사용자 메시지 출력
            self.add_chat_message("User", user_query)
            self.chatInput.clear()

            # Ollama 기반 응답 생성
            answer = generate_humanlike_response(user_query)
            self.add_chat_message("AI", answer)

    # 🔹 UI에 메시지 추가
    def add_chat_message(self, sender, message):
        formatted_message = f"<b>{sender}:</b> {message}<br>"
        self.chatArea.append(formatted_message)
        self.chatArea.moveCursor(QTextCursor.MoveOperation.End)


# ✅ 실행
if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = VoiceSearchApp()
    ex.show()
    sys.exit(app.exec())
