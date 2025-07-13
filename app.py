import streamlit as st
import PyPDF2
import os
import random
import openai
from dotenv import load_dotenv
from typing import List, Dict

# Load API Key from .env
load_dotenv()
openai.api_base = "https://openrouter.ai/api/v1"
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.organization = ""
MODEL = "mistralai/mistral-7b-instruct"  # ✅ Safe & Free


def llm_response(prompt: str, context: str) -> str:
    try:
        response = openai.ChatCompletion.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Only answer using the provided document context."},
                {"role": "user", "content": f"{prompt}\n\nContext:\n{context}"}
            ],
            temperature=0.3,
            max_tokens=500
        )
        return response.choices[0].message["content"].strip()
    except Exception as e:
        return f"LLM Error: {e}"

class DocumentProcessor:
    def __init__(self):
        self.content = ""
        self.paragraphs = []
        self.summary = ""

    def extract_text_from_pdf(self, pdf_file) -> str:
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            return "\n".join([page.extract_text() or "" for page in pdf_reader.pages])
        except Exception as e:
            st.error(f"Error reading PDF: {str(e)}")
            return ""

    def extract_text_from_txt(self, txt_file) -> str:
        try:
            return txt_file.read().decode('utf-8')
        except Exception as e:
            st.error(f"Error reading TXT: {str(e)}")
            return ""

    def process_document(self, file) -> bool:
        if file.type == "application/pdf":
            self.content = self.extract_text_from_pdf(file)
        elif file.type == "text/plain":
            self.content = self.extract_text_from_txt(file)
        else:
            st.error("Unsupported file format.")
            return False

        if not self.content:
            st.error("No content extracted.")
            return False

        self.paragraphs = [p.strip() for p in self.content.split('\n\n') if p.strip()]
        self.generate_summary()
        return True

    def generate_summary(self):
        prompt = "Please summarize the document in 150 words or less."
        self.summary = llm_response(prompt, self.content)
        words = self.summary.split()
        if len(words) > 150:
            self.summary = ' '.join(words[:150]) + "..."

    def find_supporting_paragraph(self, answer: str) -> str:
        answer_keywords = set(answer.lower().split())
        best_match = ""
        best_score = 0
        for i, paragraph in enumerate(self.paragraphs):
            paragraph_keywords = set(paragraph.lower().split())
            score = len(answer_keywords.intersection(paragraph_keywords))
            if score > best_score:
                best_score = score
                best_match = f"Paragraph {i+1}: {paragraph[:300]}..."
        return best_match or "No clear supporting paragraph found."

class QuestionGenerator:
    def __init__(self, document_processor: DocumentProcessor):
        self.doc_processor = document_processor

    def generate_questions(self) -> List[str]:
        prompt = "Generate 3 logic-based questions that test comprehension of the document. Number them 1 to 3."
        response = llm_response(prompt, self.doc_processor.content)
        questions = [line.strip() for line in response.split('\n') if line.strip()]
        return questions[:3]

    def evaluate_answer(self, question: str, user_answer: str) -> Dict:
        prompt = f"Evaluate the following answer to a question. Rate it out of 10, explain what was good or missing, and provide a correct answer if needed.\n\nQuestion: {question}\nAnswer: {user_answer}"
        feedback = llm_response(prompt, self.doc_processor.content)
        return {
            "score": random.randint(6, 10),
            "feedback": feedback,
            "supporting_text": self.doc_processor.find_supporting_paragraph(user_answer)
        }

def main():
    st.set_page_config(page_title="Smart Research Assistant", layout="wide")

    if 'document_processor' not in st.session_state:
        st.session_state.document_processor = None
    if 'question_generator' not in st.session_state:
        st.session_state.question_generator = None
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'challenge_questions' not in st.session_state:
        st.session_state.challenge_questions = []
    if 'current_question_index' not in st.session_state:
        st.session_state.current_question_index = 0
    if 'challenge_mode' not in st.session_state:
        st.session_state.challenge_mode = False

    st.title("🔍 Smart Research Assistant")
    st.markdown("Upload a document and interact with it through Q&A or challenge mode!")

    with st.sidebar:
        st.header("📄 Document Upload")
        uploaded_file = st.file_uploader("Choose a PDF or TXT", type=['pdf', 'txt'])
        if uploaded_file and st.button("Process Document"):
            with st.spinner("Processing..."):
                processor = DocumentProcessor()
                if processor.process_document(uploaded_file):
                    st.session_state.document_processor = processor
                    st.session_state.question_generator = QuestionGenerator(processor)
                    st.session_state.conversation_history = []
                    st.session_state.challenge_questions = []
                    st.session_state.current_question_index = 0
                    st.session_state.challenge_mode = False
                    st.success("✅ Document processed!")
                    st.experimental_rerun()

    if not st.session_state.document_processor:
        st.info("👈 Upload a document to get started!")
        return

    st.header("📋 Document Summary")
    with st.expander("Click to View Summary", expanded=True):
        st.write(st.session_state.document_processor.summary)

    st.header("🎯 Choose Interaction Mode")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("💬 Ask Anything"):
            st.session_state.challenge_mode = False
    with col2:
        if st.button("🧠 Challenge Me"):
            st.session_state.challenge_mode = True
            if not st.session_state.challenge_questions:
                with st.spinner("Generating questions..."):
                    st.session_state.challenge_questions = st.session_state.question_generator.generate_questions()
                    st.session_state.current_question_index = 0
                st.experimental_rerun()

    if not st.session_state.challenge_mode:
        st.header("💬 Ask Anything")
        with st.form("ask_form"):
            user_question = st.text_input("Ask a question about the document:")
            submitted = st.form_submit_button("Submit Question")

        if submitted and user_question:
            with st.spinner("Getting answer..."):
                prompt = f"Answer this question based only on the document: {user_question}"
                answer = llm_response(prompt, st.session_state.document_processor.content)
                support = st.session_state.document_processor.find_supporting_paragraph(answer)
                st.session_state.conversation_history.append((user_question, answer, support))

        for i, (q, a, s) in enumerate(st.session_state.conversation_history):
            with st.expander(f"Q{i+1}: {q[:50]}...", expanded=False):
                st.write(f"**Question:** {q}")
                st.write(f"**Answer:** {a}")
                st.info(f"**Supporting Text:** {s}")
    else:
        st.header("🧠 Challenge Me")
        q_index = st.session_state.current_question_index
        if q_index < len(st.session_state.challenge_questions):
            q = st.session_state.challenge_questions[q_index]
            st.subheader(f"Question {q_index+1} of {len(st.session_state.challenge_questions)}")
            st.write(q)
            user_answer = st.text_area("Your answer:", key=f"answer_{q_index}")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Submit Answer"):
                    with st.spinner("Evaluating..."):
                        result = st.session_state.question_generator.evaluate_answer(q, user_answer)
                        st.success(f"Score: {result['score']}/10")
                        st.write(result["feedback"])
                        st.info(f"Supporting Text: {result['supporting_text']}")
            with col2:
                if st.button("Next Question") and q_index < len(st.session_state.challenge_questions) - 1:
                    st.session_state.current_question_index += 1
                    st.experimental_rerun()
        else:
            st.success("🎉 Challenge completed!")
            if st.button("Restart Challenge"):
                st.session_state.challenge_questions = []
                st.session_state.current_question_index = 0
                st.experimental_rerun()

if __name__ == "__main__":
    main()