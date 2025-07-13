# EZ-Data-Science-Intern-Assignment

# 🧠 Smart Research Assistant

An AI-powered document assistant built with Streamlit. Upload a PDF or TXT file and get:

- 📋 Auto-generated summary
- 💬 Context-based Q&A
- 🧠 Logic-based challenge questions with evaluation

Powered by [OpenRouter](https://openrouter.ai) using free LLMs like Mistral or Claude.



## 📦 Requirements

Create a `requirements.txt` file with the following dependencies:

```txt
streamlit>=1.28.0
PyPDF2>=3.0.0
openai>=1.3.0
python-dotenv>=1.0.0


## 🚀 Installation and Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. OpenRouter API Integration

The app uses OpenRouter to call LLMs like Mistral.

Set up the LLM call in your `app.py` like this:

```python
import openai
import os
from dotenv import load_dotenv

load_dotenv()
openai.api_base = "https://openrouter.ai/api/v1"
openai.api_key = os.getenv("OPENAI_API_KEY")

def llm_response(prompt: str, context: str) -> str:
    response = openai.ChatCompletion.create(
        model="mistralai/mistral-7b-instruct",
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Only answer using the provided document context."},
            {"role": "user", "content": f"{prompt}\n\nContext:\n{context}"}
        ],
        temperature=0.3,
        max_tokens=500
    )
    return response.choices[0].message["content"].strip()

### 3. Environment Variables

Create a `.env` file in your project directory:

```env
OPENAI_API_KEY=your_openrouter_key_here
```

Get your key from https://openrouter.ai/ and make sure it's enabled for your selected model.



## 🖥 Running the Application

### In Jupyter Notebook (for testing only)

```python
!pip install streamlit PyPDF2

with open('app.py', 'w') as f:
    f.write("""Paste your app.py code here""")

!streamlit run app.py
```

### From Terminal (recommended)

```bash
streamlit run app.py
```

---

## ✅ Features

### 📚 Core Functionality
- 📄 **PDF/TXT Upload**
- 🧾 **Auto Summary (≤150 words)**
- 💬 **Ask Anything Mode**
- 🧠 **Challenge Mode with Logic Questions**
- 🔎 **Document-grounded Responses**
- 📍 **Answer Justification Paragraphs**

### ⚙ Technical Stack
- 💻 Built with **Streamlit**
- 🔐 `.env` for API key protection
- 🧠 LLM via OpenRouter (e.g., Mistral)

### ✨ Bonus Features
- 🧠 Memory Tracking with Q&A history
- 📌 Paragraph-level supporting evidence
- 🧩 Progressive challenge question difficulty



## 🧠 App Architecture


📦 Smart Research Assistant
├── 🔧 DocumentProcessor
│   ├── PDF/TXT extraction
│   ├── Content segmentation
│   └── Summary generation
├── 🧠 QuestionGenerator
│   ├── Create logic-based Qs
│   ├── Evaluate answers
│   └── Provide feedback
├── 💬 Conversation Tracker
│   ├── Maintains chat history
│   ├── Manages Q&A
│   └── Anchors responses
└── 🎨 Streamlit Interface
    ├── File upload UI
    ├── Mode toggles
    └── Interactive Q&A / Challenges


## 🔁 Usage Flow


1. **Upload** a PDF or TXT
2. **Get Summary** automatically
3. **Pick Mode**: Ask or Challenge
4. **Receive Responses** with supporting context
5. **Track Your Questions & Answers**


## 🧪 Testing Tips

Try uploading:
- 📝 Research Papers
- ⚖ Legal Docs
- 🔬 Scientific Articles
- 📊 Financial Reports


## 🌐 Deployment Notes

- Runs on `http://localhost:8501`
- Works offline (except LLM call)
- Streamlit handles UI + sessions



## 🛠 Troubleshooting

| Issue                  | Fix                                                   |
|------------------------|--------------------------------------------------------|
| PDF errors             | Ensure it's a text-based (non-scanned) PDF            |
| Empty responses        | Check API key or model ID validity                    |
| Model not working      | Use `mistralai/mistral-7b-instruct` (recommended)     |
| `.env` not working     | Ensure file is named exactly `.env`                  |
| UI lag with large docs | Add chunking / reduce input to 1000 tokens            |



