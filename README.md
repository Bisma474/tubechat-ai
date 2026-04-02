<div align="center">

# ▶️ TubeChat — YouTube AI Chatbot 🤖📺

*Chat with any YouTube video instantly, for free, and in any language.*

[![Python Version](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=Streamlit&logoColor=white)](https://streamlit.io)
[![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=flat-square&logo=langchain&logoColor=white)](https://langchain.com)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=flat-square&logo=huggingface&logoColor=000)](https://huggingface.co)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](https://opensource.org/licenses/MIT)

[View Demo (Coming Soon)]() · [Report Bug]() · [Request Feature]()

</div>

---

## 📖 Table of Contents
1. [About the Project](#-about-the-project)
2. [Key Features](#-key-features)
3. [How It Works (Architecture)](#-how-it-works-architecture)
4. [Tech Stack](#-tech-stack)
5. [Getting Started (Local Installation)](#-getting-started-local-installation)
6. [Deployment (Streamlit Cloud)](#-deployment-streamlit-cloud)
7. [Usage Guide](#-usage-guide)
8. [Troubleshooting & FAQ](#-troubleshooting--faq)
9. [Limitations](#-limitations)
10. [Roadmap](#-roadmap)
11. [Author](#-author)

---

## 🌟 About the Project

**TubeChat** is a Retrieval-Augmented Generation (RAG) chatbot designed to extract, process, and converse with the contents of YouTube videos. 

Have you ever wanted to find a specific piece of information in a 2-hour podcast or summarize an educational video without watching the whole thing? TubeChat solves this by acting as your personal AI viewing assistant. You provide the YouTube URL, and TubeChat "watches" the video by processing its transcript, allowing you to ask natural language questions and get immediate, context-aware answers.

Unlike many AI tools that require expensive OpenAI API keys, **TubeChat is built to be 100% free**, leveraging the power of open-source models via the HuggingFace Inference API.

---

## ✨ Key Features

- 🌍 **Universal Language Support:** If a video isn't in English, TubeChat's underlying LangChain integration automatically detects the foreign language captions and translates them to English on the fly before processing.
- 💸 **100% Free & Open-Source LLMs:** Powered by models like Mistral 7B, Zephyr 7B, and Phi-3. No credit card or paid API keys required.
- 🧠 **Advanced RAG Pipeline:** Uses `all-MiniLM-L6-v2` for dense vector embeddings and `FAISS` for lightning-fast local semantic search.
- 🎨 **Immersive UI/UX:** Features a custom CSS-styled interface, dark mode, a live pipeline visualizer, embedded YouTube player, and real-time document statistics.
- ⚡ **High Performance:** Implements document chunking and overlapping to ensure the AI doesn't lose context, even on long videos.

---

## 🏗️ How It Works (Architecture)

TubeChat follows a strict Retrieval-Augmented Generation (RAG) pipeline to ensure the AI only answers based on the video's actual content, preventing hallucinations.

1. **📄 Load:** LangChain's `YoutubeLoader` fetches the video's transcript. If the transcript is in a foreign language, it is automatically translated to English.
2. **✂️ Split:** The transcript is passed into a `RecursiveCharacterTextSplitter` which breaks the long text into smaller, overlapping chunks (800 characters each).
3. **🔢 Embed:** Each text chunk is converted into a mathematical vector representation using the `sentence-transformers/all-MiniLM-L6-v2` model.
4. **🗄️ Index:** These embeddings are stored locally in a `FAISS` (Facebook AI Similarity Search) vector database for highly efficient similarity matching.
5. **🔍 Retrieve:** When a user asks a question, the query is embedded, and FAISS searches for the top 4 most relevant text chunks from the video.
6. **🧩 Augment & ✨ Generate:** The retrieved context is injected into a strict prompt, and the HuggingFace Inference Client generates a concise answer.

---

## 🛠️ Tech Stack

| Component | Technology Used | Purpose |
| :--- | :--- | :--- |
| **Frontend UI** | `Streamlit` | Web framework, custom CSS, state management |
| **Orchestration** | `LangChain` | Tying the LLM, loaders, and vector stores together |
| **Video Loader** | `youtube-transcript-api` & `pytube` | Fetching and parsing YouTube captions |
| **Embeddings** | `HuggingFaceEmbeddings` | Generating dense vector representations (`MiniLM`) |
| **Vector DB** | `FAISS (faiss-cpu)` | Storing and querying vectors locally |
| **LLM Engine** | `huggingface_hub` | API Client for Mistral/Zephyr text generation |

---

## 🚀 Getting Started (Local Installation)

Want to run TubeChat on your own machine? Follow these simple steps.

### Prerequisites
* Python 3.9 or higher
* Git installed
* A free [HuggingFace account](https://huggingface.co/) (to get your API Token)

### 1. Clone the repository
```bash
git clone https://github.com/bismahhhh/tubechat-ai.git
cd tubechat-ai
