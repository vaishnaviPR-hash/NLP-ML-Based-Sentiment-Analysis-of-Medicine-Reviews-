# 💊 Medicine Review Sentiment Analyzer

A web app that analyzes medicine reviews using a **multi-task BERT model**.  
It predicts the sentiment for four aspects of each review:

- ✅ **Overall sentiment**  
- 💪 **Effectiveness sentiment**  
- 💸 **Price sentiment**  
- ⚕️ **Side effects sentiment**

Users can submit their reviews through a sleek frontend and instantly see AI-driven analysis of their experience.

---

## 🚀 Features

- 🧠 **Multi-task BERT** model with four output heads  
- 🌐 **FastAPI backend** for real-time inference  
- 🎨 **Modern TailwindCSS frontend** for seamless interaction  
- ⚙️ **Fully connected** backend–frontend pipeline  
- ☁️ **Ready for deployment on Render** or other platforms  
- ✅➖❌ **Visual sentiment mapping** (Positive / Neutral / Negative)

---

## 🧩 Tech Stack

| Component | Technology |
|------------|-------------|
| Model | BERT (PyTorch, Transformers) |
| Backend | FastAPI |
| Frontend | HTML, TailwindCSS, JavaScript |
| Deployment | Render / Localhost |

---

## 📁 Folder Structure

medicine-review-sentiment-analyzer/
│
├── backend/
│ ├── main.py # FastAPI app
│ ├── model.pkl # Trained BERT model
│ ├── tokenizer/ # Tokenizer folder (from Hugging Face)
│ ├── requirements.txt # Python dependencies
│
├── frontend/
│ └── index.html # User interface
│
├── README.md
