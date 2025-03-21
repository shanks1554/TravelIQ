# TravelIQ
 LLM-Powered Booking Analytics &amp; QA System that processes hotel booking data, extracts insights, and enables retrieval-augmented question answering (RAG).

# 📚 Project Overview

This project processes hotel booking data, extracts insights, and enables retrieval-augmented question answering (RAG) using FAISS and Mistral AI.

It includes:

✔ Hotel Booking Data Processing 📊 

✔ FAISS-powered Vector Search ⚡

✔ Mistral AI LLM Integration for Q&A 🤖

✔ FastAPI RESTful API 🚀

# 1. Clone the Project
```bash
git clone https://github.com/shanks1554/TravelIQ
cd TravelIQ
```

# 2. Create a Virtual Environment

🔹 If using Conda:

```bash
conda create --name traveliq python -y
conda activate traveliq
```

🔹 If using venv:
```bash
python -m venv venv
source venv/bin/activate  # For Mac/Linux
venv\Scripts\activate # For Windows
```

# 3. Install Dependencies

```bash 
pip install -r requirements.txt
```
# 4. Create a Hugging Face access token and add it in created .env file in following format

```bash
HUGGINGFACE_API_KEY = YOUR API KEY
```
# 5. Setup Required Files

Before running the API, download required files OR generate them manually:

# Option 1: Generate Files Manually

```bash
jupyter notebook 03_vector_search.ipynb
```

This will create:

```bash
faiss_index.bin  
hotel_bookings_embeddings.npy  
```

# Option 2: Download Precomputed Files

Download FAISS & Embeddings from Google Drive

https://drive.google.com/drive/folders/1PxDP44uWNy8G8BDzT_4pswfoEkIrr4CD

After downloading, place them in root directory.

# 6. Run the API

```bash
uvicorn api.app:app --reload
```

Then open: http://127.0.0.1:8000/docs

# 7. Additional Documentation

📄 Sample Test Queries & Expected Answers → test_queries.md

📄 Implementation Report → implementation_report.md

# 8. Test Results

Test Results are stored in `Test Results` 
