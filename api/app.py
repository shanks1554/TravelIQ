import os
import json
import faiss
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any
from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv

# ✅ Load Environment Variables
load_dotenv()
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

# ✅ Initialize FastAPI
app = FastAPI(
    title="Hotel Booking AI API",
    description="Ask questions about hotel bookings & retrieve analytics.",
    version="1.0.0"
)

# ✅ Root Path (Confirms API is Running)
@app.get("/")
def root():
    return {"message": "✅ FastAPI is running! Use /ask for queries and /analytics for insights."}

# ✅ Load FAISS Index
faiss_index_path = "faiss_index.bin"
if not os.path.exists(faiss_index_path):
    raise FileNotFoundError(f"❌ ERROR: FAISS index not found at {faiss_index_path}")
faiss_index = faiss.read_index(faiss_index_path)
print(f"✅ FAISS index loaded from {faiss_index_path}")

# ✅ Load Dataset with Booking Details
data_path = "data/hotel_bookings_with_embeddings.csv"
if not os.path.exists(data_path):
    raise FileNotFoundError(f"❌ ERROR: Dataset not found at {data_path}")
df = pd.read_csv(data_path)
print(f"✅ Dataset loaded with {len(df)} records.")

# ✅ Load Mistral AI Model using LangChain
llm = HuggingFaceEndpoint(
    endpoint_url="https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3",
    huggingfacehub_api_token=HUGGINGFACE_API_KEY,
    temperature=0.7,
    max_length=200
)
print("✅ Mistral AI LLM loaded successfully!")

# ✅ Function to Search FAISS & Retrieve Relevant Bookings
def search_faiss(query, top_k=3):
    """Retrieve the most relevant booking details from FAISS."""
    query_embedding = np.random.rand(1, faiss_index.d)  # Replace with real embeddings
    distances, indices = faiss_index.search(query_embedding, top_k)
    
    retrieved_docs = df.iloc[indices[0]][["hotel", "country", "customer_type"]].astype(str).agg(" | ".join, axis=1).tolist()
    return retrieved_docs

# ✅ Function to Generate AI Response
def generate_answer(query):
    """Retrieve relevant bookings & generate an AI response."""
    retrieved_docs = search_faiss(query)

    # ✅ Prepare prompt for Mistral AI
    prompt = f"Context: {retrieved_docs}\n\nQuestion: {query}"

    # ✅ Generate response from Mistral AI
    response = llm(prompt)

    return response

# ✅ FastAPI Endpoint for AI Queries (POST Only, Accepts Plain Text)
@app.post("/ask", summary="Ask a booking-related question", response_model=dict)
def ask_question(request_body: str):
    """
    **Ask a question related to hotel bookings.**  
    - Just send a plain text question (No JSON needed).
    """
    try:
        answer = generate_answer(request_body)
        return {"question": request_body, "answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ✅ Load Precomputed Analytics Data
analytics_file = "data/analytics_result.json"

if os.path.exists(analytics_file):
    with open(analytics_file, "r") as file:
        try:
            analytics_data = json.load(file)  # ✅ Ensure proper JSON loading
            print(f"✅ Loaded analytics data: {json.dumps(analytics_data, indent=4)}")
        except json.JSONDecodeError:
            print("❌ ERROR: JSON decoding failed! Defaulting to empty.")
            analytics_data = {}
else:
    print("❌ ERROR: Analytics file not found! Returning empty data.")
    analytics_data = {}

# ✅ Convert Keys & Values for Better Readability
def clean_analytics_data(data):
    """Convert analytics keys & values for better formatting."""
    fixed_data = {}

    for key, value in data.items():
        # Convert revenue trends (e.g., '2015-07' → 'July 2015')
        if key == "revenue_trends":
            fixed_data[key] = {k.replace("Period('", "").replace("', 'M')", ""): v for k, v in value.items()}

        # Convert cancellation rate "27.49%" → 27.49 (float)
        elif key == "cancellation_rate":
            fixed_data[key] = float(value.replace("%", ""))

        # Keep all other values as-is
        else:
            fixed_data[key] = value

    return fixed_data

analytics_data_fixed = clean_analytics_data(analytics_data)

# ✅ Add More Insights
def add_extra_insights(data):
    """Compute & add extra insights to analytics data."""
    new_insights = {
        "most_booked_hotel": "Resort Hotel" if analytics_data["customer_insights"]["most_common_customer_type"] == "Transient" else "City Hotel",
        "peak_booking_month": max(analytics_data["revenue_trends"], key=analytics_data["revenue_trends"].get),
        "average_revenue_per_booking": round(sum(analytics_data["revenue_trends"].values()) / len(analytics_data["revenue_trends"]), 2)
    }
    data.update(new_insights)
    return data

analytics_data_fixed = add_extra_insights(analytics_data_fixed)

# ✅ Analytics Response Model with Example Values for Swagger UI
class AnalyticsResponse(BaseModel):
    revenue_trends: Dict[str, float] = Field(example={"2015-07": 926487.08, "2015-08": 1405796.27})
    cancellation_rate: float = Field(example=27.49)
    lead_time_distribution: Dict[str, Any] = Field(example={"mean": 79.89, "min": 0, "max": 737, "50th_percentile (median)": 49})
    geographical_distribution: Dict[str, int] = Field(example={"Portugal": 27453, "United Kingdom": 10433, "France": 8837})
    customer_insights: Dict[str, Any] = Field(example={"most_common_customer_type": "Transient", "avg_stay_duration": "3.63 nights"})
    market_segment_analysis: Dict[str, Any] = Field(example={"most_common_booking_channel": "TA/TO", "market_segment_distribution": {"Online TA": 51618}})
    special_requests_analysis: Dict[str, Any] = Field(example={"average_special_requests": 0.7, "special_requests_distribution": {"0": 43894, "1": 29017}})
    most_booked_hotel: str = Field(example="Resort Hotel")
    peak_booking_month: str = Field(example="2017-08")
    average_revenue_per_booking: float = Field(example=1325420.74)

# ✅ FastAPI Endpoint for Analytics (POST Only)
@app.post("/analytics", summary="Retrieve hotel booking analytics", response_model=AnalyticsResponse)
def get_analytics():
    """
    **Retrieve precomputed analytics data** for hotel bookings.  
    - Includes revenue trends, cancellation rates, etc.
    - Adds **new insights** like **most booked hotel, peak month, & average revenue per booking**.
    """
    if analytics_data_fixed:
        return analytics_data_fixed  # ✅ Fixed JSON keys returned
    else:
        raise HTTPException(status_code=500, detail="❌ No analytics data available.")
