import os
import json
import faiss
import pandas as pd
import numpy as np
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from langchain_huggingface import HuggingFaceEndpoint
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load Environment Variables
load_dotenv()
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

# Initialize FastAPI
app = FastAPI(
    title="TravelIQ API",
    description="Ask questions about hotel bookings & retrieve analytics.",
    version="1.0.0"
)

# Root Endpoint
@app.get("/")
def root():
    return {"message": "FastAPI is running! Use /ask for queries and /analytics for insights."}

# Load FAISS Index
faiss_index_path = os.path.join(os.getcwd(), "faiss_index.bin")
if not os.path.exists(faiss_index_path):
    raise FileNotFoundError(f"ERROR: FAISS index not found at {faiss_index_path}")
faiss_index = faiss.read_index(faiss_index_path)
logger.info(f"FAISS index loaded from {faiss_index_path}")

# Load Hotel Booking Dataset
data_path = os.path.join(os.getcwd(), "data", "hotel_bookings_with_embeddings.csv")
if not os.path.exists(data_path):
    raise FileNotFoundError(f"ERROR: Dataset not found at {data_path}")
df = pd.read_csv(data_path)
logger.info(f"Dataset loaded with {len(df)} records.")

# Load Sentence Transformer for Embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load Mistral AI Model via Hugging Face
try:
    llm = HuggingFaceEndpoint(
        endpoint_url="https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3",
        huggingfacehub_api_token=HUGGINGFACE_API_KEY,
        temperature=0.7,
        max_length=200
    )
    logger.info("Mistral AI LLM loaded successfully!")
except Exception as e:
    logger.error(f"Failed to load Mistral AI model: {e}")
    raise HTTPException(status_code=500, detail="Failed to load AI model.")

# Function to Search FAISS for Relevant Bookings
def search_faiss(query, top_k=3):
    """Retrieve the most relevant booking details from FAISS."""
    try:
        query_embedding = embedding_model.encode([query]).astype(np.float32)
        distances, indices = faiss_index.search(query_embedding, top_k)
        retrieved_docs = df.iloc[indices[0]][["hotel", "country", "customer_type"]].astype(str).agg(" | ".join, axis=1).tolist()
        return retrieved_docs
    except Exception as e:
        logger.error(f"FAISS search error: {e}")
        return ["No relevant booking found."]

# Function to Generate AI Response
# Function to Generate AI Response with Analytics Support
def generate_answer(query):
    """Retrieve relevant data from FAISS and analytics, then generate an AI response."""
    
    # Search FAISS for related records
    retrieved_docs = search_faiss(query)

    # Check if the question is about analytics data
    analytics_keywords = ["cancellation rate", "revenue", "customer insights", "market segment"]
    analytics_response = None

    for keyword in analytics_keywords:
        if keyword in query.lower():
            analytics_response = analytics_data_fixed.get(keyword.replace(" ", "_"), "No data available.")

    # Prepare AI prompt with both FAISS data and analytics if available
    prompt = f"Context: {retrieved_docs}\n\n"
    
    if analytics_response:
        prompt += f"Additional Data: {analytics_response}\n\n"
    
    prompt += f"Question: {query}"

    # Generate response from Mistral AI
    try:
        response = llm(prompt)
    except Exception as e:
        logger.error(f"AI generation error: {e}")
        response = "Error generating response. Please try again."

    return response

# Pydantic Model for Multiple Queries
class QueryRequest(BaseModel):
    queries: List[str]  # Accepts a list of queries

# FastAPI Endpoint for AI Queries (Multiple Queries Supported)
@app.post("/ask", summary="Ask multiple booking-related questions", response_model=dict)
def ask_questions(request: QueryRequest):
    """
    *Ask multiple questions related to hotel bookings.*  
    - Send a JSON object: {"queries": ["Question 1", "Question 2", ...]}
    - The response contains answers for all queries.
    """
    try:
        answers = {query: generate_answer(query) for query in request.queries}
        return {"responses": answers}
    except Exception as e:
        logger.error(f"Error in /ask: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Load Precomputed Analytics Data
analytics_file = os.path.join(os.getcwd(), "data", "analytics_result.json")

if os.path.exists(analytics_file):
    with open(analytics_file, "r") as file:
        try:
            analytics_data = json.load(file)
            logger.info("Analytics data loaded successfully.")
        except json.JSONDecodeError:
            logger.error("ERROR: JSON decoding failed! Defaulting to empty.")
            analytics_data = {}
else:
    logger.error("ERROR: Analytics file not found! Returning empty data.")
    analytics_data = {}

# Convert Keys & Values for Better Readability
def clean_analytics_data(data):
    """Convert analytics keys & values for better formatting."""
    fixed_data = {}

    for key, value in data.items():
        if key == "revenue_trends":
            fixed_data[key] = {k.replace("Period('", "").replace("', 'M')", ""): v for k, v in value.items()}
        elif key == "cancellation_rate":
            fixed_data[key] = float(value.replace("%", ""))
        else:
            fixed_data[key] = value

    return fixed_data

analytics_data_fixed = clean_analytics_data(analytics_data)

# Add Extra Insights to Analytics
def add_extra_insights(data):
    """Compute & add extra insights to analytics data."""
    try:
        new_insights = {
            "most_booked_hotel": "Resort Hotel" if analytics_data["customer_insights"]["most_common_customer_type"] == "Transient" else "City Hotel",
            "peak_booking_month": max(analytics_data["revenue_trends"], key=analytics_data["revenue_trends"].get),
            "average_revenue_per_booking": round(sum(analytics_data["revenue_trends"].values()) / len(analytics_data["revenue_trends"]), 2)
        }
        data.update(new_insights)
    except Exception as e:
        logger.error(f"Error adding extra insights: {e}")

    return data

analytics_data_fixed = add_extra_insights(analytics_data_fixed)

# Analytics Response Model
class AnalyticsResponse(BaseModel):
    revenue_trends: Dict[str, float]
    cancellation_rate: float
    lead_time_distribution: Dict[str, Any]
    geographical_distribution: Dict[str, int]
    customer_insights: Dict[str, Any]
    market_segment_analysis: Dict[str, Any]
    special_requests_analysis: Dict[str, Any]
    most_booked_hotel: str
    peak_booking_month: str
    average_revenue_per_booking: float

# FastAPI Endpoint for Analytics (Compulsory)
@app.post("/analytics", summary="Retrieve hotel booking analytics", response_model=AnalyticsResponse)
def get_analytics():
    """
    *Retrieve precomputed analytics data* for hotel bookings.  
    - Includes revenue trends, cancellation rates, etc.
    - Adds *new insights* like *most booked hotel, peak month, & average revenue per booking*.
    """
    if analytics_data_fixed:
        return analytics_data_fixed
    else:
        raise HTTPException(status_code=500, detail="No analytics data available.")
