from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import ee
import json
import os
import time
from dotenv import load_dotenv
import logging

# Pinecone and Groq imports for RAG chatbot
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------
#  FASTAPI APP
# ---------------------------
app = FastAPI(title="AgriXVision Backend (GEE)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
#  EARTH ENGINE AUTH
# ---------------------------
PROJECT_ID = "premium-origin-469307-t0"

# Try to load from file first, then fall back to environment variable
service_account_file = "service-account.json"
if os.path.exists(service_account_file):
    with open(service_account_file, 'r') as f:
        service_json = f.read()
    service_account_info = json.loads(service_json)
else:
    service_json = os.getenv("SERVICE_ACCOUNT_JSON")
    if not service_json:
        raise Exception("SERVICE_ACCOUNT_JSON variable missing and service-account.json file not found")
    service_account_info = json.loads(service_json)

credentials = ee.ServiceAccountCredentials(
    email=service_account_info["client_email"],
    key_data=service_json
)

try:
    ee.Initialize(credentials, project=PROJECT_ID)
    logger.info("Earth Engine initialized successfully!")
except Exception as e:
    logger.error(f"EE initialization failed: {e}")


# ---------------------------
#  PINECONE & GROQ SETUP (RAG CHATBOT)
# ---------------------------
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "agri-knowledge-base")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# Initialize chatbot components (only if API keys are provided)
chatbot_chain = None
if PINECONE_API_KEY and GROQ_API_KEY:
    try:
        logger.info("Initializing Pinecone and Groq for RAG chatbot...")
        
        # Initialize embeddings using VoyageAI (Cloud API - Lightweight)
        from langchain_voyageai import VoyageAIEmbeddings
        embeddings = VoyageAIEmbeddings(
            voyage_api_key=os.getenv("VOYAGE_API_KEY"),
            model="voyage-large-2"
        )
        
        # Initialize Pinecone vector store
        vectorstore = PineconeVectorStore(
            index_name=PINECONE_INDEX_NAME,
            embedding=embeddings,
            pinecone_api_key=PINECONE_API_KEY
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        
        # Initialize Groq LLM
        llm = ChatGroq(
            groq_api_key=GROQ_API_KEY,
            model_name=GROQ_MODEL,
            temperature=0.7
        )
        
        # Create conversational retrieval chain
        chatbot_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            return_source_documents=True,
            verbose=False
        )
            
        logger.info("✓ Pinecone and Groq chatbot initialized successfully!")
    except Exception as e:
        logger.warning(f"Chatbot initialization failed: {e}")
        logger.warning("Chatbot endpoint will not be available")
else:
    logger.warning("PINECONE_API_KEY or GROQ_API_KEY not found - chatbot disabled")


# ---------------------------
#  ROOT ROUTE
# ---------------------------
@app.get("/")
def root():
    return {"status": "GEE Backend Running on Railway!"}


# ---------------------------
#  TEST ROUTE
# ---------------------------
@app.get("/test")
def test():
    return {"message": "API is working!"}


# ---------------------------
#  CHATBOT ENDPOINT (RAG)
# ---------------------------
class ChatRequest(BaseModel):
    user_id: str
    question: str

@app.post("/ask-chatbot")
async def ask_chatbot(req: ChatRequest):
    """
    Agricultural chatbot endpoint using Pinecone + Groq RAG system.
    """
    if not chatbot_chain:
        return {
            "answer": "Chatbot is not available. Please configure PINECONE_API_KEY and GROQ_API_KEY in .env file."
        }
    
    try:
        logger.info(f"Processing chatbot question: {req.question}")
        
        # Use the conversational retrieval chain
        result = chatbot_chain.invoke({
            "question": req.question,
            "chat_history": []  # Simple implementation without chat history
        })
        
        answer = result.get("answer", "I couldn't find a relevant answer.")
        logger.info(f"Chatbot response generated successfully")
        
        return {"answer": answer}
        
    except Exception as e:
        logger.error(f"Error in /ask-chatbot: {str(e)}")
        return {
            "answer": "I'm sorry, I encountered an error processing your question. Please try again."
        }



# ---------------------------
#  SATELLITE ANALYSIS ROUTE
# ---------------------------
@app.get("/get_field_health")
def get_field_health(lat: float, lon: float):
    try:
        user_point = ee.Geometry.Point(lon, lat)
        field_aoi = user_point.buffer(50).bounds()

        end_date = ee.Date(int(time.time() * 1000))
        start_date = end_date.advance(-90, 'day')

        # Sentinel-2
        s2 = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(field_aoi)
            .filterDate(start_date, end_date)
            .sort("CLOUDY_PIXEL_PERCENTAGE")
            .first()
        )

        # Sentinel-1 (Radar)
        s1 = (
            ee.ImageCollection("COPERNICUS/S1_GRD")
            .filterBounds(field_aoi)
            .filterDate(start_date, end_date)
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
            .sort("system:time_start", False)
            .first()
        )

        # Landsat (Temperature)
        landsat = (
            ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
            .merge(ee.ImageCollection("LANDSAT/LC09/C02/T1_L2"))
            .filterBounds(field_aoi)
            .filterDate(start_date, end_date)
            .sort("CLOUD_COVER")
            .first()
        )

        # Soil Organic Carbon
        soc_image = ee.Image("projects/soilgrids-isric/soc_mean")

        response = {
            "health_status": "Analyzing...",
            "avg_temp_celsius": "N/A",
            "soil_organic_carbon": "N/A",
            "ndvi_map_url": None,
            "ndwi_map_url": None,
            "soil_moisture_map_url": None,
            "lst_map_url": None,
            "field_boundary": [
                [lat - 0.0005, lon - 0.0005],
                [lat + 0.0005, lon - 0.0005],
                [lat + 0.0005, lon + 0.0005],
                [lat - 0.0005, lon + 0.0005]
            ]
        }

        results = {}

        # NDVI + NDWI
        if s2:
            ndvi = s2.normalizedDifference(["B8", "B4"]).rename("NDVI")
            results["avg_ndvi"] = ndvi.reduceRegion(
                ee.Reducer.mean(), field_aoi, 10
            ).get("NDVI")

            response["ndvi_map_url"] = ndvi.getMapId({
                "min": -0.2, "max": 0.8,
                "palette": ["red", "yellow", "green"]
            })["tile_fetcher"].url_format

            ndwi = s2.normalizedDifference(["B3", "B8"]).rename("NDWI")
            response["ndwi_map_url"] = ndwi.getMapId({
                "min": -0.6, "max": 0.6,
                "palette": ["brown", "yellow", "cyan", "blue"]
            })["tile_fetcher"].url_format

        # Radar Soil Moisture
        if s1:
            vv = s1.select("VV")
            response["soil_moisture_map_url"] = vv.getMapId({
                "min": -25, "max": 0,
                "palette": ["red", "orange", "yellow", "cyan", "blue"]
            })["tile_fetcher"].url_format

        # LST (Surface Temperature)
        if landsat:
            lst_kelvin = landsat.select("ST_B10").multiply(0.00341802).add(149.0)
            lst_celsius = lst_kelvin.subtract(273.15)

            results["avg_lst"] = lst_celsius.reduceRegion(
                ee.Reducer.mean(), field_aoi, 30
            ).get("ST_B10")

            response["lst_map_url"] = lst_celsius.getMapId({
                "min": 15, "max": 45,
                "palette": ["blue", "cyan", "green", "yellow", "orange", "red"]
            })["tile_fetcher"].url_format

        # Soil Organic Carbon
        soc_val = soc_image.select("soc_0-5cm_mean").reduceRegion(
            ee.Reducer.mean(), field_aoi, 250
        ).get("soc_0-5cm_mean")

        results["soc"] = soc_val

        # Convert dictionary
        info = ee.Dictionary(results).getInfo()

        # HEALTH STATUS
        ndvi_val = info.get("avg_ndvi")
        if ndvi_val is not None:
            if ndvi_val >= 0.6:
                response["health_status"] = "Very Healthy"
            elif ndvi_val >= 0.4:
                response["health_status"] = "Healthy"
            elif ndvi_val >= 0.2:
                response["health_status"] = "Stressed"
            else:
                response["health_status"] = "Very Stressed / Bare Soil"

        # Temperature
        lst_val = info.get("avg_lst")
        if lst_val is not None:
            response["avg_temp_celsius"] = round(float(lst_val), 1)

        # Soil Carbon
        soc_val = info.get("soc")
        if soc_val is not None:
            response["soil_organic_carbon"] = f"{round(soc_val/10.0, 2)}%"

        return response

    except Exception as e:
        return {"error": f"GEE Processing Error: {str(e)}"}
