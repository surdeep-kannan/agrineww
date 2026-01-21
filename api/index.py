from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import ee
import json
import os
import time
from dotenv import load_dotenv
import logging

# ---------------------------
#  CHATBOT ENDPOINT (Direct Groq)
# ---------------------------
from groq import Groq

# Initialize Groq Client
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = None
if GROQ_API_KEY:
    try:
        groq_client = Groq(api_key=GROQ_API_KEY)
        logger.info("✓ Groq client initialized successfully!")
    except Exception as e:
        logger.warning(f"Groq initialization failed: {e}")

class ChatRequest(BaseModel):
    user_id: str
    question: str

@app.post("/ask-chatbot")
async def ask_chatbot(req: ChatRequest):
    """
    Direct Agricultural chatbot using Groq API (Llama-3).
    """
    if not groq_client:
        return {
            "answer": "Chatbot is not available. Please configure GROQ_API_KEY in .env file."
        }
    
    try:
        logger.info(f"Processing chatbot question: {req.question}")
        
        system_prompt = (
            "You are an expert agricultural AI assistant named AgriXVision. "
            "Your goal is to help farmers with crop health, soil, and weather advice. "
            "Be helpful, concise, and accurate."
        )

        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": req.question}
            ],
            model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            temperature=0.7,
            max_tokens=500,
        )

        answer = chat_completion.choices[0].message.content
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
