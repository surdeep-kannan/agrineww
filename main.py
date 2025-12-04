from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import ee
import time
app = FastAPI(title="AgriXVision Backend (GEE + LLAMA 3.2)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],           
    allow_credentials=True,
    allow_methods=["*"],           
    allow_headers=["*"],
)
class ChatRequest(BaseModel):
    question: str
@app.post("/ask-chatbot")
async def ask_chatbot(req: ChatRequest):
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3.2:1b",
                "prompt": req.question,
                "stream": False
            },
            timeout=60
        )
        data = response.json()
        return {"answer": data.get("response", "No response from model")}
    except Exception as e:
        return {"error": f"Chatbot error: {str(e)}"}
PROJECT_ID = "premium-origin-469307-t0"  
try:
    ee.Initialize(project=PROJECT_ID)
    print(f"GEE initialized with project {PROJECT_ID}")
except Exception as e:
    print("GEE not authenticated yet. Running ee.Authenticate()...")
    ee.Authenticate()
    ee.Initialize(project=PROJECT_ID)
    print("GEE authenticated and initialized!")
@app.get("/get_field_health")
async def get_field_health(lat: float, lon: float):
    try:
        user_point = ee.Geometry.Point(lon, lat)
        field_aoi = user_point.buffer(50).bounds()
        end_date = ee.Date(int(time.time() * 1000))
        start_date = end_date.advance(-90, 'day')
        s2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
              .filterBounds(field_aoi)
              .filterDate(start_date, end_date)
              .sort('CLOUDY_PIXEL_PERCENTAGE')
              .first())
        s1 = (ee.ImageCollection('COPERNICUS/S1_GRD')
              .filterBounds(field_aoi)
              .filterDate(start_date, end_date)
              .filter(ee.Filter.eq('instrumentMode', 'IW'))
              .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
              .sort('system:time_start', False)
              .first())
        landsat = (ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
                   .merge(ee.ImageCollection('LANDSAT/LC09/C02/T1_L2'))
                   .filterBounds(field_aoi)
                   .filterDate(start_date, end_date)
                   .sort('CLOUD_COVER')
                   .first())
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
        if s2:
            ndvi = s2.normalizedDifference(['B8', 'B4']).rename('NDVI')
            results['avg_ndvi'] = ndvi.reduceRegion(ee.Reducer.mean(), field_aoi, 10).get('NDVI')
            response["ndvi_map_url"] = ndvi.getMapId({
                'min': -0.2, 'max': 0.8,
                'palette': ['red', 'yellow', 'green']
            })['tile_fetcher'].url_format
            ndwi = s2.normalizedDifference(['B3', 'B8']).rename('NDWI')
            response["ndwi_map_url"] = ndwi.getMapId({
                'min': -0.6, 'max': 0.6,
                'palette': ['brown', 'yellow', 'cyan', 'blue']
            })['tile_fetcher'].url_format
        if s1:
            vv = s1.select('VV')
            response["soil_moisture_map_url"] = vv.getMapId({
                'min': -25, 'max': 0,
                'palette': ['red', 'orange', 'yellow', 'cyan', 'blue']
            })['tile_fetcher'].url_format
        if landsat:
            lst_kelvin = landsat.select('ST_B10').multiply(0.00341802).add(149.0)
            lst_celsius = lst_kelvin.subtract(273.15)
            results['avg_lst'] = lst_celsius.reduceRegion(ee.Reducer.mean(), field_aoi, 30).get('ST_B10')
            response["lst_map_url"] = lst_celsius.getMapId({
                'min': 15, 'max': 45,
                'palette': ['blue', 'cyan', 'green', 'yellow', 'orange', 'red']
            })['tile_fetcher'].url_format
        soc = soc_image.select('soc_0-5cm_mean').reduceRegion(ee.Reducer.mean(), field_aoi, 250)
        results['soc'] = soc.get('soc_0-5cm_mean')
        info = ee.Dictionary(results).getInfo()
        ndvi_val = info.get('avg_ndvi')
        if ndvi_val is not None:
            if ndvi_val >= 0.6:
                response["health_status"] = "Very Healthy"
            elif ndvi_val >= 0.4:
                response["health_status"] = "Healthy"
            elif ndvi_val >= 0.2:
                response["health_status"] = "Stressed"
            else:
                response["health_status"] = "Very Stressed / Bare Soil"
        lst_val = info.get('avg_lst')
        if lst_val is not None:
            response["avg_temp_celsius"] = round(float(lst_val), 1)
        soc_val = info.get('soc')
        if soc_val is not None:
            response["soil_organic_carbon"] = f"{round(soc_val / 10.0, 2)}%"
        return response
    except Exception as e:
        return {"error": f"GEE Processing Error: {str(e)}"}
@app.get("/")
def root():
    return {"status": "AgriXVision Backend Running! GEE + LLAMA 3.2 Ready"}