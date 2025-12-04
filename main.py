from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import ee
import json
import os

# ---------------------------
#  CREATE FASTAPI APP
# ---------------------------
app = FastAPI(title="AgriXVision Backend (GEE)")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
#  EARTH ENGINE AUTH (Railway)
# ---------------------------
PROJECT_ID = "premium-origin-469307-t0"

service_json = os.getenv("SERVICE_ACCOUNT_JSON")
if not service_json:
    raise Exception("SERVICE_ACCOUNT_JSON env variable missing!")

service_account_info = json.loads(service_json)

credentials = ee.ServiceAccountCredentials(
    email=service_account_info["client_email"],
    key_data=service_json
)

try:
    ee.Initialize(credentials, project=PROJECT_ID)
    print("Earth Engine initialized on Railway!")
except Exception as e:
    print("Failed EE init:", e)

# ---------------------------
#   BASIC TEST ROUTE
# ---------------------------
@app.get("/")
def root():
    return {"status": "GEE Backend Running on Railway!"}

# ---------------------------
#   EXAMPLE ROUTE (TEST)
# ---------------------------
@app.get("/test")
def test():
    return {"message": "API is working!"}
