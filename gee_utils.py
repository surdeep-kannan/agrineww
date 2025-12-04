# backend/gee_utils.py
# A utility file to handle Google Earth Engine tasks.

import ee
import requests
import base64   

def get_sentinel_data(lat, lon):
    """
    Initializes GEE, fetches a Sentinel-2 satellite image,
    and returns it as a Base64 encoded data URI.
    """
    try:
        # Initialize the Earth Engine library.
        # It automatically uses the project authenticated via the command line.
        ee.Initialize()
        print("GEE Initialized Successfully for this request.")
    except Exception as e:
        print(f"Error initializing GEE: {e}")
        # Return a placeholder or raise an exception if initialization fails
        return "https://placehold.co/512x512/000000/FFFFFF?text=GEE+Error"

    # Define the Area of Interest (AOI) using the provided coordinates.
    aoi = ee.Geometry.Point([lon, lat])

    # Find the least cloudy Sentinel-2 image for the location and a recent date range.
    # Using the non-deprecated 'S2_SR_HARMONIZED' collection.
    collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterBounds(aoi) \
        .filterDate('2024-01-01', '2024-05-31') \
        .sort('CLOUDY_PIXEL_PERCENTAGE')

    # Get the first (least cloudy) image from the collection.
    first_image = collection.first()

    # Define visualization parameters to create a natural-color image.
    # We select the Red (B4), Green (B3), and Blue (B2) bands.
    visualization_params = {
        'bands': ['B4', 'B3', 'B2'],
        'min': 0.0,
        'max': 3000.0, # These values help with image contrast
        'gamma': 1.4,
    }

    # Generate the temporary thumbnail URL using the visualization parameters.
    temp_url = first_image.getThumbURL({
        **visualization_params,
        'region': aoi.buffer(5000).bounds().getInfo()['coordinates'],
        'dimensions': 512,
        'format': 'png'
    })
    
    # --- NEW: Fetch the image data immediately from the temporary URL ---
    try:
        print("Fetching image data from temporary GEE URL...")
        image_response = requests.get(temp_url)
        image_response.raise_for_status() # Raise an exception for bad status codes (like 404)
        
        # Encode the binary image content into a Base64 string
        base64_image = base64.b64encode(image_response.content).decode('utf-8')
        
        # Create a data URI, which can be used directly in an HTML <img> tag
        data_uri = f"data:image/png;base64,{base64_image}"
        
        print("Successfully converted image to Base64 data URI.")
        return data_uri

    except requests.exceptions.RequestException as e:
        print(f"Error fetching image from GEE URL: {e}")
        return "https://placehold.co/512x512/ff0000/FFFFFF?text=Fetch+Error"
