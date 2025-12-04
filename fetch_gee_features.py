# fetch_gee_features.py

import pandas as pd
import ee
import time

# Initialize the Earth Engine API (run once)
ee.Initialize()

def get_ndvi_and_rainfall(lat, lon, start_date, end_date):
    """
    Fetch mean NDVI and total rainfall for a small AOI around lat/lon.
    Returns (ndvi, rainfall) or (None, None) on error.
    """
    try:
        aoi = ee.Geometry.Rectangle([lon - 0.05, lat - 0.05, lon + 0.05, lat + 0.05])

        # Sentinel-2 SR Harmonized collection
        sentinel_collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                               .filterBounds(aoi)
                               .filterDate(start_date, end_date)
                               .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)))

        # NDVI computation
        def add_ndvi(image):
            ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
            return image.addBands(ndvi)

        with_ndvi = sentinel_collection.map(add_ndvi)
        mean_ndvi = with_ndvi.select('NDVI').mean().reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=aoi,
            scale=30,
            bestEffort=True).get('NDVI')

        # CHIRPS rainfall collection
        rainfall_collection = (ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')
                               .filterBounds(aoi)
                               .filterDate(start_date, end_date))

        total_rainfall = rainfall_collection.sum().reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=aoi,
            scale=5566,
            bestEffort=True).get('precipitation')

        ndvi_value = mean_ndvi.getInfo()
        rainfall_value = total_rainfall.getInfo()

        # Return None if data missing
        if ndvi_value is None or rainfall_value is None:
            return None, None

        return ndvi_value, rainfall_value

    except Exception as e:
        print(f"Error fetching GEE data for {lat}, {lon}: {e}")
        return None, None


# --- Main Execution ---

# Load your cleaned data CSV
df = pd.read_csv('cleaned_production_data.csv')

# Example: Add lat/lon columns to your DataFrame.
# Replace this with actual district centroids.
# For demo, use fixed lat/lon (replace with your mapping logic)
df['lat'] = 25.0
df['lon'] = 80.0

print(f"Fetching GEE data for {len(df)} records...")

ndvi_list = []
rainfall_list = []

# Define crop growing season
start_date = '2023-06-01'
end_date = '2023-10-31'

for idx, row in df.iterrows():
    lat = row['lat']
    lon = row['lon']

    ndvi_val, rain_val = get_ndvi_and_rainfall(lat, lon, start_date, end_date)
    ndvi_list.append(ndvi_val)
    rainfall_list.append(rain_val)

    # Rate limiting to avoid GEE quota issues
    time.sleep(1)

df['mean_ndvi'] = ndvi_list
df['total_rainfall_mm'] = rainfall_list

# Save the new enriched dataset
output_file = 'cleaned_with_gee_features.csv'
df.to_csv(output_file, index=False)
print(f"Saved enriched data with GEE features to '{output_file}'")
