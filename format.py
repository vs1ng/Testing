import pandas as pd
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import time
import csv

def geocode_districts(input_csv="missing_districts.csv", output_csv="missing_districts_with_coords.csv"):
    """
    Geocode districts to get latitude and longitude coordinates
    """
    
    # Read the CSV
    df = pd.read_csv(input_csv)
    
    # Initialize geocoder
    geolocator = Nominatim(user_agent="warehouse_locator_v1")
    
    # Add rate limiter to respect Nominatim's terms of service
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
    
    print(f"Geocoding {len(df)} districts...")
    print("This may take several minutes due to rate limiting...")
    
    latitudes = []
    longitudes = []
    success_count = 0
    
    for idx, row in df.iterrows():
        district = row['District']
        state = row['State']
        
        # Create query - try different formats for better results
        queries = [
            f"{district}, {state}, India",
            f"{district} district, {state}, India",
            f"{district}, India"
        ]
        
        location = None
        for query in queries:
            try:
                location = geocode(query)
                if location:
                    break
            except Exception as e:
                time.sleep(1)
                continue
        
        if location:
            latitudes.append(location.latitude)
            longitudes.append(location.longitude)
            success_count += 1
            print(f"✓ {district}, {state}: {location.latitude}, {location.longitude}")
        else:
            latitudes.append(None)
            longitudes.append(None)
            print(f"✗ {district}, {state}: Not found")
        
        # Progress indicator
        if (idx + 1) % 10 == 0:
            print(f"Progress: {idx + 1}/{len(df)} | Success: {success_count}")
    
    # Add coordinates to dataframe
    df['lat'] = latitudes
    df['long'] = longitudes
    
    # Save to new CSV
    df.to_csv(output_csv, index=False)
    
    print(f"\nGeocoding complete!")
    print(f"Successfully geocoded: {success_count}/{len(df)} districts")
    print(f"Saved to: {output_csv}")
    
    # Show some statistics
    missing = df[df['lat'].isna()]
    if len(missing) > 0:
        print(f"\nDistricts that couldn't be geocoded:")
        for _, row in missing.iterrows():
            print(f"  - {row['WarehouseName']} ({row['District']}, {row['State']})")
    
    return df

def batch_geocode_with_retry(input_csv="missing_districts.csv", 
                           output_csv="missing_districts_with_coords.csv",
                           batch_size=50,
                           max_retries=3):
    """
    Alternative: Batch geocoding with retry logic
    """
    
    df = pd.read_csv(input_csv)
    
    # You can use different geocoding services in batch mode
    # Option 1: Using OSM's bulk geocoding (requires setup)
    # Option 2: Using Google Maps API (more accurate, needs API key)
    
    print("For batch processing with better accuracy, consider:")
    print("1. Google Maps Geocoding API (requires API key)")
    print("2. OpenCage Geocoder (free tier available)")
    print("3. Mapbox Geocoding API (free tier available)")
    
    # Placeholder - you'd implement your chosen service here
    return None

def geocode_with_google(api_key, df, output_csv):
    """
    Using Google Maps API (most accurate)
    Requires: pip install googlemaps
    """
    import googlemaps
    
    gmaps = googlemaps.Client(key=api_key)
    
    latitudes = []
    longitudes = []
    
    for idx, row in df.iterrows():
        query = f"{row['District']}, {row['State']}, India"
        
        try:
            geocode_result = gmaps.geocode(query)
            if geocode_result:
                lat = geocode_result[0]['geometry']['location']['lat']
                lng = geocode_result[0]['geometry']['location']['lng']
                latitudes.append(lat)
                longitudes.append(lng)
                print(f"✓ {row['District']}: {lat}, {lng}")
            else:
                latitudes.append(None)
                longitudes.append(None)
                print(f"✗ {row['District']}: Not found")
        except Exception as e:
            print(f"Error for {row['District']}: {e}")
            latitudes.append(None)
            longitudes.append(None)
    
    df['lat'] = latitudes
    df['long'] = longitudes
    df.to_csv(output_csv, index=False)
    return df

# Quick alternative using a free API service
def geocode_with_opencage(api_key, df, output_csv):
    """
    Using OpenCage Geocoder (2500 free requests/day)
    Requires: pip install opencage
    """
    from opencage.geocoder import OpenCageGeocode
    
    geocoder = OpenCageGeocode(api_key)
    
    latitudes = []
    longitudes = []
    
    for idx, row in df.iterrows():
        query = f"{row['District']}, {row['State']}, India"
        
        try:
            results = geocoder.geocode(query)
            if results and len(results) > 0:
                lat = results[0]['geometry']['lat']
                lng = results[0]['geometry']['lng']
                latitudes.append(lat)
                longitudes.append(lng)
            else:
                latitudes.append(None)
                longitudes.append(None)
        except:
            latitudes.append(None)
            longitudes.append(None)
    
    df['lat'] = latitudes
    df['long'] = longitudes
    df.to_csv(output_csv, index=False)
    return df

# Main execution
if __name__ == "__main__":
    # Option 1: Free geocoding (slower, less accurate)
    print("Option 1: Using Nominatim (Free, no API key needed)")
    result = geocode_districts("missing_districts.csv", "geocoded_districts.csv")
    
    # Option 2: For better accuracy, use an API service
    print("\n\nFor better results, consider these services:")
    print("""
    1. Google Maps Geocoding API - Most accurate
       - Cost: $5 per 1000 requests
       - Setup: https://developers.google.com/maps/documentation/geocoding
    
    2. OpenCage Geocoder - Good alternative
       - Free: 2500 requests/day
       - Setup: https://opencagedata.com/api
    
    3. Mapbox Geocoding - Good for global coverage
       - Free: 100,000 requests/month
       - Setup: https://docs.mapbox.com/api/search/geocoding/
    
    Quick implementation with Google Maps:
    ```python
    import googlemaps
    gmaps = googlemaps.Client(key='YOUR_API_KEY')
    geocode_with_google('YOUR_API_KEY', df, 'geocoded_districts.csv')
    ```
    """)
