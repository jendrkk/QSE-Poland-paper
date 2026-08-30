import requests
import geopandas as gpd
from pathlib import Path
from io import BytesIO

REPO_ROOT = Path(__file__).parent.parent.parent

url = (
    "https://services.arcgis.com/8df8p0NlLFEShl0r/"
    "arcgis/rest/services/Historic_National_Boundaries_NEW/"
    "FeatureServer/6/query"
)

params = {
    "where": "1=1",
    "outFields": "*",
    "returnGeometry": "true",
    "f": "geojson",
}

response = requests.get(url, params=params)
response.raise_for_status()

gdf = gpd.read_file(BytesIO(response.content))

# Filter only the partitions of interest: Prussian (P), Austrian (A), Russian (R)
countries_of_interest = {'P': 196, 'A': 118, 'R': 197}
oid_to_partition = {oid: code for code, oid in countries_of_interest.items()}
gdf = gdf[gdf['OBJECTID_12'].isin(countries_of_interest.values())].copy()

# Load the communes map
COMMUNES_PATH = REPO_ROOT / 'data' / 'processed' / 'shapefiles' / 'communes_2021.gpkg'
communes_gdf = gpd.read_file(COMMUNES_PATH, layer='communes')
communes_gdf['partition'] = ''

# Historic boundaries arrive as GeoJSON (typically WGS84); align to commune CRS
# before area-based intersection assignment.
gdf = gdf.to_crs(communes_gdf.crs)

# For each commune, find the country that it intersects with
# If it intersects with multiple countries, then assign the commune to the country that it intersects with the most (percentage of the commune's area)
for index, row in communes_gdf.iterrows():
    max_intersection = 0
    max_intersection_country = ''
    geom = row.geometry
    for _, row_gdf in gdf.iterrows():
        if row_gdf['geometry'].intersects(geom):
            intersection_area = row_gdf['geometry'].intersection(geom).area
            if intersection_area > max_intersection:
                max_intersection = intersection_area
                max_intersection_country = oid_to_partition[row_gdf['OBJECTID_12']]
    communes_gdf.at[index, 'partition'] = max_intersection_country

# Save the communes map with the partitions
communes_gdf.to_file(REPO_ROOT / 'data' / 'processed' / 'shapefiles' / 'communes_2021_partitions.gpkg', driver='GPKG')