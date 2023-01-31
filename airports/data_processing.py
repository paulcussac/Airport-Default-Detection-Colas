import pandas as pd
import geopandas as gpd
import os
import yaml


# load yaml config file
with open("config.yml") as f:
    config = yaml.safe_load(f)

os.chdir(config["bdd_carto_filepath"])
data = gpd.GeoDataFrame()

for folder in os.listdir():
    if folder == ".DS_Store":
        continue
    
    print("#"*80)
    print("Entering folder {}".format(folder))
    # enter subfolder BDCARTO
    os.chdir(folder + "/BDCARTO")
    
    # enter subfolder 1_DONNEES_LIVRAISONS
    livraison_folder_name = [name for name in os.listdir() if name.startswith("1_DONNEES")][0]
    os.chdir(livraison_folder_name)
    
    # enter subfolder BDC_4-0, not .md5 file
    data_folder_name = [name for name in os.listdir() if not name.endswith(".md5") and name.startswith("BDC")][0]
    os.chdir(data_folder_name)
    
    # enter subfolder EQUIPEMENT
    os.chdir("EQUIPEMENT")
    
    new_aerodrome = gpd.read_file("AERODROME.SHP").to_crs(4326)
    print("Adding {} new aerodrome instances".format(len(new_aerodrome)))
    data = pd.concat([data, new_aerodrome])

    # back to main folder
    os.chdir(config["bdd_carto_filepath"])

# Write GeoDataFrame to geojson
print("#"*80)
print("Writing to file")
os.chdir(config["ROOT_DIR_PATH"])
# Check if there is a folder named data inside airports folder. If not, create one. 
if os.path.exists(config["ROOT_DIR_PATH"] + "/airports/data") == True:
    print('data folder already exists')
else: 
    os.mkdir(config["ROOT_DIR_PATH"] + "/airports/data")
    print('data folder is created')
    
data[["ID", "TOPONYME", "geometry"]].to_file("airports/data/airports.geojson", driver='GeoJSON')
