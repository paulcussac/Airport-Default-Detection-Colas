import requests as r
import pandas as pd
import json
from PIL import Image
import geopandas as gpd
from shapely.geometry import Point
import pyproj


def clean_coordinates(data):
    """
    Input : Dataframe with extracted data from French airports database passed through data_preprocessing.py
        
    This function cleans the coordinates in order to remove useless characters.
    """
    data['geometry'] = data['geometry'].astype(str)
    data['coordinates_clean'] = data.apply(lambda x: x.geometry.replace('POLYGON','').replace('MULTI','').strip().split(','), axis=1)
        
    return data


def get_all_coor(list_coor):
    """
    Input : List of polygon coordinates.
    
    This function returns a tuple of lists containing respectively all latitudes values and all longitudes values for a given polygon.
    """
    
    all_lat = []
    all_lon = []
    
    for string in list_coor:
        
        string_clean = string.strip()
        
        lat = float(string_clean.split()[1].replace(')','').replace('(',''))
        lon = float(string_clean.split()[0].replace(')','').replace('(',''))
        
        all_lat.append(lat)
        all_lon.append(lon)
        
    return (all_lat, all_lon)


def construct_bbox(data): 
    """
    Input  : Dataframe 
    
    Prerequisites : Dataframe has to be passed through get_all_coor function.
    
    This function construct in a new column the rectangle bbox that defines an airport geogriphical area.
    """
    
    data['all_coor'] = data.apply(lambda x: get_all_coor(x.coordinates_clean), axis=1)

    data['max_lat']  = data.apply(lambda x: str(max(x.all_coor[0])) ,axis=1)
    data['min_lat']  = data.apply(lambda x: str(min(x.all_coor[0])),axis=1)
    data['max_lon']  = data.apply(lambda x: str(max(x.all_coor[1])),axis=1)
    data['min_lon']  = data.apply(lambda x: str(min(x.all_coor[1])),axis=1)

    data['bbox']     = data.apply(lambda x: x.min_lat + ',' + x.min_lon + ',' + x.max_lat + ',' + x.max_lon, axis=1)
    
    return data 


def get_width_height(data):
    """
    Input : Dataframe 
    
    Prerequisites : Dataframe has to be passed through construct_bbox function.
    
    This function constructs 2 new columns providing expected width and height for the airport image to keep its original shape 
    and to have a resolution of 20cm per pixel.
    """
    geod = pyproj.Geod(ellps='WGS84')

    
    data['up_left']  = data.apply(lambda row: Point(float(row['max_lat']), float(row['min_lon'])), axis=1)
    data['up_right'] = data.apply(lambda row: Point(float(row['max_lat']), float(row['max_lon'])), axis=1)
    data['bot_left'] = data.apply(lambda row: Point(float(row['min_lat']), float(row['min_lon'])), axis=1)

    geod = pyproj.Geod(ellps='WGS84')

    data['width']  = data.apply(lambda row: geod.inv(  row['up_left'].x, row['up_left'].y, 
                                                       row['up_right'].x, row['up_right'].y)[2], axis=1)

    data['height'] = data.apply(lambda row: geod.inv(  row['up_left'].x, row['up_left'].y, 
                                                       row['bot_left'].x, row['bot_left'].y)[2], axis=1)
    
    return data


def get_url(data):
    """
    Input : Dataframe
    
    Prerequisites : Dataframe has to be passed through get_width_height function.
    
    This function builds the url request to download the image in a new column. 
    """

    url_start  = "https://wxs.ign.fr/ortho/geoportail/r/wms?LAYERS=HR.ORTHOIMAGERY.ORTHOPHOTOS&EXCEPTIONS=text/xml&" +\
                 "FORMAT=image/png&SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap&STYLES=&CRS=EPSG:4326&BBOX=" 
    
    url_width  = " &WIDTH="
    url_height = "&HEIGHT="

    data['url_path'] = data.apply(lambda x:   url_start + x.bbox + 
                                              url_width + str(int(x.width * 5)) + 
                                              url_height +str(int(x.height * 5)), axis=1)
    return data
