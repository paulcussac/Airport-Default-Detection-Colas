# Colas - Deep Learning

## STEP 1 - Collect French airports open data

First, you should download the 19 regions and departments cartographic data, available [here](https://geoservices.ign.fr/bdcarto).
 

Unzip them. And group them into a folder called `BDD_CARTO`.

From the root directory, you should be able to execute the python script `data_processing.py` with the command `python airports/data_processing.py`.
Note that you need to fill in the path to your `BDD_CARTO` folder in the `config.yml`.

You will find the output in `airports/data/airports.geojson` containing :
- ID : id reference, could be useful to join `PISTE_AERODROME` objects
- TOPONYME : name of the airport
- geometry : geometry of the airport

## STEP 2 - Collect French orthophotos centered on airports

From the `orthophotos` directory, you can execute the python script `main.py` with the command `python orthophotos/main.py`.

You will find the output in `orthophotos/image_folder` containing all png images of the BD ORTHO centered on the airports defined in STEP 1 with optimal resolution (1 pixel / 20 cm).

## STEP 3 - Train a Computer Vision model to detect defections on runways

In the `model` directory, you may find the python script `main.py` that will train a Hugging Face Vision Transformer (ViT) model. Please find further documentation [here](https://huggingface.co/docs/transformers/model_doc/vit). 

This is Multi-Label Task as one can find various types of defections (cracks, potholes, ...) that have been labeled. 
