#!/usr/bin/env python3
import shapely
import shapely.geometry
import web.backend.DataLoader as DL
import fiona
import fiona.transform
import json
import os
import utm

_DATASET_FN = "conf/dataset.json"
DATA_DIR = os.environ["DATA_DIR"]

def get_area_from_geometry(geom, src_crs="epsg:4326"):
    if geom["type"] == "Polygon":
        lon, lat = geom["coordinates"][0][0]
    elif geom["type"] == "MultiPolygon":
        lon, lat = geom["coordinates"][0][0][0]
    else:
        raise ValueError("Polygons and MultiPolygons only")

    zone_number = utm.latlon_to_zone_number(lat, lon)
    hemisphere = "+north" if lat > 0 else "+south"
    dest_crs = "+proj=utm +zone=%d %s +datum=WGS84 +units=m +no_defs" % (zone_number, hemisphere)
    projected_geom = fiona.transform.transform_geom(src_crs, dest_crs, geom)
    area = shapely.geometry.shape(projected_geom).area / 1000000.0 # we calculate the area in square meters then convert to square kilometers
    return area


def _load_dataset(dataset):
    print(dataset)
    data_dir = pathlib.Path(os.environ["DATA_DIR"])
    data_loader = DL.DataLoaderGlacier(data_dir / dataset["dataLayer"]["padding"], dataset["dataLayer"]["path"])
    print(data_loader)
    return {"data_loader": data_loader}

def load_dataset():
    dataset_json = json.load(open(os.path.join(os.environ["ROOT_DIR"], _DATASET_FN),"r"))
    return _load_dataset(dataset_json)
