#!/usr/bin/env bash

# This must be run from the glacier mapping root directory
source .env

# if data or directories don't exist, you'll need to fill them in
# mkdir -p $WEB_DIR/processed/preds/
# mkdir -p $DATA_DIR/web/basemap/
export WEB_DIR=$DATA_DIR/web_data/

# data prep for backend
python3 -m web.backend.backend_data -d $DATA_DIR/unique_tiles/ -o $DATA_DIR/web/basemap2/ -n output-full.vrt;
python3 -m web.backend.backend_data -d $DATA_DIR/unique_tiles/ -o $DATA_DIR/web/basemap2/ -n output-245.vrt --bandList 5 4 2

cd $DATA_DIR/web/basemap/
gdal_translate -ot Byte output-245.vrt output-245-byte.vrt

for i in $( seq 8 14 )
do
    gdal2tiles.py -z $i --processes 25 output-245-byte.vrt .
done;

# copy tile outputs to $ROOT_DIR/web/outputs/
# mkdir -p $ROOT_DIR/web/outputs/tiles/
# you will see results at http://0.0.0.0:4040/web/frontend/index.html
python3 -m web.frontend_server & python3 -m web.backend.server
