export default {
  "classes": [
    {
      "count": 0,
      "color": "#0000FF",
      "name": "Background"
    },
    {
      "count": 0,
      "color": "#008000",
      "name": "Ice"
    },
    {
      "count": 0,
      "color": "#80FF80",
      "name": "Debris"
    }
  ],
  "shapeLayers": [
    {
      "name": "Area Boundaries",
      "zoneNameId": "HKH",
      "shapesFn": "shapes/hkh.geojson"
    }
  ],
  "basemapLayer": {
    "bounds": null,
    "args": {
      "minZoom": 8,
      "maxZoom": 14,
      "maxNativeZoom": 10,
      "attribution": "Georeferenced Image"
    },
    "initialLocation": [
      30.4290779,
      82.8789515
    ],
    "url": "outputs/basemap/{z}/{x}/{-y}.png",
    "initialZoom": 11
  },
  "predictionLayer": {
    "bounds": null,
    "args": {
      "minZoom": 10,
      "maxZoom": 10,
      "maxNativeZoom": 10,
      "attribution": "Georeferenced Image"
    },
    "initialLocation": [
      28.1517627,
      93.1757374
    ],
    "url": "outputs/pred_tiles/{z}/{x}/{-y}.png",
    "initialZoom": 10
  },
  "dataLayer": {
    "padding": 1100,
    "path": "/datadrive/glaciers/web/basemap/output-full.vrt",
    "type": "GLACIER"
  },
  "metadata": {
    "id": "hkh2000_landsat",
    "locationName": null,
    "imageryName": "Landsat Imagery",
    "displayName": "Hindu Kush Himalaya 2000"
  }
}
