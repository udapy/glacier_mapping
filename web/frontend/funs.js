import { state, tiles, map, backendUrl } from './globals.js';
import dataset from '../../conf/dataset.js';
import models from '../../conf/models.js';

let sideBySide = null;

export function initializeMap() {
  map.pm.addControls({
    deleteLayer: true,
    drawCircle: false,
    drawCircleMarker: false,
    drawMarker: false,
    drawPolygon: true,
    drawPolyline: false,
    drawRectangle: false,
    editPolygon: true
  });

  // add svg overlay
  L.svg({clickable:true}).addTo(map);
  const overlay = d3.select(map.getPanes().overlayPane);
  overlay.select('svg')
    .attrs({
      "pointer-events": "auto",
      "id": "mapOverlay"
    });

  // add sweeping and prediction controls
  sideBySide = L.control.sideBySide(tiles["5-4-2_left"], tiles["5-4-2_right"]).addTo(map);
  map.on("keydown", function(event) {
    if (event.originalEvent.key == "Shift") {
      predictionExtent(event.latlng, "add");
    }
    if (event.originalEvent.key == "Enter") {
      retrain(state.polygons);
    }
  });

  map.on('pm:drawstart', switchColors);
  map.on('pm:drawend', labelsEdited);
}

export function layerSelect() {
  let layers = ["#layerLeft", "#layerRight"];
  for (let i in layers) {
    d3.select(layers[i])
      .on("change", switchLayers);

    d3.select(layers[i])
      .selectAll("option")
      .data(["5-4-2", "ESRI", "prediction"]).enter()
      .append("option")
      .attr("value", (d) => d)
      .text((d) => d);
  }
}

export function labelSelect() {
  let labels = ["clean_ice", "debris", "background"];
  d3.select("#labels")
    .on("change", switchLabels);

  d3.select("#labels")
    .selectAll("option")
    .data(labels).enter()
    .append("option")
    .attr("value", (d) => d)
    .text((d) => d);
}

function switchLayers() {
  let newValues = [
    d3.select("#layerLeft").property('value') + "_left",
    d3.select("#layerRight").property('value') + "_right"
  ];

  map.eachLayer(function(layer) {
    if( layer instanceof L.TileLayer )
      map.removeLayer(layer);
  });

  map.addLayer(tiles[newValues[0]]);
  map.addLayer(tiles[newValues[1]]);
  sideBySide.setLeftLayers(tiles[newValues[0]]);
  sideBySide.setRightLayers(tiles[newValues[1]]);
}

function switchLabels() {
}

function predictionExtent(latlng) {
  state.box = L.polygon([[0, 0], [0, 0]], {"id": "predictionBox"});
  state.box.addTo(map);
  map.on("mousemove", extentMoved);
  map.on("keydown", removePatch);
  map.on("click", predPatch);
}

function get_radius(z) {
  switch (z) {
    case 8: return 90000;
    case 9: return 70000;
    case 10: return 50000;
    case 11: return 15000;
    case 12: return 6000;
    case 13: return 4000;
    case 14: return 3000;
  };
}

/*
 * Associate a Listener with an Extent
 *
 * We need a function factory because we need to associate our mousemove with a
 * function that has a single 'event' argument. However, that event needs to
 * refer to a previously instantiated extent / box. So, we return a function
 * that has access to the box in its scope.
 */
function extentMoved(event) {
  let r = get_radius(map.getZoom());
  let box_coords = getPolyAround(event.latlng, r);
  state.box.setLatLngs(box_coords);
}

function removePatch(event) {
  if (event.originalEvent.key == "Escape") {
    state.box.remove();
    map.off("click", predPatch);
  }
}

function predPatch(event) {
  const coords = state.box.getBounds();

  $.ajax({
    type: 'POST',
    url: backendUrl + "/predPatch",
    contentType: "application/json",
    crossDomain:'true',
    dataType: "json",
    data: JSON.stringify({
      extent: {
        xmin: coords._southWest.lng,
        xmax: coords._northEast.lng,
        ymin: coords._southWest.lat,
        ymax: coords._northEast.lat,
        crs: 4326
      },
      classes: dataset["classes"],
      models: models["benjamins_unet"]
    }),
    success: function(response){
      displayPred(response);
      // update polygon list. Initialize all the unselected for labeling.
    },
  });
}

function decode_img(img_str) {
  return "data:image/jpeg;base64," + img_str;
}

function displayPred(data, show_pixel_map=false) {
  let coords = [[data.extent.ymin, data.extent.xmin],
                [data.extent.ymax, data.extent.xmax]];
  if (show_pixel_map) {
    L.imageOverlay(decode_img(data["output_soft"]), coords).addTo(map);
  }

  state.polygons.push(L.geoJSON(data["y_geo"], {pmIgnore: false}));
  L.layerGroup(state.polygons).addTo(map).on("zoomend", (e) => { map.fire("viewreset"); return; });
}

function getPolyAround(latlng, radius){
  // We convert the input lat/lon into the EPSG3857 projection, define our
  // square, then re-convert to lat/lon
  let latlngProjected = L.CRS.EPSG3857.project(latlng),
      x = latlngProjected.x,
      y = latlngProjected.y;

  let top = Math.round(y + radius/2),
      bottom = Math.round(y - radius/2),
      left = Math.round(x - radius/2),
      right = Math.round(x + radius/2);

  // left / right are "x" points while top/bottom are the "y" points
  let topleft = L.CRS.EPSG3857.unproject(L.point(left, top));
  let bottomright = L.CRS.EPSG3857.unproject(L.point(right, bottom));

  return [[topleft.lat, topleft.lng],
          [topleft.lat, bottomright.lng],
          [bottomright.lat, bottomright.lng],
          [bottomright.lat, topleft.lng]];
}

function retrain(polygons) {
  console.log(state.polygons);

  $.ajax({
    type: 'POST',
    url: backendUrl,
    contentType: "application/json",
    crossDomain:'true',
    dataType: "json",
    data: JSON.stringify({ polygons: state.polygons }),
    success: function(response){
      displayPred(response);},
  });
}

function switchColors() {
  let label_type = d3.select("#labels").property('value'),
      options = {};

  if (label_type == "clean_ice") {
    map.pm.setPathOptions({
      color: '#add8e6',
      fillColor: '#add8e6',
      fillOpacity: 0.4,
    });
  } else if (label_type == "debris") {
    map.pm.setPathOptions({
      color: '#e6bbad',
      fillColor: '#e6bbad',
      fillOpacity: 0.4,
    });
  } else {
    map.pm.setPathOptions({
      color: '#c6cbcd',
      fillColor: '#c6cbcd',
      fillOpacity: 0.4,
    });
  }
}


function labelsEdited(event) {
  let layers = event.sourceTarget._layers,
      keys = Object.keys(layers).map((d) => parseInt(d)),
      label_type = d3.select("#labels").property('value');

  // update coordinates 2
  state.polygons[d3.max(keys)] = {
    type: label_type,
    latlngs: layers[d3.max(keys)]._latlngs[0]
  };
  console.log(state.polygons);
}

function polygonClicked(event) {
  // get coordinates of clicked polygon
  // toggle whether it's in the set of selected for labeling

}
