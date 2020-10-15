#!/usr/bin/env python
"""
Inference Module

This module contains functions for drawing predictions from an already trained
model, writing the results to file, and extracting corresponding geojsons.
"""
from pathlib import Path
from addict import Dict
import numpy as np
import rasterio
import torch
import yaml
import geopandas as gpd
import shapely.geometry
from shapely.ops import unary_union
import skimage.measure
from skimage.util.shape import view_as_windows
from rasterio.windows import Window
from .data.process_slices_funs import postprocess_tile
from .models.frame import Framework


def squash(x):
    return (x - x.min()) / x.ptp()


def append_name(s, args, filetype="png"):
    return f"{s}_{Path(args.input).stem}-{Path(args.model).stem}-{Path(args.process_conf).stem}.{filetype}"


def write_geotiff(y_hat, meta, output_path):
    """
    Write predictions to geotiff

    :param y_hat: A numpy array of predictions.
    :type y_hat: np.ndarray
    """
    # create empty raster with write geographic information
    dst_file = rasterio.open(
        output_path, 'w',
        driver='GTiff',
        height=y_hat.shape[0],
        width=y_hat.shape[1],
        count=y_hat.shape[2],
        dtype=np.float32,
        crs=meta["crs"],
        transform=meta["transform"]
    )

    y_hat = 255.0 * y_hat.astype(np.float32)
    for k in range(y_hat.shape[2]):
        dst_file.write(y_hat[:, :, k], k + 1)


def predict_tiff(path, model, subset_size=None, conf_path="conf/postprocess.yaml"):
    """
    Load a raster and make predictions on a subwindow
    """
    imgf = rasterio.open(path)
    if subset_size is not None:
        img = imgf.read(window=Window(0, 0, subset_size[0], subset_size[1]))
    else:
        img = imgf.read()
    x, y_hat = run_model_on_tile(img, model)
    return img, x, y_hat


def run_model_on_tile(tile, model, device=None, batch_size=256, num_output_channels=3,
                      input_size=256, down_weight_padding=10):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    height, width, _ = tile.shape
    stride_x = input_size - down_weight_padding*2
    stride_y = input_size - down_weight_padding*2

    output = np.zeros((height, width, num_output_channels), dtype=np.float32)
    counts = np.zeros((height, width), dtype=np.float32) + 0.000000001
    kernel = np.ones((input_size, input_size), dtype=np.float32) * 0.1
    kernel[10:-10, 10:-10] = 1
    kernel[down_weight_padding:down_weight_padding+stride_y,
           down_weight_padding:down_weight_padding+stride_x] = 5

    # build batches for parallelizing inference
    batches = []
    batch_indices = []
    batch_count = 0
    for y_index in (list(range(0, height - input_size, stride_y)) + [height - input_size,]):
        for x_index in (list(range(0, width - input_size, stride_x)) + [width - input_size,]):
            img = tile[y_index:y_index+input_size, x_index:x_index+input_size, :].copy()
            img = np.rollaxis(img, 2, 0).astype(np.float32)

            batches.append(img)
            batch_indices.append((y_index, x_index))
            batch_count += 1

    # compute predictions on all the batches
    batches = np.array(batches)
    model_output = []
    for i in range(0, batch_count, batch_size):
        batch = torch.from_numpy(batches[i:i+batch_size])
        batch = batch.to(device)
        with torch.no_grad():
            outputs = model(batch)
        outputs = outputs.cpu().numpy()
        outputs = np.rollaxis(outputs, 1, 4)
        model_output.append(outputs)
    model_output = np.concatenate(model_output, axis=0)

    for i, (y, x) in enumerate(batch_indices):
        output[y:y+input_size, x:x+input_size] += model_output[i] * kernel[..., np.newaxis]
        counts[y:y+input_size, x:x+input_size] += kernel

    output = output / counts[..., np.newaxis]
    return output


def convert_to_geojson(y_hat, bounds, threshold=0.8):
    """Convert a probability mask to geojson

    :param y_hat: A three dimensional numpy array of mask probabilities.
    :type y_hat: np.array
    :param bounds: The latitude / longitude bounding box of the region
      to write as geojson.
    :type bounds: tuple
    :param threshold: The probability above which an object is
      segmented into the geojson.
    :type threshold: float
    :return (geo_interface, geo_df) (tuple): tuple giving the geojson and
      geopandas data frame corresponding to the thresholded y_hat.
    """
    contours = skimage.measure.find_contours(y_hat, threshold, fully_connected="high")

    for i in range(len(contours)):
        contours[i] = contours[i][:, [1, 0]]
        contours[i][:, 1] = y_hat.shape[1] - contours[i][:, 1]
        contours[i][:, 0] = bounds[0] + (bounds[2] - bounds[0]) * contours[i][:, 0] / y_hat.shape[0]
        contours[i][:, 1] = bounds[1] + (bounds[3] - bounds[1]) * contours[i][:, 1] / y_hat.shape[1]

    contours = [c for c in contours if len(c) > 2]
    polys = [shapely.geometry.Polygon(a) for a in contours]
    polys = unary_union([p for p in polys if p.area > 4e-6])
    mpoly = shapely.geometry.multipolygon.MultiPolygon(polys)
    mpoly = mpoly.simplify(tolerance=0.0005)
    geo_df = gpd.GeoSeries(mpoly)
    return geo_df.__geo_interface__, geo_df


def load_model(train_yaml, model_path):
    """
    :param train_yaml: The path to the yaml file containing training options.
    :param model_path: The path to the saved model checkpoint, from which to
    load the state dict.
    :return model: The model with checkpoint loaded.
    """
    # loads an empty model, without weights
    train_conf = Dict(yaml.safe_load(open(train_yaml, "r")))
    model = Framework(torch.nn.BCEWithLogitsLoss(), train_conf.model_opts, train_conf.optim_opts).model

    # if GPU is available, inference will be faster
    if torch.cuda.is_available():
        state_dict = torch.load(model_path)
    else:
        state_dict = torch.load(model_path, map_location="cpu")

    model.load_state_dict(state_dict)
    return model
