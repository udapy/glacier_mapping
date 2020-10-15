#!/usr/bin/env python3
from pathlib import Path
from glacier_mapping.models.unet_dropout import UnetDropout
import glacier_mapping.data.process_slices_funs as pf
from glacier_mapping.infer import run_model_on_tile
from web.backend.ServerModelsAbstract import BackendModel
import numpy as np
import os
import sys
import time
import torch
import yaml
import json

DATA_DIR = os.environ["DATA_DIR"]

class PytorchUNet(BackendModel):
    def __init__(self, model_spec, gpuid, verbose=False):
        self.input_size = model_spec["inputShape"]
        self.downweight_padding = 0
        self.stride_x, self.stride_y, _ = self.input_size
        self.process_conf = model_spec["process"]
        self.outchannels = model_spec["args"]["outchannels"]

        model_path = Path(DATA_DIR, model_spec["fn"])
        if torch.cuda.is_available():
            state = torch.load(model_path)
        else:
            state = torch.load(model_path, map_location=torch.device("cpu"))

        self.model = UnetDropout(**model_spec["args"])
        self.model.load_state_dict(state)
        self.model.eval()
        self.verbose = verbose

    def run(self, img, **kwargs):
        img = self.preprocess(img)
        return run_model_on_tile(img, self.model, num_output_channels=self.outchannels)

    def preprocess(self, img, **kwargs):
        conf = yaml.safe_load(open(self.process_conf, "r"))
        pf = conf["process_funs"]
        img = np.nan_to_num(img, nan=pf["impute"]["value"])

        stats = json.load(open(pf["normalize"]["stats_path"], "r"))
        img = pf.normalize_(img, stats["means"], stats["stds"])
        channels = pf["extract_channel"]["img_channels"]
        return img[:, :, channels]


    def run_model_on_batch(self, batch_data, batch_size=32, predict_central_pixel_only=False):
        """ Expects batch_data to have shape (none, 240, 240, 4) and have values in the [0, 255] range.
        """
        raise NotImplementedError("run_model_on_batch method of ServerModelsPytorch not implemented.")

    def retrain(self, **kwargs):
        raise NotImplementedError("retrain method of ServerModelsPytorch not implemented.")

    def add_sample(self, tdst_row, bdst_row, tdst_col, bdst_col, class_idx):
        raise NotImplementedError("add_sample method of ServerModelsPytorch not implemented.")

    def undo(self):
        raise NotImplementedError("undo method of ServerModelsPytorch not implemented.")

    def reset(self):
        raise NotImplementedError("reset method of ServerModelsPytorch not implemented.")
