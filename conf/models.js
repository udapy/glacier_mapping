export default {
  "shimas_unet": {
    "metadata": {
      "displayName": "UNet Shima Kind Durian 140"
    },
    "model": {
      "type": "pytorch",
      "numParameters": 7790949,
      "inputShape": [
        512,
        512,
        10
      ],
      "fn": "../data/models/model_188.pt",
      "fineTuneLayer": 0
    }
  },
  "benjamins_unet": {
    "metadata": {
      "displayName": "Benjamins model"
    },
    "model": {
      "type": "pytorch",
      "numParameters": null,
      "args": {
        "inchannels": 15,
        "outchannels": 3,
        "net_depth": 5,
        "channel_layer": 16
      },
      "fn": "runs/run_clean_debris/models/model_final.pt",
      "fineTuneLayer": 0,
      "process": "conf/postprocess.yaml"
    }
  }
}
