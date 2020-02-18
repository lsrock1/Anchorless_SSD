import os

from yacs.config import CfgNode as CN


_C = CN()

_C.COLORS = ((255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128),
          (0, 255, 255, 128), (255, 0, 255, 128), (255, 255, 0, 128))
_C.MEANS = (104, 117, 123)

_C.NUM_CLASSES = 21
_C.LR_STEPS = (80000, 100000, 120000,)
_C.MAX_ITER = 120000
_C.MIN_DIM = 300
_C.STRIDES = (8, 16, 32, 60, 100, 300,)
_C.MIN_SIZES = (30, 60, 111, 162, 213, 264,)
_C.MAX_SIZES = (60, 111, 162, 213, 264, 315,)

_C.HEAD = "anchorless" # or anchor
_C.NMS_TH = 0.6

# ===== Anchorless =====
# ======================
_C.ANCHORLESS = CN()
_C.ANCHORLESS.INFERENCE_TH = 0.05
_C.ANCHORLESS.PRE_NMS_TOP_N = 1000
_C.ANCHORLESS.OUT_CHANNELS = 256


# ===== Anchor =====
# ==================
_C.ANCHOR = CN()
_C.ANCHOR.RATIOS = ([2], [2, 3], [2, 3], [2, 3], [2], [2],)
_C.ANCHOR.VARIANCE = (0.1, 0.2,)
_C.ANCHOR.CLIP = True
_C.ANCHOR.OVERLAP_THRESH = 0.5
_C.ANCHOR.INFERENCE_TH = 0.01


# ===== Backbone =====
# ====================
_C.BACKBONE_TYPE = 'vgg'
_C.FPN_ON = True


# ===== FPN =====
# ===============
_C.FPN = CN()
_C.FPN.OUT_CHANNELS = 256

# ===== TEST =====
# ================
_C.TEST = CN()
_C.TEST.DETECTIONS_PER_IMG = 100
