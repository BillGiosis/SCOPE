"""Shared constants for benchmark preprocessing and reporting."""

DEFAULT_INPUT_SIZE = 224
CLADC_INPUT_SIZE = DEFAULT_INPUT_SIZE
BDDC_INPUT_SIZE = DEFAULT_INPUT_SIZE

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

CLADC_CLASS_NAMES = ('unused', 'pedestrian', 'cyclist', 'car', 'truck', 'tram', 'tricycle')
CLADC_LOCATION_NAMES = ('Citystreet', 'Countryroad', 'Highway')
