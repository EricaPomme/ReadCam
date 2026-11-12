"""Default configuration values and shared constants."""


# Camera defaults
DEFAULT_CAMERA_INDEX = 0
DEFAULT_FRAME_INTERVAL = 50      # ms (~20 fps)
DEFAULT_OCR_INTERVAL = 1500      # ms (run OCR every 1.5s)

# Preview
PREVIEW_MAX_WIDTH = 640
PREVIEW_MAX_HEIGHT = 480

# Contour detection (text regions)
CONTOUR_MIN_AREA = 500
CONTOUR_MAX_AREA = 100000
CONTOUR_MIN_ASPECT_RATIO = 0.2
CONTOUR_MAX_ASPECT_RATIO = 10.0

# Results manager
CONFIDENCE_MAX = 100
CONFIDENCE_DECAY_DELAY = 5.0    # seconds before confidence starts decaying
CONFIDENCE_REMOVE_DELAY = 30.0  # seconds before removal from list
TEXT_MIN_CONFIDENCE = 60        # min Tesseract confidence for text results
TEXT_MIN_LENGTH = 4             # min characters for text results
TEXT_REQUIRED_HITS = 3          # text must be seen N times before appearing
TEXT_MIN_ALNUM_RATIO = 0.6      # fraction of chars that must be alphanumeric
TEXT_MIN_LETTERS = 3            # min alphabetic characters in a result

# OCR performance
OCR_MAX_WIDTH = 1000            # downscale frames wider than this before OCR
OCR_ADAPTIVE_HEADROOM = 1.25    # adaptive interval = OCR duration x this

# Bounding box colours (BGR for OpenCV)
COLOUR_QR = (46, 204, 113)          # Green
COLOUR_BARCODE = (219, 152, 52)     # Blue
COLOUR_TEXT = (34, 126, 230)        # Orange
COLOUR_UNKNOWN = (166, 165, 149)    # Grey

# Category names
CAT_QR = "QR"
CAT_BARCODE = "Barcode"
CAT_TEXT = "Text"

# UI
UI_TITLE = "ReadCam"
UI_MIN_WIDTH = 900
UI_MIN_HEIGHT = 600
UI_PREVIEW_MIN_WIDTH = 400
UI_RESULTS_MIN_WIDTH = 380