HLS_URL = "https://visdeurbel.videostreams.nl/hls/visdeurbel/index.m3u8"

FRAME_INTERVAL_SEC = 2    # seconds between processed frames
COOLDOWN_SEC = 300        # seconds between notifications (5 min)

MIN_CONTOUR_AREA = 2500   # minimum blob area in pixels to count as fish
MAX_FRAME_COVERAGE = 0.15 # reject blobs covering more than this fraction of the frame (shimmer)
MIN_FISH_LENGTH = 60      # minimum pixels on the longest bounding-box side
BLUR_KERNEL = 21          # Gaussian blur kernel applied before MOG2 (must be odd); kills sediment
MOG2_HISTORY = 200        # number of frames for background model
MOG2_THRESHOLD = 60       # detection sensitivity (lower = more sensitive)

SNAPSHOT_DIR = "snapshots"

SHOW_PREVIEW = False      # set True to open an OpenCV window for debugging
