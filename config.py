HLS_URL = "https://visdeurbel.videostreams.nl/hls/visdeurbel/index.m3u8"

FRAME_INTERVAL_SEC = 2    # seconds between processed frames
COOLDOWN_SEC = 300        # seconds between notifications (5 min)

MIN_CONTOUR_AREA = 500    # sanity-check floor; MIN_FISH_LENGTH + MIN_ELONGATION are the real filters
MAX_FRAME_COVERAGE = 0.15 # reject blobs covering more than this fraction of the frame (shimmer)
MIN_FISH_LENGTH = 80      # minimum pixels on the longest bounding-box side
MIN_ELONGATION = 2.0      # longest side must be ≥2× the shortest; rejects square/round blobs
BLUR_KERNEL = 21          # Gaussian blur kernel applied before MOG2 (must be odd); kills sediment
MOG2_HISTORY = 200        # number of frames for background model
MOG2_THRESHOLD = 60       # detection sensitivity (lower = more sensitive)

# Regions to black out before MOG2 (coordinates at 640×360 scale).
# Used to suppress static overlays that change content (e.g. burned-in timestamp).
MASK_REGIONS = [
    (380, 0, 640, 35),   # timestamp overlay — top-right corner (wider+taller to fully cover)
]

SNAPSHOT_DIR = "snapshots"

SHOW_PREVIEW = False      # set True to open an OpenCV window for debugging

# Second-pass LLM vision verification (Granite Vision via llama-server)
LLM_VERIFY = True
LLM_URL = "http://llm:8080"   # llama-server base URL (overridable via LLM_URL env var)
LLM_TIMEOUT = 30              # seconds to wait for LLM response
