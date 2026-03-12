HLS_URL = "https://visdeurbel.videostreams.nl/hls/visdeurbel/index.m3u8"

FRAME_INTERVAL_SEC = 2    # seconds between processed frames
COOLDOWN_SEC = 300        # seconds between notifications (5 min)

MIN_CONTOUR_AREA = 800    # minimum blob area in pixels to count as fish
MOG2_HISTORY = 50         # number of frames for background model
MOG2_THRESHOLD = 40       # detection sensitivity (lower = more sensitive)

SNAPSHOT_DIR = "snapshots"

SHOW_PREVIEW = False      # set True to open an OpenCV window for debugging
