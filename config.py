from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Raw LUNA16 data (already downloaded by the user).
DATA_DIR = Path(r"D:\Licenta\LungNoduleDetection\luna16")
ANNOTATIONS_CSV = DATA_DIR / "annotations.csv"
CANDIDATES_CSV = DATA_DIR / "candidates_V2" / "candidates_V2.csv"

# Project outputs.
PROJECT_DIR = Path(__file__).resolve().parent
PATCHES_DIR = PROJECT_DIR / "patches"          # preprocessed 2D slices
INDEX_CSV = PATCHES_DIR / "index.csv"          # one row per extracted patch
HARDNEG_INDEX_CSV = PATCHES_DIR / "hardneg_index.csv"  # mined hard negatives
CHECKPOINT_DIR = PROJECT_DIR / "checkpoints"
RESULTS_DIR = PROJECT_DIR / "results"

# ---------------------------------------------------------------------------
# Dataset split (by LUNA16 subset, so no scan appears in two splits)
# ---------------------------------------------------------------------------
TRAIN_SUBSETS = [0, 1, 2, 3, 4, 5, 6, 7]
VAL_SUBSETS = [8]
TEST_SUBSETS = [9]

# ---------------------------------------------------------------------------
# Patch extraction (preprocessing.py)
# ---------------------------------------------------------------------------
PATCH_SIZE_MM = 50.0      # physical size of the extracted axial patch (mm)
PATCH_SIZE_PX = 64        # patch is resampled to PATCH_SIZE_PX x PATCH_SIZE_PX
HU_MIN = -1000.0          # air
HU_MAX = 400.0            # bone / dense tissue; everything above is clipped

# For every positive candidate also take the neighbouring axial slices, which
# multiplies the number of positive samples (the nodule is visible on several
# consecutive slices).
POSITIVE_SLICE_OFFSETS = [-1, 0, 1]

# candidates_V2.csv contains ~754k negatives vs ~1.5k positives.  Keeping all
# negatives is unnecessary for training, so negatives are randomly subsampled
# per scan for the train/val subsets.  The TEST subset keeps ALL negatives so
# that the final evaluation reflects the true candidate distribution.
MAX_NEG_PER_SCAN = 80
KEEP_ALL_NEG_SUBSETS = TEST_SUBSETS
PREPROCESS_SEED = 42

# ---------------------------------------------------------------------------
# Training (train.py)
# ---------------------------------------------------------------------------
BATCH_SIZE = 128
NUM_EPOCHS = 40
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4
EARLY_STOPPING_PATIENCE = 8   # epochs without val-AUC improvement
DROPOUT = 0.3
TRAIN_SEED = 42

# ---------------------------------------------------------------------------
# Hard-negative mining (mine_hard_negatives.py)
# ---------------------------------------------------------------------------
# Run the real detection candidate generator (segmentation + watershed) over
# the scans, then keep the false candidates the current model scores >= this
# probability — i.e. the actual false positives the pipeline produces.
TRAINVAL_SUBSETS = TRAIN_SUBSETS + VAL_SUBSETS   # subsets 0-8: hard-mined
MINE_MAX_NEG_PER_SCAN = 200   # cap candidates scored per scan (bounds runtime)
MINE_HARD_PROB = 0.10         # a negative is "hard" if current model p >= this
