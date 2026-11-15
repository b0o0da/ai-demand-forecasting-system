from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "train.csv"
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
IMAGES = ROOT / "images"

DATA_URLS = [
    "https://raw.githubusercontent.com/jgonzalezab/Store-Item-Demand-Forecasting/master/Data/train.csv",
    "https://raw.githubusercontent.com/ugursaricam/store-item-demand-forecasting/master/datasets/train.csv",
]

HORIZON = 28
SEED = 42
TRAIN_END = "2016-06-30"
VAL_END = "2017-03-31"
LEAD_TIME_DAYS = 7
