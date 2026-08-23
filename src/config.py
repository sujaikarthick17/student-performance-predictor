from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "students.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "student_performance_model.joblib"
METADATA_PATH = PROJECT_ROOT / "models" / "model_metadata.json"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Change only if your CSV uses another result-column name, such as "result".
TARGET_COLUMN = "result"
FEATURE_COLUMNS = None
RANDOM_STATE = 42
TEST_SIZE = 0.20
