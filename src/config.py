RAW_FEATURES = [
    "Air temperature",
    "Process temperature",
    "Rotational speed",
    "Torque",
    "Tool wear",
]

DISPLAY_NAMES = {
    "Air temperature": "Air temperature [K]",
    "Process temperature": "Process temperature [K]",
    "Rotational speed": "Rotational speed [rpm]",
    "Torque": "Torque [Nm]",
    "Tool wear": "Tool wear [min]",
}

ENGINEERED_FEATURES = [
    "Mechanical Power W",
    "Temperature Delta K",
    "Wear-Torque Interaction",
]

MODEL_FEATURES = RAW_FEATURES + ENGINEERED_FEATURES

TARGET = "Machine failure"

COLUMN_ALIASES = {
    "Air temperature [K]": "Air temperature",
    "Process temperature [K]": "Process temperature",
    "Rotational speed [rpm]": "Rotational speed",
    "Torque [Nm]": "Torque",
    "Tool wear [min]": "Tool wear",
}

FAILURE_MODES = ["TWF", "HDF", "PWF", "OSF", "RNF"]

FAILURE_MODE_LABELS = {
    "TWF": "Tool wear failure",
    "HDF": "Heat dissipation failure",
    "PWF": "Power failure",
    "OSF": "Overstrain failure",
    "RNF": "Random failure",
}

INPUT_RANGES = {
    "Air temperature": (290.0, 320.0),
    "Process temperature": (300.0, 330.0),
    "Rotational speed": (1100.0, 2900.0),
    "Torque": (3.0, 80.0),
    "Tool wear": (0.0, 300.0),
}

RANDOM_SEED = 42
MODEL_VERSION = "v1.0-baseline"
DEFAULT_THRESHOLD = 0.5

PIPELINE_PATH = "models/predictive_maintenance_pipeline.pkl"
METADATA_PATH = "models/pipeline_metadata.json"

RISK_BANDS = [
    (0.30, "NORMAL", "green"),
    (0.65, "WATCHLIST", "orange"),
    (1.01, "HIGH RISK", "red"),
]
