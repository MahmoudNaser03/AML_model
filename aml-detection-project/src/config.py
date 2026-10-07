"""
Project-wide configuration and constants.
All notebooks import from here so paths/seeds stay consistent.
"""

from pathlib import Path

# ── Repository root (resolved relative to this file) ─────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Data paths ────────────────────────────────────────────────────────────────
DATA_DIR      = PROJECT_ROOT / "data"
RAW_DIR       = DATA_DIR / "raw"
EXTERNAL_DIR  = DATA_DIR / "external"
PROCESSED_DIR = DATA_DIR / "processed"

# ── Output directories ────────────────────────────────────────────────────────
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
ARTIFACTS_DIR   = PROJECT_ROOT / "artifacts"
MODELS_DIR      = PROJECT_ROOT / "models"
FIGURES_DIR     = PROJECT_ROOT / "figures"
REPORTS_DIR     = PROJECT_ROOT / "reports" / "week3"

# ── Known data files (resolved at import time if they exist) ──────────────────
TRANS_CSV    = RAW_DIR / "HI-Small_Trans.csv"
PATTERNS_TXT = EXTERNAL_DIR / "HI-Small_Patterns.txt"

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42

# ── Train / val / test proportions ────────────────────────────────────────────
TRAIN_FRAC = 0.60
VAL_FRAC   = 0.20
TEST_FRAC  = 0.20

# ── Model operating-point targets ─────────────────────────────────────────────
TARGET_RECALL      = 0.77
PRECISION_FLOOR    = 0.12
PRECISION_FLOORS   = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]

# ── Clustering ────────────────────────────────────────────────────────────────
BEST_K             = 8
CLUSTER_SAMPLE     = 100_000

# ── XGBoost defaults ──────────────────────────────────────────────────────────
XGB_PARAMS = dict(
    objective          = "binary:logistic",
    eval_metric        = ["aucpr"],
    tree_method        = "hist",
    n_estimators       = 1600,
    learning_rate      = 0.02,
    max_depth          = 6,
    min_child_weight   = 8,
    subsample          = 0.90,
    colsample_bytree   = 0.85,
    gamma              = 0.05,
    reg_alpha          = 0.30,
    reg_lambda         = 5.0,
    max_bin            = 256,
    early_stopping_rounds = 800,
    random_state       = SEED,
)

# ── Feature-set metadata ──────────────────────────────────────────────────────
NON_FEATURE_COLS = {
    "target", "tx_id", "row_id", "ts",
    "from_bank", "from_account", "to_bank", "to_account",
    "receiving_currency", "payment_currency", "payment_format",
    "split",
}

EXTERNAL_PATTERN_COLS = {
    "txt_pattern_memberships", "txt_pattern_type_nunique", "txt_patterned",
    "pattern_id", "pattern_type", "pattern_memberships",
    "pattern_type_nunique", "day_of_month",
}

DROP_FEATURE_COLS = {
    "dow", "weekend",
    "sender_counterparty_depth_20", "sender_counterparty_depth_100",
    "receiver_counterparty_depth_20", "receiver_counterparty_depth_100",
    "corridor_amount_z_100", "pair_velocity",
}

PATTERN_BLOCKLIST_TOKENS = ["pattern", "pattern_id", "pattern_type", "membership", "launder_pattern"]

STRUCTURAL_RENAME = {
    "struct_pattern_max_score":    "structural_max_score",
    "struct_pattern_mean_score":   "structural_mean_score",
    "struct_pattern_active_count": "structural_active_count",
}

# ── Create output dirs on import ──────────────────────────────────────────────
for _d in (CHECKPOINTS_DIR, ARTIFACTS_DIR, MODELS_DIR, FIGURES_DIR, REPORTS_DIR, PROCESSED_DIR):
    _d.mkdir(parents=True, exist_ok=True)
