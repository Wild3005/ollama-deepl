import pandas as pd
from sklearn.metrics import cohen_kappa_score

# =========================================================
# ZERO SHOT
# =========================================================

zero_df = pd.read_csv(
    "outputs/zero_shot_result.csv"
)

zero_qwk = cohen_kappa_score(
    zero_df["score"],
    zero_df["predicted_score"],
    weights="quadratic"
)

# =========================================================
# FEW SHOT
# =========================================================

few_df = pd.read_csv(
    "outputs/few_shot_result.csv"
)

few_qwk = cohen_kappa_score(
    few_df["score"],
    few_df["predicted_score"],
    weights="quadratic"
)

# =========================================================
# PRINT RESULT
# =========================================================

print("\n========== EVALUATION ==========")

print(f"Zero-Shot QWK : {zero_qwk:.4f}")

print(f"Few-Shot  QWK : {few_qwk:.4f}")