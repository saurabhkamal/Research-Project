import os, glob, re, math
import pandas as pd
from collections import Counter
from statistics import median

# ----------------------------
# Helpers: normalization + accuracy
# ----------------------------

def safe_float(x):
    """Extract numeric value from string (handles commas, decimals, signs)."""
    try:
        numbers = re.findall(r"[-+]?\d*\.?\d+", str(x).replace(",", ""))
        if not numbers:
            return None
        return float(numbers[-1])  # return the last number found
    except Exception:
        return None

def normalize_tf(pred):
    """Normalize True/False answers to canonical 'true' or 'false'."""
    p = str(pred).lower()
    p = re.sub(r'[^a-z]', '', p)  # keep only letters
    if p.startswith('true') or p == 't':   return 'true'
    if p.startswith('false') or p == 'f':  return 'false'
    if 'true' in p:  return 'true'
    if 'false' in p: return 'false'
    return p

def normalize_mc(pred):
    """Normalize multiple-choice answers (A/B/C/D)."""
    p = str(pred).lower()
    m = re.search(r'\b([abcd])\b', p)  # check if standalone a/b/c/d
    if m: return m.group(1)
    p2 = re.sub(r'[^a-d]', '', p)  # strip everything except a–d
    if p2 and p2[0] in 'abcd': return p2[0]
    return p.strip()

def acc(preds, gts, qtype):
    """
    Compute accuracy for a given question type.
    - TF → normalized true/false
    - MC → normalized multiple choice
    - QA → numeric match (safe float)
    """
    preds = list(preds)
    gts   = list(gts)

    correct = 0
    for p, gt in zip(preds, gts):
        if qtype == "QA":
            p = safe_float(p); gt = safe_float(gt)
            if p is None or gt is None:
                continue
        elif qtype == "TF":
            p = normalize_tf(p); gt = normalize_tf(gt)
        else:  # MC
            p = normalize_mc(p); gt = normalize_mc(gt)
        if p == gt:
            correct += 1
    return (correct / len(gts)) if len(gts) > 0 else 0

# ----------------------------
# Load CSVs (robust, no duplicate pred columns)
# ----------------------------

from functools import reduce

KEYS = ["idx", "type", "answer"]  # common keys across all prediction files

def model_from_path(p):
    """
    Extract model name from CSV filename.
    Example: preds_gpt-4.1-2025-04-14.csv → gpt-4.1-2025-04-14
    Also normalizes naming inconsistencies like gpt4o → gpt-4o.
    """
    base = os.path.basename(p)
    m = re.sub(r"^preds_", "", base)
    m = re.sub(r"\.csv$", "", m)
    m = m.replace("gpt4o", "gpt-4o")  # normalization
    return m

# Find all prediction CSVs
csv_paths = sorted(glob.glob("preds_*.csv"))
if not csv_paths:
    raise SystemExit("No preds_*.csv files found. Export predictions from your GPT/Gemini apps first.")

cleaned = []
seen_models = set()
for p in csv_paths:
    df = pd.read_csv(p)

    # Ensure required columns exist
    missing = [k for k in KEYS if k not in df.columns]
    if missing:
        raise ValueError(f"{p} missing columns {missing}. Got: {list(df.columns)}")

    # Normalize dtypes for keys
    df["idx"] = pd.to_numeric(df["idx"], errors="coerce").astype("Int64")
    df["type"] = df["type"].astype(str)

    # Find prediction columns
    pred_cols_here = [c for c in df.columns if c.endswith("_pred") and c not in KEYS]
    if len(pred_cols_here) == 0:
        raise ValueError(f"{p} has no *_pred column.")

    model_name = model_from_path(p)
    expected = f"{model_name}_pred"

    # Pick the prediction column that matches model name
    if expected in pred_cols_here:
        keep_col = expected
    else:
        if len(pred_cols_here) > 1:
            print(f"Warning: {p} had multiple pred cols {pred_cols_here}; using {pred_cols_here[0]}")
        keep_col = pred_cols_here[0]

    # Keep only keys + prediction col, rename to unique model name
    df = df[KEYS + [keep_col]].copy()
    new_col = f"{model_name}_pred"
    df = df.rename(columns={keep_col: new_col})

    # Skip duplicate models
    if model_name in seen_models:
        print(f"Skipping duplicate model CSV for {model_name}: {p}")
        continue
    seen_models.add(model_name)

    cleaned.append(df)

if not cleaned:
    raise SystemExit("No valid prediction files after cleaning.")

# Merge all prediction files on idx/type/answer
merged = reduce(lambda l, r: pd.merge(l, r, on=KEYS, how="outer", validate="m:1"), cleaned)

# Collect prediction columns
pred_cols = [c for c in merged.columns if c.endswith("_pred")]
if not pred_cols:
    raise SystemExit("No *_pred columns found after merge.")

# Map prediction columns to model names
def col_to_modelname(col):
    return col.replace("_pred", "")

models_in_ensemble = [col_to_modelname(c) for c in pred_cols]

# ----------------------------
# Pull weights from Excel results (if available)
# ----------------------------
def load_weights_from_excels(models):
    """
    Load model-specific weights from previous results Excel files.
    Default weight = 1.0 for all tasks if no data found.
    """
    weights = {m: {"TF":1.0, "MC":1.0, "QA":1.0} for m in models}
    excel_files = [
        "finchart_baseline_results.xlsx",              # GPT results
        "finchart_gemini_vertex_results.xlsx",         # Gemini results
    ]
    frames = []
    for x in excel_files:
        if os.path.exists(x):
            try:
                frames.append(pd.read_excel(x))
            except Exception:
                pass
    if not frames:
        return weights

    df_all = pd.concat(frames, ignore_index=True)
    if not set(["Model","TF Score","MC Score","QA Score"]).issubset(df_all.columns):
        return weights

    # For each model, take the most recent row’s scores
    for m in models:
        rows = df_all[df_all["Model"] == m]
        if len(rows) > 0:
            row = rows.iloc[-1]
            tf = float(row.get("TF Score", 1.0) or 1.0)
            mc = float(row.get("MC Score", 1.0) or 1.0)
            qa = float(row.get("QA Score", 1.0) or 1.0)
            weights[m] = {"TF": tf, "MC": mc, "QA": qa}
    return weights

weights = load_weights_from_excels(models_in_ensemble)

# ----------------------------
# Weighted voting (for ensemble)
# ----------------------------
def weighted_vote(values_by_model, task):
    """Perform weighted majority voting for TF/MC tasks."""
    score = {}
    for m, val in values_by_model.items():
        if val is None or (isinstance(val, float) and math.isnan(val)):
            continue
        w = float(weights.get(m, {}).get(task, 1.0))
        score[val] = score.get(val, 0.0) + w
    if not score:
        return None
    # return label with maximum weight
    return max(score.items(), key=lambda kv: kv[1])[0]

def ensemble_row(row):
    """Combine predictions from all models for one row."""
    items = { col_to_modelname(c): row[c] for c in pred_cols }
    t = row["type"]

    if t == "TF":
        vals = {m: normalize_tf(v) for m, v in items.items() if pd.notna(v)}
        return weighted_vote(vals, "TF")

    if t == "MC":
        vals = {m: normalize_mc(v) for m, v in items.items() if pd.notna(v)}
        return weighted_vote(vals, "MC")

    # QA task → numeric median (or weighted fallback)
    nums = [safe_float(v) for v in items.values() if pd.notna(v)]
    nums = [x for x in nums if x is not None]
    if nums:
        return float(median(nums))
    # fallback: choose non-null prediction with highest weight
    best_v, best_w = None, -1
    for m, v in items.items():
        if pd.notna(v):
            w = float(weights.get(m, {}).get("QA", 1.0))
            if w > best_w:
                best_v, best_w = v, w
    return best_v

# Apply ensemble to dataset
merged["ensemble_pred"] = merged.apply(ensemble_row, axis=1)

# ----------------------------
# Compute ensemble scores
# ----------------------------
tf_mask = merged["type"] == "TF"
mc_mask = merged["type"] == "MC"
qa_mask = merged["type"] == "QA"

# Compute accuracy for each task
tf_acc = acc(merged.loc[tf_mask, "ensemble_pred"], merged.loc[tf_mask, "answer"], "TF")
mc_acc = acc(merged.loc[mc_mask, "ensemble_pred"], merged.loc[mc_mask, "answer"], "MC")
qa_acc = acc(merged.loc[qa_mask, "ensemble_pred"], merged.loc[qa_mask, "answer"], "QA")

# Weighted overall score
total = len(merged)
weighted = ((tf_mask.sum()*tf_acc) + (mc_mask.sum()*mc_acc) + (qa_mask.sum()*qa_acc)) / total if total else 0.0

# Print results
print(f"Ensemble → TF: {tf_acc:.2%}, MC: {mc_acc:.2%}, QA: {qa_acc:.2%}, Weighted: {weighted:.2%}")

# Save predictions
merged.to_csv("preds_ensemble.csv", index=False)
print("Wrote preds_ensemble.csv")
