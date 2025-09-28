# gemini_vertex_baselines.py

import os
import json
import time
import re
import random
import base64
import mimetypes
import streamlit as st
import pandas as pd
from datetime import datetime
from PIL import Image
from io import BytesIO
import hashlib

from google import genai
from google.genai import types
from google.genai import errors as genai_errors

# ------------------------------
# Config & Env
# ------------------------------
try:
    # Load environment variables from .env file if available
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Environment variables for API authentication
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GOOGLE_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")
GOOGLE_LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION")
# Determine if using Vertex AI or direct API
USE_VERTEX = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "true").lower() in ["1","true","yes"]

# Validate environment variables based on mode
if USE_VERTEX:
    if not GOOGLE_PROJECT or not GOOGLE_LOCATION:
        st.error("GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION must be set for Vertex mode")
        st.stop()
else:
    if not GEMINI_API_KEY:
        st.error("GEMINI_API_KEY is not set (for public API mode). Either set it or enable Vertex mode.")
        st.stop()

# Set seed for reproducibility
random.seed(42)

# ------------------------------
# Models & Settings
# ------------------------------
GEMINI_MODELS = [
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
]
RESULTS_XLSX = "finchart_gemini_vertex_results.xlsx"   # Excel file for results
RATE_LIMIT_DELAY = 0.2  # Delay between API calls (to avoid rate limits)

# ------------------------------
# Normalization helpers (robust)
# ------------------------------
def safe_float(x):
    """Extract numeric values for QA-type answers."""
    try:
        numbers = re.findall(r"[-+]?\d*\.?\d+", str(x).replace(",", ""))
        if not numbers:
            return None
        return float(numbers[-1])
    except:
        return None

def normalize_tf(pred):
    """Normalize True/False answers to 'true' or 'false'."""
    p = str(pred).lower()
    # remove non-alphabet characters
    p = re.sub(r'[^a-z]', '', p)
    if p.startswith('true') or p == 't':
        return 'true'
    if p.startswith('false') or p == 'f':
        return 'false'
    if 'true' in p:
        return 'true'
    if 'false' in p:
        return 'false'
    return p

def normalize_mc(pred):
    """Normalize multiple-choice answers (A/B/C/D)."""
    p = str(pred).lower()
    # Look for a single letter a/b/c/d
    m = re.search(r'\b([abcd])\b', p)
    if m:
        return m.group(1)
    # Fallback: strip other chars, keep first a-d
    p2 = re.sub(r'[^a-d]', '', p)
    if p2 and p2[0] in 'abcd':
        return p2[0]
    return p.strip()

def acc(predicts, ground_truth, qtype):
    """Calculate accuracy based on question type (TF, MC, QA)."""
    correct = 0
    for p, gt in zip(predicts, ground_truth):
        if qtype == "QA":
            p = safe_float(p); gt = safe_float(gt)
            if p is None or gt is None:
                continue
        elif qtype == "TF":
            p = normalize_tf(p); gt = normalize_tf(gt)
        else:
            p = normalize_mc(p); gt = normalize_mc(gt)
        if p == gt:
            correct += 1
    return correct / len(ground_truth) if ground_truth else 0

# ------------------------------
# Load dataset
# ------------------------------
def clean_image_name(img_name):
    """Standardize image filenames (remove _q1, _Q2, etc.)."""
    if not isinstance(img_name, str):
        return img_name
    return (img_name.replace("_q1.jpg", ".jpg")
                  .replace("_q2.jpg", ".jpg")
                  .replace("_Q1.jpg", ".jpg")
                  .replace("_Q2.jpg", ".jpg"))

def load_local_unified_dataset(test_folder, MC_json, QA_json, TF_json, limit=1000):
    """Load QA, TF, MC data + images into one dataset."""
    unified = []
    def add_items(json_path, qtype, subfolder):
        if json_path and os.path.exists(json_path):
            with open(json_path, "r") as f:
                items = json.load(f)
            for item in items:
                img = clean_image_name(item.get("image"))
                question = item.get("question")
                answer = item.get("answer")
                choices = item.get("choices") if qtype == "MC" else None
                img_path = os.path.join(test_folder, subfolder, img)
                if not os.path.exists(img_path):
                    st.warning(f"Missing image: {img_path}")
                    continue
                unified.append({
                    "image_path": img_path,
                    "question": question,
                    "answer": answer,
                    "choices": choices,
                    "type": qtype
                })
    # Load each type
    add_items(QA_json, "QA", "QA_images")
    add_items(TF_json, "TF", "TF_images")
    add_items(MC_json, "MC", "MC_images")
    random.shuffle(unified)
    if limit:
        unified = unified[:limit]
    return unified

# ------------------------------
# Caching
# ------------------------------
def cache_key(model, question, image_path, qtype=None):
    """Create a unique cache key for a query."""
    return hashlib.md5(f"{model}-{question}-{image_path}-{qtype}".encode()).hexdigest()

def load_cache(fname):
    """Load cache file if it exists."""
    if os.path.exists(fname):
        try:
            with open(fname, "r") as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_cache(fname, data):
    """Save cache data to file."""
    with open(fname, "w") as f:
        json.dump(data, f, indent=2)

def generate_content_cached(model, question, image_path, qtype=None):
    """Return cached result if exists, otherwise run model and cache it."""
    today = datetime.now().strftime("%Y-%m-%d")
    suffix = f"_{qtype}" if qtype else ""
    fname = f"results_cache_{model}-{today}{suffix}.json"
    cache = load_cache(fname)
    key = cache_key(model, question, image_path, qtype)
    if key in cache:
        return cache[key]
    val = generate_content(model, question, image_path)
    cache[key] = val
    save_cache(fname, cache)
    return val

# ------------------------------
# NEW: export preds for ensembling
# ------------------------------
def evaluate_and_dump_preds(model, split_set, out_csv):
    """Evaluate model and dump predictions into a CSV (for ensemble use)."""
    import pandas as pd
    rows = []
    for i, ex in enumerate(split_set):
        pred = generate_content_cached(model, ex["question"], ex["image_path"], qtype=ex["type"])
        rows.append({
            "idx": i,
            "type": ex["type"],
            "answer": ex["answer"],
            f"{model}_pred": pred
        })
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    return df

# ------------------------------
# Results Excel helper
# ------------------------------
def upsert_results_excel(xlsx_path, row_dict):
    """Insert results into Excel file (append new rows)."""
    cols = ["Model", "TF Score", "MC Score", "QA Score", "Weighted Avg",
            "Total Questions", "TF Qs", "MC Qs", "QA Qs", "Timestamp"]
    try:
        if os.path.exists(xlsx_path):
            df = pd.read_excel(xlsx_path)
        else:
            df = pd.DataFrame(columns=cols)
    except Exception:
        df = pd.DataFrame(columns=cols)
    for c in cols:
        if c not in df.columns:
            df[c] = pd.Series(dtype=object)
    df = pd.concat([df, pd.DataFrame([row_dict])], ignore_index=True)
    with pd.ExcelWriter(xlsx_path, engine="openpyxl", mode="w") as writer:
        df.to_excel(writer, index=False)

# ------------------------------
# Client & generation
# ------------------------------
def make_gemini_client():
    """Create a Gemini client (Vertex or public API)."""
    if USE_VERTEX:
        return genai.Client(vertexai=True, project=GOOGLE_PROJECT, location=GOOGLE_LOCATION)
    elif GEMINI_API_KEY:
        return genai.Client(api_key=GEMINI_API_KEY)
    else:
        raise RuntimeError("No valid authentication found. Set GEMINI_API_KEY or Vertex vars.")

def maybe_resize(image_path, max_side=1024):
    """Resize large images to max_side dimension for efficiency."""
    try:
        img = Image.open(image_path)
        w, h = img.size
        if max(w, h) <= max_side:
            return image_path
        scale = max_side / max(w, h)
        img2 = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        buf = BytesIO()
        img2.save(buf, format="PNG")
        tmp = image_path + ".resized.png"
        with open(tmp, "wb") as f:
            f.write(buf.getvalue())
        return tmp
    except Exception as e:
        st.write(f"resize error: {e}")
        return image_path

def build_content_parts(question, image_path=None):
    """Build Gemini API request parts (image + text)."""
    parts = []
    if image_path:
        path2 = maybe_resize(image_path)
        with open(path2, "rb") as f:
            b = f.read()
        mime, _ = mimetypes.guess_type(path2)
        if mime is None:
            mime = "image/png"
        parts.append(types.Part(
            inline_data = types.Blob(mime_type=mime, data=base64.b64encode(b).decode())
        ))
    parts.append(types.Part(text=question))
    return parts

def generate_content(model, question, image_path=None, max_retries=5):
    """Send question + image to Gemini model, with retry logic for errors."""
    client = make_gemini_client()
    content = types.Content(role="user", parts=build_content_parts(question, image_path))

    for attempt in range(max_retries):
        try:
            resp = client.models.generate_content(
                model=model,
                contents=[content],
                config=types.GenerateContentConfig(temperature=0.0)
            )
            break
        except genai_errors.APIError as e:
            # Retry on transient errors (503, unavailable)
            if "503" in str(e) or "UNAVAILABLE" in str(e):
                wait = (2 ** attempt) + random.random()
                st.write(f"[{model}] 503 → retry in {wait:.1f}s (attempt {attempt+1}/{max_retries})")
                time.sleep(wait)
                continue
            else:
                st.write(f"[{model}] APIError: {e}")
                raise
    else:
        st.write(f"[{model}] Max retries reached, skipping")
        return None

    # Parse response content
    if not getattr(resp, "candidates", None):
        return None
    if hasattr(resp, "text") and resp.text:
        return resp.text
    cand = resp.candidates[0].content
    texts = [p.text for p in getattr(cand, "parts", []) if getattr(p, "text", None)]
    return " ".join(texts) if texts else None

# ------------------------------
# Evaluation
# ------------------------------
def evaluate_split(model, split_set):
    """Evaluate a model on a dataset split (compute accuracy per task)."""
    preds, gts, types_ = [], [], []
    total = len(split_set)
    progress = st.progress(0)
    status = st.empty()

    for i, ex in enumerate(split_set):
        try:
            pred = generate_content_cached(model, ex["question"], ex["image_path"], qtype=ex["type"])
        except Exception as e:
            st.write(f"Error at {i}, model {model}: {e}")
            return None
        preds.append(pred)
        gts.append(ex["answer"])
        types_.append(ex["type"])
        progress.progress((i+1)/total)
        status.text(f"{i+1}/{total}")

        time.sleep(RATE_LIMIT_DELAY)

    # Separate predictions by type
    tf_preds = [p for p, t in zip(preds, types_) if t == "TF"]
    tf_gts   = [gt for gt, t in zip(gts, types_) if t == "TF"]
    mc_preds = [p for p, t in zip(preds, types_) if t == "MC"]
    mc_gts   = [gt for gt, t in zip(gts, types_) if t == "MC"]
    qa_preds = [p for p, t in zip(preds, types_) if t == "QA"]
    qa_gts   = [gt for gt, t in zip(gts, types_) if t == "QA"]

    # Compute accuracies
    tf_acc = acc(tf_preds, tf_gts, "TF") if tf_gts else 0
    mc_acc = acc(mc_preds, mc_gts, "MC") if mc_gts else 0
    qa_acc = acc(qa_preds, qa_gts, "QA") if qa_gts else 0
    w = (len(tf_gts)*tf_acc + len(mc_gts)*mc_acc + len(qa_gts)*qa_acc) / total if total else 0
    return tf_acc, mc_acc, qa_acc, w, len(tf_gts), len(mc_gts), len(qa_gts)

# ------------------------------
# Streamlit UI
# ------------------------------
st.title("Gemini Baselines via Vertex / GenAI SDK")

# Dataset paths
test_folder = "C:/Users/sharm/Desktop/DSProjects/genai-agentiai-nlp/Project/test_folder"
MC_json = os.path.join(test_folder, "MC_data.json")
QA_json = os.path.join(test_folder, "QA_data.json")
TF_json = os.path.join(test_folder, "TF_data.json")

# Load dataset and split
unified = load_local_unified_dataset(test_folder, MC_json, QA_json, TF_json, limit=1000)
n = len(unified)
n_train = int(0.7 * n)
n_val = int(0.1 * n)
train_set = unified[:n_train]
val_set = unified[n_train:n_train + n_val]
test_set = unified[n_train + n_val:]
st.write("Splits (train/val/test):", len(train_set), len(val_set), len(test_set))

# Model selection dropdown
model_choice = st.selectbox("Choose Gemini model", GEMINI_MODELS, key="model_selector")

# Run evaluation for selected model
if st.button("Run this baseline", key="run_single_baseline"):
    res = evaluate_split(model_choice, test_set)
    if res is None:
        st.error(f"Evaluation failed for {model_choice}")
    else:
        tf_acc, mc_acc, qa_acc, w, tf_q, mc_q, qa_q = res
        st.write(f"**{model_choice}** → TF: {tf_acc:.2%}, MC: {mc_acc:.2%}, QA: {qa_acc:.2%}, Weighted: {w:.2%}")
        row = {
            "Model": model_choice,
            "TF Score": tf_acc, "MC Score": mc_acc, "QA Score": qa_acc,
            "Weighted Avg": w,
            "Total Questions": len(test_set),
            "TF Qs": tf_q, "MC Qs": mc_q, "QA Qs": qa_q,
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        upsert_results_excel(RESULTS_XLSX, row)
        try:
            st.dataframe(pd.read_excel(RESULTS_XLSX))
        except:
            pass

# Export predictions for selected model
if st.button("Export preds for selected Gemini model", key="export_gemini_preds_single"):
    out_csv = f"preds_{model_choice}.csv"
    df_preds = evaluate_and_dump_preds(model_choice, test_set, out_csv)
    st.success(f"Saved predictions to {out_csv}")
    st.dataframe(df_preds.head(5))

# Batch export predictions for multiple models (for ensembling)
if st.button("Export preds for Gemini ensemble models", key="export_gemini_preds_batch"):
    for m in ["gemini-2.0-flash", "gemini-2.5-pro"]:
        out_csv = f"preds_{m}.csv"
        df_preds = evaluate_and_dump_preds(m, test_set, out_csv)
        st.write(f"Saved {out_csv} ({len(df_preds)} rows)")
    st.success("Exported Gemini predictions for ensembling.")

# Run all models and show leaderboard
if st.button("Run all models", key="run_all_models"):
    leaderboard = []
    for m in GEMINI_MODELS:
        st.write("Running:", m)
        res = evaluate_split(m, test_set)
        if res is None:
            st.write(f"Skipped {m}")
            continue
        _, _, _, w, _, _, _ = res
        leaderboard.append((m, w))
        st.write(f"{m} → {w:.2%}")
    leaderboard.sort(key=lambda x: x[1], reverse=True)
    st.write("### Leaderboard")
    for mm, sc in leaderboard:
        st.write(f"{mm}: {sc:.2%}")
