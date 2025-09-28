# baseline_with_error_logging.py

import os
import json
import time
import re
import random
import base64
import mimetypes
import traceback
import streamlit as st
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError, OpenAIError
from PIL import Image
from io import BytesIO
import hashlib

# ------------------------------
# Setup & API Keys
# ------------------------------
load_dotenv()  # Load environment variables from .env file
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")   # OpenAI key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")   # Gemini key
SEED = 42
random.seed(SEED)  # Fix random seed for reproducibility

# ------------------------------
# Model Config
# ------------------------------
# List of baseline models to evaluate
BASELINE_MODELS = [
    "gpt-4o-2024-08-06",
    "gpt-4o-mini",
    "gpt-4.1-2025-04-14",
    "gemini-2.0-flash",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "qwen2.5-vl-7b-instruct",
    "llava-1.6-mistral-7b-hf",
    "llama-3.2-11b-vision-instruct",
    "claude-sonnet-3.5"
]
FINE_TUNE_BASE_MODEL = "gpt-4o-2024-08-06"  # Default base model for fine-tuning

# ------------------------------
# Helpers & Normalization
# ------------------------------
def safe_float(x):
    """
    Extract and convert numeric values from string to float.
    Handles commas and mixed text gracefully.
    """
    try:
        numbers = re.findall(r"[-+]?\d*\.?\d+", str(x).replace(",", ""))
        if not numbers:
            return None
        return float(numbers[-1])  # Take last number found
    except Exception:
        return None

def normalize_tf(pred):
    """
    Normalize True/False predictions to consistent 'true' or 'false'.
    Handles variations like yes/no/1/0.
    """
    p = str(pred).lower().strip()
    if p in ["true", "yes", "1"]:
        return "true"
    if p in ["false", "no", "0"]:
        return "false"
    return p

def normalize_mc(pred):
    """
    Normalize multiple-choice predictions to a/b/c/d.
    Handles variants like 'choice a', 'option b', etc.
    """
    p = str(pred).strip().lower()
    mappings = {
        "a": "a", "choice a": "a", "option a": "a",
        "b": "b", "choice b": "b", "option b": "b",
        "c": "c", "choice c": "c", "option c": "c",
        "d": "d", "choice d": "d", "option d": "d",
    }
    if p in mappings:
        return mappings[p]
    m = re.search(r"\b([abcd])\b", p)
    if m:
        return m.group(1)
    return p

def acc(predicts, ground_truth, question_type):
    """
    Compute accuracy based on question type:
    - QA: numeric comparison
    - TF: normalized true/false
    - MC: normalized multiple choice
    """
    correct = 0
    for p, gt in zip(predicts, ground_truth):
        if question_type == "QA":
            p = safe_float(p)
            gt = safe_float(gt)
            if p is None or gt is None:
                continue
        elif question_type == "TF":
            p = normalize_tf(p)
            gt = normalize_tf(gt)
        else:
            p = normalize_mc(p)
            gt = normalize_mc(gt)
        if p == gt:
            correct += 1
    return correct / len(ground_truth) if ground_truth else 0

def upsert_results_excel(excel_path, row_dict, key_col="Model"):
    """
    Insert or update results in an Excel file.
    Keeps a running record of model scores.
    """
    cols = [
        "Model", "TF Score", "MC Score", "QA Score",
        "Weighted Avg", "Total Questions",
        "TF Qs", "MC Qs", "QA Qs", "Timestamp"
    ]
    if os.path.exists(excel_path):
        try:
            df = pd.read_excel(excel_path)
        except Exception:
            df = pd.DataFrame(columns=cols)
    else:
        df = pd.DataFrame(columns=cols)
    for c in cols:
        if c not in df.columns:
            df[c] = pd.Series(dtype="object")
    mask = df[key_col] == row_dict[key_col]
    if mask.any():
        df.loc[mask, cols] = [row_dict.get(c) for c in cols]
    else:
        df = pd.concat([df, pd.DataFrame([row_dict], columns=cols)], ignore_index=True)
    df.to_excel(excel_path, index=False)
    return df

def clean_image_name(img_name: str) -> str:
    """
    Standardize image names by removing _q1/_q2 suffixes.
    Helps link images with their corresponding questions.
    """
    if not isinstance(img_name, str):
        return img_name
    return (
        img_name.replace("_q1.jpg", ".jpg")
                .replace("_q2.jpg", ".jpg")
                .replace("_Q1.jpg", ".jpg")
                .replace("_Q2.jpg", ".jpg")
    )

def load_local_unified_dataset(test_folder, MC_json, QA_json, TF_json, limit=None):
    """
    Load dataset from JSON files (QA, TF, MC).
    Attach image paths and shuffle for randomness.
    Optionally limit number of examples.
    """
    unified = []
    def add_items(json_path, qtype, img_subfolder):
        if json_path and os.path.exists(json_path):
            with open(json_path, "r") as f:
                items = json.load(f)
            for item in items:
                img_name = clean_image_name(item.get("image"))
                question = item.get("question")
                answer = item.get("answer")
                choices = item.get("choices") if qtype == "MC" else None
                img_path = os.path.join(test_folder, img_subfolder, img_name)
                if not os.path.exists(img_path):
                    st.warning(f"⚠️ Skipping: Could not find image {img_name}")
                    continue
                unified.append({
                    "image_path": img_path,
                    "question": question,
                    "answer": answer,
                    "choices": choices,
                    "type": qtype
                })
    add_items(QA_json, "QA", "QA_images")
    add_items(TF_json, "TF", "TF_images")
    add_items(MC_json, "MC", "MC_images")
    random.shuffle(unified)
    if limit:
        return unified[:limit]
    return unified

def cache_key(model_name, question, image_path, choices=None):
    """Generate a unique cache key for a prediction request."""
    data = f"{model_name}-{question}-{image_path}-{choices}"
    return hashlib.md5(data.encode()).hexdigest()

def load_cache(cache_file):
    """Load cached results from JSON file if exists."""
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_cache(cache_file, cache):
    """Save results to cache JSON file."""
    with open(cache_file, "w") as f:
        json.dump(cache, f, indent=2)

# ------------------------------
# Client factory
# ------------------------------
def make_client_for_model(model_name: str):
    """
    Create OpenAI client depending on model type.
    Gemini models use Google endpoint.
    Others use OpenAI API.
    """
    if model_name.startswith("gemini"):
        return OpenAI(
            api_key=GEMINI_API_KEY,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )
    else:
        return OpenAI(api_key=OPENAI_API_KEY)

# ------------------------------
# Safe call with detailed logging
# ------------------------------
def safe_completion_call(client_obj, model_name, messages, max_tokens=200, temperature=0):
    """
    Call model with retry logic.
    Retries on rate limits and transient OpenAI errors.
    Logs errors for debugging.
    """
    for attempt in range(5):
        try:
            resp = client_obj.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            return resp
        except RateLimitError as e:
            wait = (attempt + 1) * 2
            print(f"⚠️ RateLimitError on attempt {attempt+1} for {model_name}. Wait {wait}s.")
            time.sleep(wait)
        except OpenAIError as e:
            print(f"❌ OpenAIError on attempt {attempt+1} for {model_name}: {e}")
            # Try to extract structured error info
            try:
                err_field = e.error if hasattr(e, "error") else None
                print("Error details:", err_field)
            except Exception as ex2:
                print("Could not get e.error:", ex2)
            traceback.print_exc()
            msg_lower = str(e).lower()
            if "model_not_found" in msg_lower or "does not exist" in msg_lower:
                # immediate exit
                raise
            time.sleep((attempt + 1) * 2)
    raise RuntimeError(f"Failed call for model {model_name} after retries")

# ------------------------------
# Inference with fallback
# ------------------------------
def generate_content(model_name, question, image_path, qtype=None, choices=None):
    """
    Build prompt and send to model with image.
    Falls back to text-only if image upload fails.
    """
    if qtype == "MC":
        options_text = " | ".join(choices) if choices else ""
        prompt = f"[MC] Question: {question}\nOptions: {options_text}\nAnswer with only A, B, C, or D."
    elif qtype == "TF":
        prompt = f"[TF] Question: {question}\nAnswer with only True or False."
    else:
        prompt = f"[QA] Question: {question}\nAnswer with only a number."

    mime_type, _ = mimetypes.guess_type(image_path)
    if not mime_type:
        mime_type = "image/png"
    with open(image_path, "rb") as f:
        img_data = f.read()
    img_b64 = base64.b64encode(img_data).decode("utf-8")

    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{img_b64}"}}
        ]
    }]

    client_obj = make_client_for_model(model_name)
    try:
        resp = safe_completion_call(client_obj, model_name, messages, max_tokens=200, temperature=0)
        return resp.choices[0].message.content.strip()
    except OpenAIError as e:
        print(f"⚠️ Model {model_name} error. Fallback to text-only. Error: {e}")
        fallback_msgs = [{"role": "user", "content": prompt}]
        resp2 = safe_completion_call(client_obj, model_name, fallback_msgs, max_tokens=200, temperature=0)
        return resp2.choices[0].message.content.strip()

def generate_content_cached(model_name, question, image_path, qtype, choices=None):
    """
    Run model prediction with caching.
    Saves results so repeated runs don’t re-query API.
    """
    cache_file = f"results_cache_{model_name}_{qtype}.json"
    cache = load_cache(cache_file)
    key = cache_key(model_name, question, image_path, choices)
    if key in cache:
        return cache[key]
    result = generate_content(model_name, question, image_path, qtype, choices)
    cache[key] = result
    save_cache(cache_file, cache)
    return result

# ------------------------------
# Evaluation
# ------------------------------
def evaluate_split(model_id, split_set):
    """
    Evaluate a model on a given dataset split.
    Returns accuracy for TF, MC, QA and weighted average.
    Displays progress in Streamlit.
    """
    preds, gts, qtypes = [], [], []
    total_items = len(split_set)
    progress = st.progress(0)
    status = st.empty()

    for i, ex in enumerate(split_set):
        try:
            pred = generate_content_cached(
                model_id,
                ex["question"],
                ex["image_path"],
                ex["type"],
                ex.get("choices")
            )
        except OpenAIError as e:
            st.warning(f"⚠️ Skipping model {model_id} due to error: {e}")
            return None
        preds.append(pred)
        gts.append(ex["answer"])
        qtypes.append(ex["type"])
        progress.progress((i + 1) / total_items)
        status.text(f"Processed {i+1}/{total_items} examples...")
        time.sleep(0.05)

    # Split predictions by type
    tf_preds = [p for p, t in zip(preds, qtypes) if t == "TF"]
    tf_gts   = [g for g, t in zip(gts, qtypes) if t == "TF"]
    mc_preds = [p for p, t in zip(preds, qtypes) if t == "MC"]
    mc_gts   = [g for g, t in zip(gts, qtypes) if t == "MC"]
    qa_preds = [p for p, t in zip(preds, qtypes) if t == "QA"]
    qa_gts   = [g for g, t in zip(gts, qtypes) if t == "QA"]

    # Calculate accuracies
    tf_acc = acc(tf_preds, tf_gts, "TF") if tf_gts else 0
    mc_acc = acc(mc_preds, mc_gts, "MC") if mc_gts else 0
    qa_acc = acc(qa_preds, qa_gts, "QA") if qa_gts else 0

    # Weighted average across question types
    weighted_avg = (
        len(tf_gts) * tf_acc +
        len(mc_gts) * mc_acc +
        len(qa_gts) * qa_acc
    ) / total_items if total_items else 0

    return tf_acc, mc_acc, qa_acc, weighted_avg, len(tf_gts), len(mc_gts), len(qa_gts)

# ------------------------------
# Fine-Tuning code (same as earlier) ...
# ------------------------------
def write_jsonl(split, filename):
    """Convert dataset into JSONL format for fine-tuning."""
    with open(filename, "w") as f:
        for ex in split:
            img_name = os.path.basename(ex["image_path"])
            if ex["type"] == "MC":
                options_text = " | ".join(ex["choices"]) if ex["choices"] else ""
                question_with_img = f"[MC] Question: {ex['question']} (Refer to image: {img_name})\nOptions: {options_text}\nPlease answer with only A, B, C, or D."
                expected_answer = str(ex["answer"]).strip().upper()
            elif ex["type"] == "TF":
                question_with_img = f"[TF] Question: {ex['question']} (Refer to image: {img_name})\nPlease answer with only True or False."
                expected_answer = "True" if str(ex["answer"]).lower() in ["true", "1", "yes"] else "False"
            else:
                question_with_img = f"[QA] Question: {ex['question']} (Refer to image: {img_name})\nPlease answer with only a number."
                expected_answer = str(ex["answer"])
            msgs = [
                {"role": "system", "content": "You are an assistant that interprets financial charts."},
                {"role": "user", "content": question_with_img},
                {"role": "assistant", "content": expected_answer}
            ]
            f.write(json.dumps({"messages": msgs}) + "\n")


def launch_finetune(train_set, val_set, base_model=FINE_TUNE_BASE_MODEL):
    """
    Launch a fine-tuning job using OpenAI API.
    Prepares train/val sets in JSONL format and uploads them.
    """
    train_file = "train.jsonl"
    val_file = "val.jsonl"
    write_jsonl(train_set, train_file)
    write_jsonl(val_set, val_file)

    # Create API client
    upload_train = client = make_client_for_model(base_model)  # reuses model-specific client

    # Upload training and validation files to OpenAI
    upload_train = client.files.create(file=open(train_file, "rb"), purpose="fine-tune")
    upload_val = client.files.create(file=open(val_file, "rb"), purpose="fine-tune")

    # Create fine-tuning job
    job = client.fine_tuning.jobs.create(
        training_file=upload_train.id,
        validation_file=upload_val.id,
        model=base_model,
        suffix="finchart"
    )
    return job

def poll_finetune(job_id, poll_interval=15):
    """
    Poll the fine-tuning job status at fixed intervals.
    Shows live progress bar and status updates in Streamlit.
    """
    progress = st.progress(0)
    status = st.empty()
    step_label = st.empty()

    while True:
        job = client = make_client_for_model(FINE_TUNE_BASE_MODEL)
        job = job.fine_tuning.jobs.retrieve(job_id)  # Retrieve job status
        status.text(f"Job {job_id} status: {job.status}")

        try:
            # Get recent fine-tuning events
            events = client.fine_tuning.jobs.list_events(job_id=job_id, limit=50)
            for e in events.data:
                msg = getattr(e, "message", "")
                # Regex to parse progress messages like "Step 12/100"
                m = (re.search(r"Step\s+(\d+)\s*/\s*(\d+)", msg) or
                     re.search(r"(\d+)\s*/\s*(\d+)\s*steps", msg, re.I))
                if m:
                    step, total = int(m.group(1)), int(m.group(2))
                    percent = step / max(total, 1)
                    progress.progress(percent)
                    step_label.text(f"Training progress: {step}/{total} steps")
                    break
        except Exception:
            pass

        # Handle job completion or failure
        if job.status == "succeeded":
            progress.progress(1.0)
            st.success(f"✅ Fine-tuning complete! Model ID: {job.fine_tuned_model}")
            st.session_state["fine_tuned_model"] = job.fine_tuned_model
            break
        elif job.status == "failed":
            progress.progress(1.0)
            st.error("❌ Fine-tuning failed.")
            break
        elif job.status == "cancelled":
            progress.progress(1.0)
            st.warning("⚠️ Fine-tuning cancelled.")
            break

        time.sleep(poll_interval)

# ------------------------------
# Streamlit UI
# ------------------------------
st.title("FinChart-Bench – Baselines with Error Logging")

# Paths for dataset and results
test_folder = "C:/Users/sharm/Desktop/DSProjects/genai-agentiai-nlp/Project/test_folder"
MC_json = os.path.join(test_folder, "MC_data.json")
QA_json = os.path.join(test_folder, "QA_data.json")
TF_json = os.path.join(test_folder, "TF_data.json")
excel_path = os.path.join(test_folder, "finchart_baseline_results.xlsx")

# Load dataset (all examples, since limit=None)
unified = load_local_unified_dataset(test_folder, MC_json, QA_json, TF_json, limit=None)
total = len(unified)

# Split dataset: 70% train, 10% val, 20% test
n_train = int(0.7 * total)
n_val = int(0.1 * total)
train_set = unified[:n_train]
val_set = unified[n_train:n_train+n_val]
test_set = unified[n_train+n_val:]

st.write(f"Dataset split: {len(train_set)} train, {len(val_set)} val, {len(test_set)} test")

# Dropdown to pick a baseline model
selected_model = st.selectbox("Select Baseline Model", BASELINE_MODELS)

# Run a single selected model on test set
if st.button("Run Selected Baseline"):
    res = evaluate_split(selected_model, test_set)
    if res is None:
        st.error(f"Model {selected_model} skipped due to error.")
    else:
        tf_acc, mc_acc, qa_acc, weighted, tf_qs, mc_qs, qa_qs = res
        st.write(f"{selected_model} — TF: {tf_acc:.2%}, MC: {mc_acc:.2%}, QA: {qa_acc:.2%}, Weighted: {weighted:.2%}")
        row = {
            "Model": selected_model,
            "TF Score": tf_acc, "MC Score": mc_acc, "QA Score": qa_acc,
            "Weighted Avg": weighted,
            "Total Questions": len(test_set),
            "TF Qs": tf_qs, "MC Qs": mc_qs, "QA Qs": qa_qs,
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        upsert_results_excel(excel_path, row)

# Run all baseline models and produce leaderboard
if st.button("Run All Baseline Models"):
    leaderboard = []
    for model in BASELINE_MODELS:
        res = evaluate_split(model, test_set)
        if res is None:
            st.warning(f"Model {model} skipped due to error.")
            continue
        tf_acc, mc_acc, qa_acc, weighted, tf_qs, mc_qs, qa_qs = res
        st.write(f"{model} — TF: {tf_acc:.2%}, MC: {mc_acc:.2%}, QA: {qa_acc:.2%}, Weighted: {weighted:.2%}")
        row = {
            "Model": model,
            "TF Score": tf_acc, "MC Score": mc_acc, "QA Score": qa_acc,
            "Weighted Avg": weighted,
            "Total Questions": len(test_set),
            "TF Qs": tf_qs, "MC Qs": mc_qs, "QA Qs": qa_qs,
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        upsert_results_excel(excel_path, row)
        leaderboard.append((model, weighted))
    # Sort leaderboard by weighted accuracy
    leaderboard.sort(key=lambda x: x[1], reverse=True)
    st.write("🔝 Baseline Leaderboard")
    for m, sc in leaderboard:
        st.write(f"{m}: {sc:.2%}")

# Launch fine-tuning job
if st.button("Launch Fine-Tuning"):
    job = launch_finetune(train_set, val_set, base_model=FINE_TUNE_BASE_MODEL)
    st.session_state["job_id"] = job.id
    st.write(f"Fine-tuning launched: {job.id}")
    poll_finetune(job.id, poll_interval=15)

# Evaluate fine-tuned model on test set
if st.button("Evaluate Fine-Tuned"):
    if "fine_tuned_model" not in st.session_state:
        st.error("⚠️ Please fine-tune first.")
    else:
        model_id = st.session_state["fine_tuned_model"]
        res = evaluate_split(model_id, test_set)
        if res is None:
            st.error(f"Fine-tuned model {model_id} not accessible.")
        else:
            tf_acc, mc_acc, qa_acc, weighted, tf_qs, mc_qs, qa_qs = res
            st.success(f"Fine-tuned – TF: {tf_acc:.2%}, MC: {mc_acc:.2%}, QA: {qa_acc:.2%}")
            st.write(f"Weighted: {weighted:.2%}")
            row = {
                "Model": model_id,
                "TF Score": tf_acc, "MC Score": mc_acc, "QA Score": qa_acc,
                "Weighted Avg": weighted,
                "Total Questions": len(test_set),
                "TF Qs": tf_qs, "MC Qs": mc_qs, "QA Qs": qa_qs,
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            df_out = upsert_results_excel(excel_path, row)
            st.success(f"Saved fine-tuned results to {excel_path}")
            st.dataframe(df_out)
