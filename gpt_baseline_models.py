# baseline_with_error_logging.py
# -------------------------------------------------
# Purpose:
#   - Run baseline experiments on financial chart tasks (TF, MC, QA).
#   - Evaluate GPT/Gemini models with caching + error handling.
#   - Export results to Excel and CSV for ensembling.
#   - Support fine-tuning and evaluation inside a Streamlit UI.
# -------------------------------------------------

import os, json, time, re, random, base64, mimetypes, traceback, hashlib
import streamlit as st
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError, OpenAIError
from PIL import Image
from io import BytesIO

# ------------------------------
# Setup & API Keys
# ------------------------------
load_dotenv()  # Load API keys from .env file
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Fix random seed for reproducibility
SEED = 42
random.seed(SEED)

# ------------------------------
# Model Config
# ------------------------------
# Baseline models we want to test
BASELINE_MODELS = [
    "gpt-4o-2024-08-06",
    "gpt-4o-mini",
    "gpt-4.1-2025-04-14",
]

# Default model used for fine-tuning experiments
FINE_TUNE_BASE_MODEL = "gpt-4o-2024-08-06"

# ------------------------------
# Helpers & Normalization
# ------------------------------
def safe_float(x):
    """
    Extract numeric value from messy string.
    Useful for QA tasks where answers may contain commas, symbols, etc.
    """
    try:
        numbers = re.findall(r"[-+]?\d*\.?\d+", str(x).replace(",", ""))
        if not numbers:
            return None
        return float(numbers[-1])  # take last number
    except Exception:
        return None

def normalize_tf(pred):
    """
    Normalize True/False predictions into 'true' or 'false'.
    Handles variations like 'T', 'F', 'yes', 'false123', etc.
    """
    p = str(pred).lower()
    p = re.sub(r'[^a-z]', '', p)  # remove punctuation/whitespace
    if p.startswith('true') or p == 't': return 'true'
    if p.startswith('false') or p == 'f': return 'false'
    if 'true' in p: return 'true'
    if 'false' in p: return 'false'
    return p

def normalize_mc(pred):
    """
    Normalize Multiple Choice predictions into 'a', 'b', 'c', or 'd'.
    Handles extra text like 'Answer: A', 'option B', etc.
    """
    p = str(pred).lower()
    m = re.search(r'\b([abcd])\b', p)  # look for single letter
    if m: return m.group(1)
    p2 = re.sub(r'[^a-d]', '', p)  # remove junk characters
    if p2 and p2[0] in 'abcd': return p2[0]
    return p.strip()

def acc(predicts, ground_truth, question_type):
    """
    Compute accuracy between predictions and ground truth.
    Handles TF, MC, and QA separately.
    """
    correct = 0
    for p, gt in zip(predicts, ground_truth):
        if question_type == "QA":
            p, gt = safe_float(p), safe_float(gt)
            if p is None or gt is None:
                continue
        elif question_type == "TF":
            p, gt = normalize_tf(p), normalize_tf(gt)
        else:  # MC
            p, gt = normalize_mc(p), normalize_mc(gt)
        if p == gt:
            correct += 1
    return correct / len(ground_truth) if ground_truth else 0

def upsert_results_excel(excel_path, row_dict, key_col="Model"):
    """
    Save results into an Excel file.
    - If model already exists, update its row.
    - Otherwise, append a new row.
    """
    cols = ["Model", "TF Score", "MC Score", "QA Score",
            "Weighted Avg", "Total Questions",
            "TF Qs", "MC Qs", "QA Qs", "Timestamp"]

    if os.path.exists(excel_path):
        try:
            df = pd.read_excel(excel_path)
        except Exception:
            df = pd.DataFrame(columns=cols)
    else:
        df = pd.DataFrame(columns=cols)

    # Ensure all required columns exist
    for c in cols:
        if c not in df.columns:
            df[c] = pd.Series(dtype="object")

    # Update row if exists, otherwise append
    mask = df[key_col] == row_dict[key_col]
    if mask.any():
        df.loc[mask, cols] = [row_dict.get(c) for c in cols]
    else:
        df = pd.concat([df, pd.DataFrame([row_dict], columns=cols)], ignore_index=True)

    df.to_excel(excel_path, index=False)
    return df

def clean_image_name(img_name: str) -> str:
    """
    Standardize image names by removing q1/q2 suffixes.
    Example: chart_q1.jpg → chart.jpg
    """
    if not isinstance(img_name, str):
        return img_name
    return (img_name.replace("_q1.jpg", ".jpg")
                  .replace("_q2.jpg", ".jpg")
                  .replace("_Q1.jpg", ".jpg")
                  .replace("_Q2.jpg", ".jpg"))

def load_local_unified_dataset(test_folder, MC_json, QA_json, TF_json, limit=1000):
    """
    Load dataset from JSON files and unify into a single list.
    - Each item contains image_path, question, answer, choices, type.
    - Randomly shuffled and truncated to `limit`.
    """
    unified = []

    def add_items(json_path, qtype, img_subfolder):
        if json_path and os.path.exists(json_path):
            with open(json_path, "r") as f:
                items = json.load(f)
            for item in items:
                img_name = clean_image_name(item.get("image"))
                question, answer = item.get("question"), item.get("answer")
                choices = item.get("choices") if qtype == "MC" else None
                img_path = os.path.join(test_folder, img_subfolder, img_name)
                if not os.path.exists(img_path):
                    st.warning(f"⚠️ Missing image {img_name}")
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
    return unified[:limit] if limit else unified

# ------------------------------
# Cache Helpers
# ------------------------------
def cache_key(model_name, question, image_path, choices=None):
    """Generate unique key for caching predictions."""
    data = f"{model_name}-{question}-{image_path}-{choices}"
    return hashlib.md5(data.encode()).hexdigest()

def load_cache(cache_file):
    """Load predictions from cache file if exists."""
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_cache(cache_file, cache):
    """Save predictions to cache file."""
    with open(cache_file, "w") as f:
        json.dump(cache, f, indent=2)

# ------------------------------
# Client Factory
# ------------------------------
def make_client_for_model(model_name: str):
    """Return API client (OpenAI or Gemini)."""
    if model_name.startswith("gemini"):
        return OpenAI(api_key=GEMINI_API_KEY,
                      base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
    return OpenAI(api_key=OPENAI_API_KEY)

# ------------------------------
# Safe API Call (with retries + logging)
# ------------------------------
def safe_completion_call(client_obj, model_name, messages, max_tokens=200, temperature=0):
    """
    Wrapper to safely call the API.
    Retries up to 5 times in case of errors or rate limits.
    """
    for attempt in range(5):
        try:
            return client_obj.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
        except RateLimitError:
            wait = (attempt + 1) * 2
            print(f"⚠️ Rate limit hit. Retrying in {wait}s.")
            time.sleep(wait)
        except OpenAIError as e:
            print(f"❌ API Error on {model_name}: {e}")
            traceback.print_exc()
            if "model_not_found" in str(e).lower():  # unrecoverable error
                raise
            time.sleep((attempt + 1) * 2)
    raise RuntimeError(f"Failed call for model {model_name} after retries")

# ------------------------------
# Inference with Fallback
# ------------------------------
def generate_content(model_name, question, image_path, qtype=None, choices=None):
    """
    Generate answer from a model using text + image.
    If vision call fails, fallback to text-only mode.
    """
    # Build task-specific prompt
    if qtype == "MC":
        options_text = " | ".join(choices) if choices else ""
        prompt = f"[MC] Question: {question}\nOptions: {options_text}\nAnswer with only A, B, C, or D."
    elif qtype == "TF":
        prompt = f"[TF] Question: {question}\nAnswer with only True or False."
    else:
        prompt = f"[QA] Question: {question}\nAnswer with only a number."

    # Encode image as base64
    mime_type, _ = mimetypes.guess_type(image_path)
    if not mime_type: mime_type = "image/png"
    with open(image_path, "rb") as f: img_data = f.read()
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
        resp = safe_completion_call(client_obj, model_name, messages)
        return resp.choices[0].message.content.strip()
    except OpenAIError:
        # Fallback to text-only
        fallback_msgs = [{"role": "user", "content": prompt}]
        resp2 = safe_completion_call(client_obj, model_name, fallback_msgs)
        return resp2.choices[0].message.content.strip()

def generate_content_cached(model_name, question, image_path, qtype, choices=None):
    """
    Cached wrapper around generate_content().
    Saves and loads results to avoid repeated billing.
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
# Export Predictions (for ensembling)
# ------------------------------
def evaluate_and_dump_preds(model, split_set, out_csv):
    """
    Run inference for a given model and export predictions to CSV.
    Used later for ensembling.
    """
    rows = []
    for i, ex in enumerate(split_set):
        pred = generate_content_cached(model, ex["question"], ex["image_path"], ex["type"], ex.get("choices"))
        rows.append({"idx": i, "type": ex["type"], "answer": ex["answer"], f"{model}_pred": pred})
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    return df

# ------------------------------
# Evaluation
# ------------------------------
def evaluate_split(model_id, split_set):
    """
    Evaluate a single model on a dataset split.
    Shows progress bar and returns per-task accuracy.
    """
    preds, gts, qtypes = [], [], []
    total_items = len(split_set)
    progress, status = st.progress(0), st.empty()

    for i, ex in enumerate(split_set):
        try:
            pred = generate_content_cached(model_id, ex["question"], ex["image_path"], ex["type"], ex.get("choices"))
        except OpenAIError as e:
            st.warning(f"⚠️ Skipping model {model_id} due to error: {e}")
            return None
        preds.append(pred); gts.append(ex["answer"]); qtypes.append(ex["type"])
        progress.progress((i + 1) / total_items)
        status.text(f"Processed {i+1}/{total_items} examples...")
        time.sleep(0.05)

    # Split results by task
    tf_acc = acc([p for p, t in zip(preds, qtypes) if t == "TF"],
                 [g for g, t in zip(gts, qtypes) if t == "TF"], "TF")
    mc_acc = acc([p for p, t in zip(preds, qtypes) if t == "MC"],
                 [g for g, t in zip(gts, qtypes) if t == "MC"], "MC")
    qa_acc = acc([p for p, t in zip(preds, qtypes) if t == "QA"],
                 [g for g, t in zip(gts, qtypes) if t == "QA"], "QA")

    weighted_avg = ((qtypes.count("TF") * tf_acc +
                     qtypes.count("MC") * mc_acc +
                     qtypes.count("QA") * qa_acc) / total_items)

    return tf_acc, mc_acc, qa_acc, weighted_avg, qtypes.count("TF"), qtypes.count("MC"), qtypes.count("QA")

# ------------------------------
# Fine-Tuning Helpers
# ------------------------------
def write_jsonl(split, filename):
    """
    Convert dataset split into JSONL format for fine-tuning.
    Each row contains system, user, and assistant messages.
    """
    with open(filename, "w") as f:
        for ex in split:
            img_name = os.path.basename(ex["image_path"])
            if ex["type"] == "MC":
                options_text = " | ".join(ex["choices"]) if ex["choices"] else ""
                question_with_img = f"[MC] Question: {ex['question']} (Refer to image: {img_name})\nOptions: {options_text}"
                expected_answer = str(ex["answer"]).strip().upper()
            elif ex["type"] == "TF":
                question_with_img = f"[TF] Question: {ex['question']} (Refer to image: {img_name})"
                expected_answer = "True" if str(ex["answer"]).lower() in ["true", "1", "yes"] else "False"
            else:
                question_with_img = f"[QA] Question: {ex['question']} (Refer to image: {img_name})"
                expected_answer = str(ex["answer"])
            msgs = [{"role": "system", "content": "You are an assistant that interprets financial charts."},
                    {"role": "user", "content": question_with_img},
                    {"role": "assistant", "content": expected_answer}]
            f.write(json.dumps({"messages": msgs}) + "\n")

def launch_finetune(train_set, val_set, base_model=FINE_TUNE_BASE_MODEL):
    """
    Launch fine-tuning job on OpenAI API.
    """
    train_file, val_file = "train.jsonl", "val.jsonl"
    write_jsonl(train_set, train_file)
    write_jsonl(val_set, val_file)
    client = make_client_for_model(base_model)
    upload_train = client.files.create(file=open(train_file, "rb"), purpose="fine-tune")
    upload_val = client.files.create(file=open(val_file, "rb"), purpose="fine-tune")
    return client.fine_tuning.jobs.create(training_file=upload_train.id,
                                          validation_file=upload_val.id,
                                          model=base_model, suffix="finchart")

def poll_finetune(job_id, poll_interval=15):
    """
    Monitor fine-tuning job in real-time with progress bar.
    """
    progress, status, step_label = st.progress(0), st.empty(), st.empty()
    while True:
        client = make_client_for_model(FINE_TUNE_BASE_MODEL)
        job = client.fine_tuning.jobs.retrieve(job_id)
        status.text(f"Job {job_id} status: {job.status}")
        if job.status == "succeeded":
            st.success(f"✅ Fine-tuning complete! {job.fine_tuned_model}")
            st.session_state["fine_tuned_model"] = job.fine_tuned_model
            break
        elif job.status in ["failed", "cancelled"]:
            st.error(f"❌ Fine-tuning {job.status}.")
            break
        time.sleep(poll_interval)

# ------------------------------
# Streamlit UI
# ------------------------------
st.title("FinChart-Bench – Baselines with Error Logging")

# Dataset setup
test_folder = "C:/Users/sharm/Desktop/DSProjects/genai-agentiai-nlp/Project/test_folder"
MC_json, QA_json, TF_json = [os.path.join(test_folder, f"{x}_data.json") for x in ["MC", "QA", "TF"]]
excel_path = os.path.join(test_folder, "finchart_baseline_results.xlsx")

# Load dataset and split
unified = load_local_unified_dataset(test_folder, MC_json, QA_json, TF_json, limit=1000)
n_train, n_val = int(0.7 * len(unified)), int(0.1 * len(unified))
train_set, val_set, test_set = unified[:n_train], unified[n_train:n_train+n_val], unified[n_train+n_val:]
st.write(f"Dataset split: {len(train_set)} train, {len(val_set)} val, {len(test_set)} test")

# Select model
selected_model = st.selectbox("Select Baseline Model", BASELINE_MODELS)

# Run selected model
if st.button("Run Selected Baseline"):
    res = evaluate_split(selected_model, test_set)
    if res:
        tf, mc, qa, w, tfq, mcq, qaq = res
        st.write(f"{selected_model} — TF: {tf:.2%}, MC: {mc:.2%}, QA: {qa:.2%}, Weighted: {w:.2%}")

        # Save results to Excel for later analysis
        row = {
            "Model": selected_model,
            "TF Score": tf, "MC Score": mc, "QA Score": qa,
            "Weighted Avg": w,
            "Total Questions": len(test_set),
            "TF Qs": tfq, "MC Qs": mcq, "QA Qs": qaq,
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        upsert_results_excel(excel_path, row)
    else:
        st.error(f"Model {selected_model} skipped due to error.")

# Export predictions for GPT models (for ensembling later)
if st.button("Export preds for GPT ensemble models"):
    for m in ["gpt-4o-mini", "gpt-4.1-2025-04-14"]:
        out_csv = f"preds_{m}.csv"
        df_preds = evaluate_and_dump_preds(m, test_set, out_csv)
        st.write(f"Saved {out_csv} ({len(df_preds)} rows)")
    st.success("Exported GPT predictions for ensembling.")

# Run all baseline models (Leaderboard mode)
if st.button("Run All Baseline Models"):
    leaderboard = []
    for model in BASELINE_MODELS:
        res = evaluate_split(model, test_set)
        if res is None:
            st.warning(f"⚠️ Skipping {model} due to error.")
            continue

        tf, mc, qa, w, tfq, mcq, qaq = res
        st.write(f"{model} — TF: {tf:.2%}, MC: {mc:.2%}, QA: {qa:.2%}, Weighted: {w:.2%}")

        row = {
            "Model": model,
            "TF Score": tf, "MC Score": mc, "QA Score": qa,
            "Weighted Avg": w,
            "Total Questions": len(test_set),
            "TF Qs": tfq, "MC Qs": mcq, "QA Qs": qaq,
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        upsert_results_excel(excel_path, row)
        leaderboard.append((model, w))

    # Sort models by weighted score and display leaderboard
    leaderboard.sort(key=lambda x: x[1], reverse=True)
    st.write("🔝 Baseline Leaderboard")
    for m, sc in leaderboard:
        st.write(f"{m}: {sc:.2%}")

# Launch fine-tuning job on the chosen base model
if st.button("Launch Fine-Tuning"):
    job = launch_finetune(train_set, val_set, base_model=FINE_TUNE_BASE_MODEL)
    st.session_state["job_id"] = job.id
    st.write(f"Fine-tuning launched: {job.id}")
    poll_finetune(job.id, poll_interval=15)

# Evaluate fine-tuned model
if st.button("Evaluate Fine-Tuned"):
    if "fine_tuned_model" not in st.session_state:
        st.error("⚠️ Please fine-tune first.")
    else:
        model_id = st.session_state["fine_tuned_model"]
        res = evaluate_split(model_id, test_set)
        if res:
            tf, mc, qa, w, tfq, mcq, qaq = res
            st.success(f"Fine-tuned – TF: {tf:.2%}, MC: {mc:.2%}, QA: {qa:.2%}, Weighted: {w:.2%}")

            row = {
                "Model": model_id,
                "TF Score": tf, "MC Score": mc, "QA Score": qa,
                "Weighted Avg": w,
                "Total Questions": len(test_set),
                "TF Qs": tfq, "MC Qs": mcq, "QA Qs": qaq,
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            df_out = upsert_results_excel(excel_path, row)
            st.success(f"Saved fine-tuned results to {excel_path}")
            st.dataframe(df_out)
        else:
            st.error(f"⚠️ Fine-tuned model {model_id} not accessible.")

