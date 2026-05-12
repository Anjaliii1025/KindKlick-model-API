import gc
import math
import re
from pathlib import Path
from urllib.parse import urlparse

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "Models"

TEXT_MODEL_PATH = MODELS_DIR / "text_model.pkl"
URL_FEATURE_MODEL_PATH = MODELS_DIR / "url_feature_model.pkl"
URL_FEATURE_COLUMNS_PATH = MODELS_DIR / "url_feature_columns.pkl"
URL_TEXT_MODEL_PATH = MODELS_DIR / "url_text_model.pkl"


# Lazy-loaded globals
text_model = None
feature_model = None
feature_columns = None
text_url_model = None


# -----------------------------
# Lazy Loading Functions
# -----------------------------
def get_text_model():
    global text_model

    if text_model is None:
        text_model = joblib.load(TEXT_MODEL_PATH)

    return text_model


def get_feature_model():
    global feature_model

    if feature_model is None:
        feature_model = joblib.load(URL_FEATURE_MODEL_PATH)

    return feature_model


def get_feature_columns():
    global feature_columns

    if feature_columns is None:
        feature_columns = joblib.load(URL_FEATURE_COLUMNS_PATH)

    return feature_columns


def get_text_url_model():
    global text_url_model

    if text_url_model is None:
        text_url_model = joblib.load(URL_TEXT_MODEL_PATH)

    return text_url_model


# -----------------------------
# FastAPI App
# -----------------------------
app = FastAPI(title="KindKlick Safety API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


TRUSTED_HOSTS = {
    "accounts.google.com",
    "mail.google.com",
    "github.com",
    "docs.github.com",
    "www.youtube.com",
    "youtube.com",
    "www.wikipedia.org",
    "en.wikipedia.org",
    "www.paypal.com",
    "paypal.com",
    "www.instagram.com",
    "instagram.com",
    "www.amazon.in",
    "amazon.in",
    "login.live.com",
    "www.google.com",
    "google.com",
    "www.facebook.com",
    "facebook.com",
    "www.linkedin.com",
    "linkedin.com",
    "www.dropbox.com",
    "dropbox.com",
    "zoom.us",
    "support.microsoft.com",
    "www.microsoft.com",
    "microsoft.com",
}

SUSPICIOUS_TLDS = {".xyz", ".tk", ".ml", ".ga", ".cf", ".gq"}


# -----------------------------
# Request Models
# -----------------------------
class TextRequest(BaseModel):
    text: str
    threshold: float = 0.55


class UrlRequest(BaseModel):
    url: str
    threshold: float = 0.80
    suspicious_tld_threshold: float = 0.65
    feature_weight: float = 0.5
    text_weight: float = 0.5


# -----------------------------
# Helper Functions
# -----------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def normalize_url(url):
    url = str(url).strip().lower()
    url = re.sub(r"\s+", "", url)

    return url


def has_suspicious_tld(host):
    return any(host.endswith(tld) for tld in SUSPICIOUS_TLDS)


def extract_features(url):
    parsed = urlparse(url)

    hostname = parsed.netloc
    path = parsed.path

    shorteners = [
        "bit.ly",
        "tinyurl.com",
        "t.co",
        "goo.gl",
        "is.gd",
        "buff.ly",
        "ow.ly",
        "rb.gy",
    ]

    probs = [url.count(c) / len(url) for c in set(url)] if url else [1]

    return pd.Series({
        "url_length": len(url),
        "hostname_length": len(hostname),
        "path_length": len(path),
        "dot_count": url.count("."),
        "hostname_dot_count": hostname.count("."),
        "hyphen_count": url.count("-"),
        "slash_count": url.count("/"),
        "digit_count": sum(c.isdigit() for c in url),
        "special_char_count": len(re.findall(r"[@#&=%]", url)),
        "subdomain_count": max(hostname.count(".") - 1, 0),
        "path_segment_count": len([p for p in path.split("/") if p]),
        "has_ip": int(bool(re.search(r"\d+\.\d+\.\d+\.\d+", hostname))),
        "has_https": int(parsed.scheme == "https"),
        "has_at_symbol": int("@" in url),
        "has_double_slash_path": int("//" in path),
        "has_suspicious_tld": int(
            any(hostname.endswith(tld) for tld in SUSPICIOUS_TLDS)
        ),
        "has_shortener": int(
            any(short in hostname for short in shorteners)
        ),
        "entropy": -sum(p * math.log2(p) for p in probs),
    })


# -----------------------------
# Routes
# -----------------------------
@app.get("/")
def root():
    return {"message": "KindKlick Safety API is running"}


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/analyze/text")
def analyze_text(request: TextRequest):

    model = get_text_model()

    cleaned = clean_text(request.text)

    probs = model.predict_proba([cleaned])[0]
    labels = list(model.classes_)

    top_index = int(probs.argmax())

    top_label = labels[top_index]
    top_score = float(probs[top_index])

    result = (
        "needs_review"
        if top_score < request.threshold
        else top_label
    )

    gc.collect()

    return {
        "result": result,
        "top_label": top_label,
        "confidence": round(top_score, 4),
        "scores": {
            str(labels[i]): round(float(probs[i]), 4)
            for i in range(len(labels))
        },
    }


@app.post("/api/analyze/url")
def analyze_url(request: UrlRequest):

    feature_cols = get_feature_columns()
    feature_model_instance = get_feature_model()
    text_url_model_instance = get_text_url_model()

    normalized_url = normalize_url(request.url)

    host = urlparse(normalized_url).netloc.lower()

    feature_row = extract_features(normalized_url)

    feature_row = feature_row.reindex(
        feature_cols,
        fill_value=0
    )

    feature_prob = float(
        feature_model_instance.predict_proba(
            pd.DataFrame([feature_row])
        )[0][1]
    )

    text_prob = float(
        text_url_model_instance.predict_proba(
            [normalized_url]
        )[0][1]
    )

    final_prob = (
        request.feature_weight * feature_prob
        + request.text_weight * text_prob
    )

    if host in TRUSTED_HOSTS:
        result = "safe"
        decision_reason = "trusted_host"

    elif (
        has_suspicious_tld(host)
        and final_prob >= request.suspicious_tld_threshold
    ):
        result = "phishing"
        decision_reason = "suspicious_tld_rule"

    else:
        result = (
            "phishing"
            if final_prob >= request.threshold
            else "safe"
        )

        decision_reason = "hybrid_score"

    gc.collect()

    return {
        "url": request.url,
        "host": host,
        "result": result,
        "decision_reason": decision_reason,
        "feature_probability": round(feature_prob, 4),
        "text_probability": round(text_prob, 4),
        "final_probability": round(final_prob, 4),
    }
