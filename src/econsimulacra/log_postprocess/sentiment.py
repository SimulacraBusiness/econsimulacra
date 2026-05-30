from __future__ import annotations

from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .base import LogPostProcessor

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()

LABEL2SCORE = {
    "negative": -1.0,
    "neutral": 0.0,
    "positive": 1.0,
}


def predict_sentiment(text: str) -> dict[str, Any]:
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=-1)[0]
    label_id = int(torch.argmax(probs).item())
    confidence = float(probs[label_id].item())

    label = model.config.id2label[label_id].lower()
    sentiment = LABEL2SCORE[label] * confidence

    return {
        "sentiment": sentiment,
        "sentiment_label": label,
        "sentiment_confidence": confidence,
    }


def add_tweet_sentiment(tweet_log: dict[str, Any]) -> dict[str, Any]:
    assert tweet_log["type"] == "tweet", (
        "add_tweet_sentiment should only be applied to tweet logs"
    )
    text = str(tweet_log["message"])
    tweet_log.update(predict_sentiment(text))
    return tweet_log


def add_inner_thought_sentiment(inner_thought_log: dict[str, Any]) -> dict[str, Any]:
    assert inner_thought_log["type"] == "inner_thought", (
        "add_inner_thought_sentiment should only be applied to inner_thought logs"
    )
    text = str(inner_thought_log["inner_thought"])
    inner_thought_log.update(predict_sentiment(text))
    return inner_thought_log


def build_sentiment_processor() -> LogPostProcessor:
    processor = LogPostProcessor()
    processor.add_processor("tweet", add_tweet_sentiment)
    processor.add_processor("inner_thought", add_inner_thought_sentiment)
    return processor
