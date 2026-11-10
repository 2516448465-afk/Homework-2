import csv
import re
from pathlib import Path

import jieba
import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TEXT_PATH = ROOT / "corpora" / "牵风记-徐怀中.txt"
MODEL_PATH = ROOT / "outputs" / "sentiment_model.joblib"
CSV_PATH = ROOT / "outputs" / "sentiment_arc.csv"
PLOT_PATH = ROOT / "outputs" / "sentiment_arc.png"


def tokenize(text):
    return [word for word in jieba.lcut(text) if word.strip()]


def split_passages(text, target_chars=500):
    sentences = re.split(r"(?<=[。！？])", text)
    passages = []
    current = ""

    for sentence in sentences:
        if not sentence.strip():
            continue
        if current and len(current) + len(sentence) > target_chars:
            passages.append(current.strip())
            current = sentence
        else:
            current += sentence

    if current.strip():
        passages.append(current.strip())
    return passages


def main():
    text = TEXT_PATH.read_text(encoding="utf-8")
    passages = split_passages(text)
    bundle = joblib.load(MODEL_PATH)
    vectorizer = bundle["vectorizer"]
    model = bundle["model"]

    passage_vectors = vectorizer.transform(passages)
    probabilities = model.predict_proba(passage_vectors)
    positive_index = list(model.classes_).index(1)
    negative_index = list(model.classes_).index(0)
    scores = probabilities[:, positive_index] - probabilities[:, negative_index]

    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["passage_number", "score", "text"])
        writer.writerows(
            (number, f"{score:.8f}", passage)
            for number, (score, passage) in enumerate(zip(scores, passages), 1)
        )

    passage_numbers = np.arange(1, len(scores) + 1)
    figure, axis = plt.subplots(figsize=(12, 5))
    axis.plot(passage_numbers, scores, alpha=0.45, linewidth=0.8, label="Score")
    if len(scores) >= 10:
        moving_average = np.convolve(scores, np.ones(10) / 10, mode="valid")
        axis.plot(
            passage_numbers[9:],
            moving_average,
            linewidth=2,
            label="10-passage moving average",
        )
    axis.axhline(0, color="black", linewidth=1, linestyle="--")
    axis.set_xlabel("Passage number")
    axis.set_ylabel("P(positive) - P(negative)")
    axis.set_title("Sentiment arc")
    axis.legend()
    figure.tight_layout()
    figure.savefig(PLOT_PATH, dpi=160)
    plt.close(figure)

    print(f"Passages: {len(passages)}")
    print(f"Saved CSV: {CSV_PATH.relative_to(ROOT)}")
    print(f"Saved plot: {PLOT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()