import argparse
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


def extract_hulanhe_body(text):
    lines = text.splitlines()
    preface_end = next(
        (index for index, line in enumerate(lines) if line.strip() == "一九四六年八月于上海"),
        None,
    )
    if preface_end is None:
        raise ValueError("Could not find the end of the preface")

    chapter_titles = [
        index
        for index, line in enumerate(lines)
        if index > preface_end and line.strip() == "第一章"
    ]
    if len(chapter_titles) < 2:
        raise ValueError("Could not find the start of the novel body")

    colophon = next(
        (
            index
            for index, line in enumerate(lines)
            if line.strip().startswith("一九四○年十二月二十日香港完稿")
        ),
        None,
    )
    if colophon is None:
        raise ValueError("Could not find the end of the novel body")

    body_lines = lines[chapter_titles[1] + 2 : colophon]
    headings = re.compile(r"(?:第[一二三四五六七]章|尾声|[一二三四五六七])")
    return "\n".join(line for line in body_lines if not headings.fullmatch(line.strip()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=Path, default=TEXT_PATH)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--hulanhe-body", action="store_true")
    args = parser.parse_args()

    text_path = args.text if args.text.is_absolute() else ROOT / args.text
    model_path = args.model if args.model.is_absolute() else ROOT / args.model
    output_dir = (
        args.output_dir
        if args.output_dir.is_absolute()
        else ROOT / args.output_dir
    )
    csv_path = output_dir / "sentiment_arc.csv"
    plot_path = output_dir / "sentiment_arc.png"

    text = text_path.read_text(encoding="utf-8")
    if args.hulanhe_body:
        text = extract_hulanhe_body(text)
    passages = split_passages(text)
    bundle = joblib.load(model_path)
    vectorizer = bundle["vectorizer"]
    model = bundle["model"]

    passage_vectors = vectorizer.transform(passages)
    probabilities = model.predict_proba(passage_vectors)
    positive_index = list(model.classes_).index(1)
    negative_index = list(model.classes_).index(0)
    scores = probabilities[:, positive_index] - probabilities[:, negative_index]

    output_dir.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
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
    figure.savefig(plot_path, dpi=160)
    plt.close(figure)

    print(f"Passages: {len(passages)}")
    print(f"Saved CSV: {csv_path.relative_to(ROOT)}")
    print(f"Saved plot: {plot_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()