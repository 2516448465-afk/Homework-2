import json
from pathlib import Path

import jieba
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "corpora" / "douban.json"
MODEL_PATH = ROOT / "outputs" / "sentiment_model.joblib"


def tokenize(text):
    return [word for word in jieba.lcut(text) if word.strip()]


def build_dataset():
    with DATA_PATH.open(encoding="utf-8") as data_file:
        reviews = json.load(data_file)

    by_star = {1: [], 5: []}
    for review in reviews:
        star = review["label"]
        if star in by_star:
            by_star[star].append(review["text"])

    train_texts = []
    train_labels = []
    test_texts = []
    test_labels = []

    for star, sentiment in ((1, 0), (5, 1)):
        if len(by_star[star]) < 6000:
            raise ValueError(f"Need at least 6,000 reviews with label {star}")

        class_train, class_test = train_test_split(
            by_star[star],
            train_size=5000,
            test_size=1000,
            random_state=42,
        )
        train_texts.extend(class_train)
        train_labels.extend([sentiment] * len(class_train))
        test_texts.extend(class_test)
        test_labels.extend([sentiment] * len(class_test))

    return train_texts, train_labels, test_texts, test_labels


def main():
    train_texts, train_labels, test_texts, test_labels = build_dataset()

    vectorizer = TfidfVectorizer(
        tokenizer=tokenize,
        token_pattern=None,
        lowercase=False,
    )
    train_vectors = vectorizer.fit_transform(train_texts)
    test_vectors = vectorizer.transform(test_texts)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(train_vectors, train_labels)

    accuracy = accuracy_score(test_labels, model.predict(test_vectors))
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"vectorizer": vectorizer, "model": model}, MODEL_PATH)

    terms = vectorizer.get_feature_names_out()
    weights = model.coef_[0]
    positive_indices = weights.argsort()[-20:][::-1]
    negative_indices = weights.argsort()[:20]

    print(f"Test accuracy ({len(test_labels)} reviews): {accuracy:.4f}")
    print(f"Saved model: {MODEL_PATH.relative_to(ROOT)}")
    print("\n20 words with the most positive weights:")
    for index in positive_indices:
        print(f"{terms[index]}\t{weights[index]:.4f}")
    print("\n20 words with the most negative weights:")
    for index in negative_indices:
        print(f"{terms[index]}\t{weights[index]:.4f}")


if __name__ == "__main__":
    main()