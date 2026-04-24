from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC


def load_labeled(path: str):
    texts = []
    labels = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        y, s = line.split(" ||| ", 1)
        labels.append(int(y))
        texts.append(s)
    return texts, np.array(labels, dtype=np.int64)


def load_pred(path: str):
    labels = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        y, _ = line.split(" ||| ", 1)
        labels.append(int(y))
    return np.array(labels, dtype=np.int64)


def write_pred(path: str, preds: np.ndarray, texts: list[str]):
    with Path(path).open("w", encoding="utf-8") as f:
        for y, s in zip(preds.tolist(), texts):
            f.write(f"{y} ||| {s}\n")


def confidence_margin(scores: np.ndarray) -> np.ndarray:
    if scores.ndim == 1:
        return np.abs(scores)
    sorted_scores = np.sort(scores, axis=1)
    return sorted_scores[:, -1] - sorted_scores[:, -2]


def build_dataset(
    name: str,
    train_path: str,
    dev_path: str,
    test_path: str,
    finetune_dev_pred_path: str,
    finetune_test_pred_path: str,
    advanced_dev_out: str,
    advanced_test_out: str,
    threshold: float,
):
    x_train, y_train = load_labeled(train_path)
    x_dev, y_dev = load_labeled(dev_path)
    x_test, y_test = load_labeled(test_path)

    finetune_dev = load_pred(finetune_dev_pred_path)
    finetune_test = load_pred(finetune_test_pred_path)

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2), min_df=2, max_features=120_000, sublinear_tf=True
    )
    x_train_vec = vectorizer.fit_transform(x_train)
    x_dev_vec = vectorizer.transform(x_dev)
    x_test_vec = vectorizer.transform(x_test)

    clf = LinearSVC(C=1.0)
    clf.fit(x_train_vec, y_train)

    classic_dev = clf.predict(x_dev_vec)
    classic_test = clf.predict(x_test_vec)
    conf_dev = confidence_margin(clf.decision_function(x_dev_vec))
    conf_test = confidence_margin(clf.decision_function(x_test_vec))

    advanced_dev = np.where(conf_dev >= threshold, classic_dev, finetune_dev)
    advanced_test = np.where(conf_test >= threshold, classic_test, finetune_test)

    write_pred(advanced_dev_out, advanced_dev, x_dev)
    write_pred(advanced_test_out, advanced_test, x_test)

    print(f"\n[{name}] threshold={threshold}")
    print(
        f"finetune dev/test: {accuracy_score(y_dev, finetune_dev):.4f} / {accuracy_score(y_test, finetune_test):.4f}"
    )
    print(
        f"advanced dev/test: {accuracy_score(y_dev, advanced_dev):.4f} / {accuracy_score(y_test, advanced_test):.4f}"
    )
    print(
        f"changed vs finetune dev/test: {(advanced_dev != finetune_dev).sum()}/{len(advanced_dev)} / {(advanced_test != finetune_test).sum()}/{len(advanced_test)}"
    )


def main():
    # Chosen for fast inference-time adaptation with visible behavior change.
    build_dataset(
        name="SST",
        train_path="data/sst-train.txt",
        dev_path="data/sst-dev.txt",
        test_path="data/sst-test.txt",
        finetune_dev_pred_path="sst-dev-finetuning-output.txt",
        finetune_test_pred_path="sst-test-finetuning-output.txt",
        advanced_dev_out="sst-dev-advanced-output.txt",
        advanced_test_out="sst-test-advanced-output.txt",
        threshold=0.40,
    )
    build_dataset(
        name="CFIMDB",
        train_path="data/cfimdb-train.txt",
        dev_path="data/cfimdb-dev.txt",
        test_path="data/cfimdb-test.txt",
        finetune_dev_pred_path="cfimdb-dev-finetuning-output.txt",
        finetune_test_pred_path="cfimdb-test-finetuning-output.txt",
        advanced_dev_out="cfimdb-dev-advanced-output.txt",
        advanced_test_out="cfimdb-test-advanced-output.txt",
        threshold=0.10,
    )


if __name__ == "__main__":
    main()
