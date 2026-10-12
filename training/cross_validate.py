
import pandas as pd
from scipy.sparse import hstack
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score

df = pd.read_csv("dataset/job_contract_scam_dataset.csv")
df = df.drop(columns=["posting_id"])

X = df.drop(columns=["is_fraudulent"])
y = df["is_fraudulent"]

# Keep identical job details in the same fold
groups = pd.util.hash_pandas_object(X, index=False).values

splitter = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

text_columns = [
    "title",
    "company_name",
    "contract_type",
    "experience_letter_terms",
    "package_benefits_detail"
]

numeric_columns = [
    "contract_withholding_clause",
    "has_official_email"
]

accuracies = []
fraud_recalls = []

for fold, (train_idx, test_idx) in enumerate(
    splitter.split(X, y, groups), start=1
):
    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    train_text = X_train[text_columns].fillna("").astype(str).agg(" ".join, axis=1)
    test_text = X_test[text_columns].fillna("").astype(str).agg(" ".join, axis=1)

    tfidf = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        max_features=5000
    )

    train_tfidf = tfidf.fit_transform(train_text)
    test_tfidf = tfidf.transform(test_text)

    train_numeric = X_train[numeric_columns].astype(float).values
    test_numeric = X_test[numeric_columns].astype(float).values

    X_train_final = hstack([train_tfidf, train_numeric])
    X_test_final = hstack([test_tfidf, test_numeric])

    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42
    )

    model.fit(X_train_final, y_train)
    predictions = model.predict(X_test_final)

    accuracy = accuracy_score(y_test, predictions)
    fraud_recall = recall_score(
        y_test, predictions, pos_label=1, zero_division=0
    )

    accuracies.append(accuracy)
    fraud_recalls.append(fraud_recall)

    print(
        f"Fold {fold}: "
        f"accuracy={accuracy * 100:.2f}%, "
        f"fraud recall={fraud_recall * 100:.2f}%, "
        f"test records={len(test_idx)}"
    )

print(f"\nMean accuracy: {sum(accuracies) / len(accuracies) * 100:.2f}%")
print(f"Mean fraud recall: {sum(fraud_recalls) / len(fraud_recalls) * 100:.2f}%")

