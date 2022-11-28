"""
Train model

When the sklearn update their module, update also the models
If the models are not updated, it will show *UserWarning*
"""
import pandas as pd
from joblib import dump
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

clean_data_lists = ["clean_data.csv","tagalog_clean_data.csv"]

for clean_data in clean_data_lists:

    data = pd.read_csv(clean_data)
    texts = data["text"].astype(str)
    y = data["is_offensive"]

    vectorizer = TfidfVectorizer(stop_words="english", min_df=0.0001)
    X = vectorizer.fit_transform(texts)

    model = LinearSVC(class_weight="balanced", dual=False, tol=1e-2, max_iter=1e5)
    cclf = CalibratedClassifierCV(base_estimator=model)
    cclf.fit(X, y)

    data_prefix = clean_data.split('_', maxsplit=1)[0]
    data_prefix = "" if data_prefix == "clean" else f"{data_prefix}_"

    dump(vectorizer, f"{data_prefix}vectorizer.joblib")
    dump(cclf, f"{data_prefix}model.joblib")
