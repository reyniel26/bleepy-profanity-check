"""
Bleepy Profanity Check

This version of profanity check is mainly for Bleepy
"""
import joblib
import numpy as np
import pkg_resources

vectorizer = joblib.load(
  pkg_resources.resource_filename('profanity_check', 'data/vectorizer.joblib')
  )
model = joblib.load(pkg_resources.resource_filename('profanity_check', 'data/model.joblib'))

tagalog_vectorizer = joblib.load(
  pkg_resources.resource_filename('profanity_check', 'data/tagalog_vectorizer.joblib')
  )
tagalog_model = joblib.load(
  pkg_resources.resource_filename('profanity_check', 'data/tagalog_model.joblib')
  )

def _get_profane_prob(prob):
    """Get probability"""
    return prob[1]

def predict(texts, lang="english"):
    """
    Get prediction

    You can indicate the language of the word you want to predict, such as

    - `lang = 'english'` for English language
    - `lang = 'tagalog'` for Tagalog/Filipino language
    """
    return (
      tagalog_predict(texts)
      if lang == "tagalog"
      else model.predict(vectorizer.transform(texts))
      )

def predict_prob(texts, lang="english"):
    """
    Get probability

    You can indicate the language of the word you want to predict, such as

    - `lang = 'english'` for English language
    - `lang = 'tagalog'` for Tagalog/Filipino language
    """
    return (
      tagalog_predict_prob(texts)
      if lang == "tagalog"
      else np.apply_along_axis(
        _get_profane_prob, 1, model.predict_proba(vectorizer.transform(texts))
        )
      )

def tagalog_predict(texts):
    """Get prediction for Tagalog/Filipino"""
    return tagalog_model.predict(tagalog_vectorizer.transform(texts))

def tagalog_predict_prob(texts):
    """Get probability for Tagalog/Filipino"""
    return np.apply_along_axis(
    _get_profane_prob, 1, tagalog_model.predict_proba(tagalog_vectorizer.transform(texts))
    )
