"""
Setup
"""
from pathlib import Path

import setuptools

long_description = Path("README.md").read_text(encoding="utf8")

authors = [
  "Victor Zhou (original author)",
  "Menelaos Kotoglou",
  "Dimitrios Mistriotis",
  "Reyniel Mark T. Escamillas"
]

setuptools.setup(
    name="alt-profanity-check",
    version="0.1.8",
    author= ", ".join(authors),
    author_email="rtescamillas@ccc.edu.ph",
    description=(
        "A fast, robust library to check for offensive language in strings. "
        'This version of "profanity-check" is used for Bleepy'
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/reyniel26/bleepy-profanity-check",
    packages=setuptools.find_packages(),
    install_requires=["scikit-learn==1.1.3", "joblib>=1.2.0"],
    python_requires=">=3.7",
    package_data={"profanity_check": [
      "data/model.joblib", "data/vectorizer.joblib",
      "data/tagalog_vectorizer.joblib","data/tagalog_model.joblib"
      ]},
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    # entry_points={
    #     "console_scripts": ["profanity_check=profanity_check.command_line:main"],
    # },
)
