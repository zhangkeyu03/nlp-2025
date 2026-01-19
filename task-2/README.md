# Semantic Grep (Word2Vec based Search)

This project implements a **Semantic Search Utility** (similar to `grep`) that finds lines in a text file containing not just the query word, but also its contextually similar synonyms. It utilizes **Word2Vec** for semantic embedding and **Lemmatization** for morphological normalization.

## 📌 Project Overview
* **Goal**: Create a search tool that understands semantics (e.g., searching for "money" also finds "funds", "payment").
* **Method**: Training a Word2Vec model on the corpus and expanding user queries with nearest neighbors in the vector space.
* **Language**: Python 3.8+ (Russian language support).

## 🛠 Implementation Details

### 1. Word Embedding (Word2Vec)
* Uses `gensim` to train a **Word2Vec** model (Skip-gram/CBOW) on the provided news corpus.
* Maps words to 100-dimensional vectors to capture semantic meaning based on the **Distributional Hypothesis**.

### 2. Advanced Preprocessing Pipeline
To ensure high-quality vectors and search results, a strict pipeline is applied:
* **Tokenization**: Splits text into tokens using `gensim.utils.simple_preprocess`.
* **Stopword Filtering**: Removes standard Russian stopwords (via `nltk`) and high-frequency noise words (e.g., `эти`, `который`) to prevent irrelevant associations.
* **Lemmatization**: Uses `pymorphy2` to convert words to their canonical forms (Normal Forms).
    * *Example*: `денег` (genitive) -> `деньги` (nominative).
    * This ensures that different morphological forms of the same word are mapped to the same vector.

### 3. Compatibility Fix (Python 3.11+)
* Includes a **Monkey Patch** for the `inspect` module.
* `pymorphy2` relies on `inspect.getargspec` (removed in Python 3.11). The script dynamically patches this function to ensure stability on modern Python environments.

### 4. Semantic Search Logic
1.  **Query Expansion**: The user's query is lemmatized, and the top-N synonyms are retrieved using `model.wv.most_similar()`.
2.  **Intersection Matching**: Instead of simple substring matching, the script checks for the **intersection** of the query's lemma set and the line's lemma set, ensuring accurate semantic hits.

## 🚀 How to Run

1.  Install dependencies:
    ```bash
    pip install gensim pymorphy2 nltk
    ```

2.  Run the script (ensure `news.txt` is in the same directory):
    ```bash
    python semantic_grep.py
    ```

3.  The script will:
    * Train the model (if not already in memory).
    * Execute example queries (e.g., "футбол", "деньги").
    * Output lines matching the query or its synonyms.

## 📂 Files
* `semantic_grep.py`: Main executable script.
* `news.txt`: Corpus data (Russian news).
