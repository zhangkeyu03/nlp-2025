# NLP Course Assignments (2025)

This repository contains solutions for the Natural Language Processing course tasks, focusing on Information Extraction and Semantic Search using Python.

## Task 1: Information Extraction with Yargy
**File:** `solve_task_1_fast.py`

### Description
Extracts structured information about people's birth details (Name, Birth Date, Birth Place) from unstructured Russian news texts (`news.txt`).

### Implementation Details
* **Library:** Uses `Yargy`, a rule-based extraction library for Russian, leveraging `pymorphy2` for morphological analysis.
* **Rules:** Defined Context-Free Grammars (CFG) for:
    * **Names:** Capitalized words tagged as `Name`.
    * **Dates:** Formats like "12 мая 1990 года" or "1990 г.".
    * **Places:** Capitalized words tagged as `Geox` (Locations).
* **Patterns:** Combined rules to match sentence structures like:
    * `Name` + `Born_Verb` + `Date` + `Place`
    * `Native_of` + `Place` + `Name`
* **Optimization:** Implemented a **keyword pre-filtering mechanism**. Lines not containing roots like `родил` (born) or `урожен` (native of) are skipped before parsing, significantly improving processing speed on large datasets.

### How to Run
```bash
python solve_task_1_fast.py
