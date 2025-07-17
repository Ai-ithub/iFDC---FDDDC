# 🚀 Work Proposal

## Identify and manage duplicate data Issue

**Prepared by:**

- **Name:** Shayan Talebian
- **Field:** Artificial Intelligence
- **Project Title:** iFDC---FDDDC
- **Email:** <Shayantalebianwork@gmail.com>
- **Branch Name:** Shayan-Talebian
- **Start Date:** 1404/04/16 (2025-07-07)
- **Date:** 1404/04/24 (2025-07-15)

---

## 🎯 Project Objective

The primary goal of this task is to detect duplicate records in drilling data logs, thoroughly analyze their causes, and apply appropriate cleaning strategies (such as removal or aggregation).  
This process ensures data integrity, prevents misleading patterns, and enhances the overall quality of datasets for downstream machine learning model training and predictive analytics.

---

## 🔍 Detailed Scope of Work

### 1. Exploratory Duplicate Analysis

- Load raw drilling datasets in **CSV** or **Parquet** format.
- Identify:
  - **Fully duplicated rows:** using standard DataFrame checks.
  - **Partial duplicates:** records sharing the same key features (`timestamp`, `depth`, `pressure`, etc.) but differing in other columns.
- Perform statistical summaries:
  - Frequency counts of duplicate records.
  - Percentage of total records affected.
  - Distributions across features to understand clustering of duplicates.

### 2. Root Cause Investigation

- Examine potential reasons for duplicates:
  - Sensor noise or misfires causing repeated logging.
  - Automated systems duplicating logs during retries or failovers.
  - Historical merges from multiple sources without deduplication.

### 3. Cleaning Strategy Formulation

- Define specific policies to address duplicates:
  - **Exact duplicates:** drop all but one occurrence.
  - **Partial duplicates:** aggregate by mean, select most recent (last), or based on highest data quality indicators.
- Document these policies in a clear `config.yaml` for transparency and reproducibility.

### 4. Data Cleaning Implementation

- Build a robust script (or notebook) that:
  - Loads the data.
  - Applies the deduplication strategy.
  - Saves a cleaned dataset to `data/cleaned/no_duplicates.csv`.
- Implement optional parameters such as:
  - `--method mean` or `--method last`
  - `--dry-run` for preview without writing changes.

### 5. Reporting & Documentation

- Generate a text summary report (`reports/duplicates_summary.txt`) containing:
  - Initial vs. final record counts.
  - Number of full duplicates removed.
  - Handling strategy used.
  - Notes on any assumptions or data quality caveats.
- Optionally save charts (PNG/SVG) illustrating duplication patterns.

---

## 🛠 Tools & Technology Stack

| Tool / Library       | Purpose                             |
| -------------------- | ----------------------------------- |
| Python (3.9+)        | Primary language                    |
| Pandas               | Data loading & manipulation         |
| NumPy                | Numerical operations & stats        |
| Matplotlib / Seaborn | Visualizations & distribution plots |
| Jupyter Notebook     | Interactive EDA & documentation     |
| Click / argparse     | Command-line interface for scripts  |
| rich (optional)      | Stylish terminal outputs            |
| pytest (optional)    | Unit tests to validate cleaning     |

---

## 🗂 Files & Outputs

| File/Folder                          | Description                        |
| ------------------------------------ | ---------------------------------- |
| `data/raw/*.csv or *.parquet`        | Raw drilling data files            |
| `scripts/remove_duplicates.py`       | Script to clean duplicates         |
| `notebooks/duplicate_analysis.ipynb` | Notebook for EDA & visualization   |
| `config.yaml`                        | Configuration for cleaning policy  |
| `data/cleaned/no_duplicates.csv`     | Final cleaned dataset              |
| `reports/duplicates_summary.txt`     | Text report on duplicates handling |

---

## 💡 Proposed Enhancements

To maximize long-term value and maintainability:

- ✅ **Modular & Configurable:**  
  Cleaning logic parameterized via CLI flags and `config.yaml`.

- ✅ **Rich Logging:**  
  Utilize `rich` or `loguru` for clear, color-coded logs—making it easy to track which records were removed or aggregated.

- ✅ **Dry Run Support:**  
  Ability to preview cleaning actions without writing outputs (`--dry-run`), ensuring full transparency.

- ✅ **Heatmaps & Histograms:**  
  Visualizations of duplicate clustering by depth and time can reveal underlying operational issues.

- ✅ **Unit Tests:**  
  Validate no duplicate remains post-cleaning and that aggregation strategies function as intended.

- ✅ **Future Scalability:**  
  Designed to handle other wells or multi-well logs with minimal modification.

---

## ✈ Expected Deliverables

🎯 By end of project, you will receive:

- 📊 A comprehensive analytical text report (`reports/duplicates_summary.txt`) with duplicate counts, cleaning strategy, and root cause notes.
- 🗂 A cleaned dataset (`data/cleaned/no_duplicates.csv`).
- ⚙ A Python script (`scripts/remove_duplicates.py`) ready for reuse with CLI options.
- 📖 A detailed exploratory notebook (`notebooks/duplicate_analysis.ipynb`) including charts & commentary.
- 📝 A sample `config.yaml` outlining cleaning rules.
- (Optionally) unit tests ensuring no residual duplicates exist.

---

## 💎 Business Value & Impact

- 📈 **Enhanced Data Integrity:**  
  Ensures the ML models are trained on high-quality, non-redundant data.

- ⏱ **Faster Iteration Cycles:**  
  With a clean dataset and reusable pipeline, future wells can be processed with minimal extra effort.

- 🔬 **Deeper Operational Insights:**  
  Duplicate heatmaps can even help drilling teams improve sensor calibration or data capture protocols.

---

## 👨‍💻 Ready to Kick Off

This proposal is designed to deliver robust, documented, and scalable data cleaning capabilities, specifically tailored to your drilling log datasets.  
It will not only improve the immediate dataset quality but also set a gold standard for similar data pipelines in future wells.

---

## 🔎 Note on Skill Development & Learning

I would like to transparently share that I am still a junior-level data scientist.  
I have a solid understanding of **Python**, and I am very comfortable with **Pandas** for data manipulation and exploratory analysis.  
Currently, I am actively working on improving my skills in **NumPy** and more advanced statistical and cleaning techniques.

This project is perfectly aligned with my learning path.  
While I may not yet have deep hands-on experience with all aspects (especially complex statistical deduplication or production-level pipeline structuring), I am highly committed to upskilling and researching best practices to deliver quality results.

I am confident that with focused effort, time, and iterative learning, I will be able to successfully complete all the tasks described in this proposal.  
I deeply value honesty and prefer to clearly communicate my current skill level, as I strongly believe in growing through real-world challenges.

Thank you for considering this, and I look forward to the opportunity to not only deliver this project but also advance my expertise in the process.

---

**Prepared by:**  
**Shayan Talebian**
