
#  Day 2 Report – Data Preprocessing Analysis
#Name: Amin Moghadam
**Date:** 1404-04-24 
**Subject:** Ai

---

##  Continued Research – Review of `preprocces_pipeline.py`

On Day 2, the primary focus was on analyzing the structure and logic of the `preprocces_pipeline.py` file, which is responsible for cleaning and preparing raw data before passing it to the machine learning models.

###  Overview of the Pipeline:
- **Input:** A CSV file from `dataset/` containing drilling parameters such as `rpm`, `spp`, `flow_rate`, `ecd`, `temperature`, and more.
- **Main processing steps:**
  1. Detect and remove or fill missing values
  2. Validate value ranges for key parameters (e.g., `0 < flow_rate < 1500`)
  3. Normalize or scale specific features (if necessary)
  4. Tag anomalous or suspicious data entries
  5. Save the cleaned output to `data/clean_data.csv`

###  Key Snippets:
```python
df = pd.read_csv("dataset/raw_data.csv")
df.dropna(inplace=True)
df = df[(df["flow_rate"] > 0) & (df["flow_rate"] < 1500)]
df.to_csv("data/clean_data.csv", index=False)
```

---

##  Key Observations:
- Some important fields (e.g., `ph`, `cl_concentration`) are not yet validated in the current pipeline.
- The directory `data/clean_data.csv` did not exist and had to be created manually.
- The current pipeline can be further enhanced with modular design and testing support.

---

##  Tasks Completed:
- Successfully ran the pipeline on `dataset/sample_data.csv`
- Verified and inspected the cleaned output
- Prepared cleaned data for upcoming model training tasks

---

##  Plan for Tomorrow – Start ML Model Training

###  Day 3 Priorities:
1. Review the model training script (likely in `Scripts/train_model.py`)
2. Train a baseline model (e.g., XGBoost or simple regression) using cleaned data
3. Evaluate model using RMSE and MAE metrics
4. Save trained model artifact to the `models/` directory

---
