#  Day 1 Report – Project Research Phase
# Name: Amin Moghadam
**Date:** 1404-04-23
**Subject:** Ai 

---

## 🧠 Research Phase – Understanding the Project and Requirements

On the first day, the focus was on reading and understanding the overall structure and goals of the project. The `README.md` file served as the main **Software Requirements Specification (SRS)** document.

### ✳️ Summary of Project Goals:
- To develop an intelligent **Formation Damage Monitoring System (FDMS)** that monitors and predicts formation damage during drilling and cementing operations.
- To leverage real-time MWD/LWD sensor data and machine learning models (XGBoost, LSTM, GRU) for prediction and alerting.
- To build an interactive dashboard for real-time monitoring, alerting, and analytics for engineers.

### ✳️ Core Components Identified:
- ✅ Data validation pipeline
- ✅ Fluid loss and emulsion risk detection engine
- ✅ Predictive maintenance model for formation damage
- ✅ Interactive dashboard (React + Plotly)
- ✅ Backend API layer (FastAPI)

### ✳️ Project Structure:
The project includes modular components for:
- Raw and cleaned datasets (`dataset/`)
- Trained ML models (`models/`)
- Preprocessing logic (`preprocces_pipeline.py`)
- Frontend and backend services (`frontend/`, `backend/`)

---

## 🗓️ Plan for Tomorrow – Start of Implementation Phase

### ✅ Day 2 Priorities:
1. Review `preprocces_pipeline.py` to understand the data cleaning and validation logic
2. Run the pipeline on sample data in the `dataset/` folder
3. Generate a cleaned CSV output for ML model training
4. Document and summarize pipeline findings for Day 2 report

---

## 💡 GPT Suggested Technical Focus:
To kick off practical work, focus on the **data validation pipeline**. Analyze `preprocces_pipeline.py` line by line:
- Categorize validation checks (e.g., missing values, range constraints, unit consistency)
- Clean or tag faulty/suspicious data points
- Save the cleaned output to a `data/clean/` directory

Suggested tools:
- `pandas` for analysis and cleanup
- `matplotlib` or `seaborn` for data distribution visualization
- Optional: add a small notebook in `notebooks/` to test pipeline performance

---
