# 🌱 Food & GHG Emissions Dashboard

This Streamlit application provides an interactive analytics dashboard for predicting and tracking greenhouse gas (GHG) emissions across the food lifecycle. The tool allows analysts to generate autonomous predictions, explore Scope 1 emission data, and export professional enterprise reports.

## ✨ Features

- **Predictive Analytics (AI-Powered):** 
  - Uses an integrated machine learning model to predict Total Global Average GHG Emissions per kg for food products.
  - Granular lifecycle stage inputs: *Land Use Change, Feed, Farm, Processing, Transport, Packaging, Retail*.
- **Data Visualizations:** 
  - Automatically generates dynamic visual explorations including pie charts, heatmaps, and stacked bar comparisons using Plotly.
  - Cross-analysis of custom food data against standard Scope 1 Emission sources.
- **Reporting & Export:**
  - One-click downloads for predicted data in `.csv` format.
  - Integrated `reportlab` engine to generate detailed PDF summaries straight from the dashboard.
- **Batch Processing:** 
  - Process hundreds of food products at scale by uploading standard `.csv` files.

## 🛠️ Technology Stack

- **Frontend/UI:** Streamlit
- **Data Processing:** Pandas, NumPy
- **Visuals:** Plotly Express, Matplotlib, Seaborn
- **Machine Learning Engine:** Scikit-Learn (Joblib models)
- **PDF Generation:** ReportLab

## 🚀 Setup & Installation

**1. Clone the repository:**
```bash
git clone https://github.com/Srijaisuryaa007/major-project-carbon-accounting-and-ESG-dashboard-.git
cd "major-project-carbon-accounting-and-ESG-dashboard-"
```

**2. Install Dependencies:**
Ensure you have Python installed, then install the required libraries:
```bash
pip install streamlit pandas numpy joblib plotly matplotlib seaborn reportlab scikit-learn
```

**3. Provide Machine Learning Artifacts:**
Ensure the compiled models are in the root directory alongside `code.py`:
- `best_emissions_model.pkl`
- `feature_scaler.pkl`

**4. Run the Application:**
```bash
streamlit run code.py
```

## 📂 Expected File Formats

For Batch Prediction, the expected `.csv` structure is:
| Food product | Land Use Change | Feed | Farm | Processing | Transport | Packaging | Retail |

*Values should be represented in kg CO2e.*
