import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

# ================================
# Load Model and Scaler
# ================================
@st.cache_resource
def load_model():
    model = joblib.load("best_emissions_model.pkl")
    scaler = joblib.load("feature_scaler.pkl")
    return model, scaler

model, scaler = load_model()

# ================================
# Define Columns
# ================================
COMPONENT_COLS = ["Land Use Change", "Feed", "Farm", "Processing", "Transport", "Packaging", "Retail"]

st.set_page_config(page_title="Food & GHG Emissions Dashboard", layout="wide")

st.title("🌱 Food & GHG Emissions Dashboard")

st.markdown("""
This app predicts **Total Global Average GHG Emissions per kg of food products** 
and also visualizes **Scope 1 emission factors** from the EPA dataset.

You can:
- Upload food product lifecycle data for predictions.
- Analyze GHG emission factors for different sectors.
- Compare and visualize both datasets together.
""")

# ================================
# Sidebar Navigation
# ================================
menu = st.sidebar.radio(
    "Navigation",
    ["Home", "Manual Prediction", "Batch Prediction", "Visualizations", "Generate Report"]
)

# ================================
# Helper Functions
# ================================
def predict_emissions(dataframe):
    """Predict emissions using the trained model."""
    dataframe["sum_components"] = dataframe[COMPONENT_COLS].sum(axis=1)
    dataframe["transport_frac"] = dataframe["Transport"] / dataframe["sum_components"].replace(0, np.nan)
    dataframe["transport_frac"] = dataframe["transport_frac"].fillna(0.0)
    dataframe["transport_ef"] = 2.68  # Default emission factor for transport

    # Scale features
    X_scaled = scaler.transform(dataframe[COMPONENT_COLS + ["transport_ef", "sum_components", "transport_frac"]])
    predictions = model.predict(X_scaled)
    return predictions

def generate_pdf_report(data, predictions):
    """Generate a simple PDF report."""
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    c.setFont("Helvetica", 12)
    c.drawString(50, 800, "Food Product Emissions Report")
    c.drawString(50, 780, "-------------------------------------------")

    for idx, (product, pred) in enumerate(zip(data['Food product'], predictions)):
        c.drawString(50, 750 - (idx * 20), f"{product}: {pred:.2f} kg CO2e/kg")

    c.save()
    buffer.seek(0)
    return buffer

# ================================
# Home
# ================================
if menu == "Home":
    st.subheader("Welcome!")
    st.write("""
    **Features of this Dashboard:**
    - Predict total emissions per food product using AI.
    - Explore Scope 1 GHG emission factors.
    - Compare and visualize both datasets.
    - Export reports as CSV and PDF.
    """)

    st.image("https://images.unsplash.com/photo-1506806732259-39c2d0268443", caption="Sustainable food systems", use_column_width=True)

# ================================
# Manual Prediction
# ================================
elif menu == "Manual Prediction":
    st.subheader("Manual Prediction")
    st.write("Enter lifecycle stage values below:")

    inputs = {}
    for col in COMPONENT_COLS:
        inputs[col] = st.number_input(f"{col} (kg CO2e)", min_value=0.0, value=1.0, step=0.1)

    input_df = pd.DataFrame([inputs])
    prediction = predict_emissions(input_df)[0]

    st.success(f"Predicted Total Emissions: **{prediction:.2f} kg CO2e/kg**")

# ================================
# Batch Prediction
# ================================
elif menu == "Batch Prediction":
    st.subheader("Batch Prediction from CSV")
    st.write("""
    Upload a CSV with the following columns:
    - Food product
    - Land Use Change
    - Feed
    - Farm
    - Processing
    - Transport
    - Packaging
    - Retail
    """)

    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if all(col in df.columns for col in COMPONENT_COLS):
            preds = predict_emissions(df)
            df["Predicted Total Emissions (kg CO2e/kg)"] = preds

            st.write("### Prediction Results")
            st.dataframe(df)

            # Download results
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Predictions as CSV", csv, "predictions.csv", "text/csv")

            # PDF Report
            pdf_buffer = generate_pdf_report(df, preds)
            st.download_button("Download PDF Report", pdf_buffer, "emissions_report.pdf")
        else:
            st.error("Uploaded file missing required columns.")

# ================================
# Visualizations
# ================================
elif menu == "Visualizations":
    st.subheader("Visualizations")

    viz_type = st.radio(
        "Select Visualization Type",
        ["Food Emissions", "GHG Scope 1 Data", "Combined Analysis"]
    )

    # -------------------------------
    # Food Emissions Visualization
    # -------------------------------
    if viz_type == "Food Emissions":
        st.write("### Upload Food Data for Visualization")
        uploaded_file = st.file_uploader("Upload food emissions CSV", type=["csv"])
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            preds = predict_emissions(df)
            df["Predicted Total Emissions (kg CO2e/kg)"] = preds

            # 1️⃣ Average Lifecycle Contribution (Pie Chart)
            st.write("#### Average Lifecycle Contribution")
            avg_contrib = df[COMPONENT_COLS].mean()
            fig = px.pie(
                values=avg_contrib.values,
                names=avg_contrib.index,
                title="Average Lifecycle Stage Contribution"
            )
            st.plotly_chart(fig)

            # 2️⃣ Scatter plot - Retail vs Predicted Emissions
            st.write("#### Retail vs Predicted Emissions")
            scatter_fig = px.scatter(
                df,
                x="Retail",
                y="Predicted Total Emissions (kg CO2e/kg)",
                color="Food product",
                title="Retail vs Predicted Emissions"
            )
            st.plotly_chart(scatter_fig)

            # 3️⃣ Correlation Heatmap
            st.write("#### Correlation Heatmap")
            corr = df[COMPONENT_COLS + ["Predicted Total Emissions (kg CO2e/kg)"]].corr()
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)

            # 4️⃣ Top 10 Food Products by Emissions
            st.write("#### Top 10 Food Products by Predicted Emissions")
            top_products = df.groupby("Food product")["Predicted Total Emissions (kg CO2e/kg)"].mean().sort_values(ascending=False).head(10)
            bar_fig = px.bar(
                x=top_products.index,
                y=top_products.values,
                labels={"x": "Food Product", "y": "Average Predicted Emissions (kg CO2e/kg)"},
                title="Top 10 Food Products by Emissions"
            )
            st.plotly_chart(bar_fig)

            # 5️⃣ Lifecycle Breakdown for Top Products (Stacked Bar Chart)
            # 5️⃣ Lifecycle Breakdown for Top Products (Stacked Bar Chart)
            st.write("#### Lifecycle Stage Breakdown for Top Products")

# Fix: select numeric columns only
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            top_foods = (
                df.groupby("Food product")[numeric_cols]
                .mean()
                .sort_values(by="Predicted Total Emissions (kg CO2e/kg)", ascending=False)
                .head(5)
            )

            stack_df = top_foods[COMPONENT_COLS].reset_index().melt(
                id_vars=["Food product"], var_name="Lifecycle Stage", value_name="Emissions"
            )

            stack_fig = px.bar(
                stack_df,
                x="Food product",
                y="Emissions",
                color="Lifecycle Stage",
                title="Lifecycle Stage Contribution per Top Product",
                barmode="stack"
            )
            st.plotly_chart(stack_fig)

            

    # -------------------------------
    # GHG Scope 1 Visualization
    # -------------------------------
    elif viz_type == "GHG Scope 1 Data":
        st.write("### Upload GHG Emission Factors Excel")
        ghg_file = st.file_uploader("Upload Scope 1 GHG Emissions Excel", type=["xlsx"])
        if ghg_file:
            ghg_df = pd.read_excel(ghg_file)

            st.write("#### Preview of GHG Scope 1 Data")
            st.dataframe(ghg_df.head())

            if 'GHG Emission Factor' in ghg_df.columns:
                top_factors = ghg_df.sort_values(by='GHG Emission Factor', ascending=False).head(10)
                fig = px.bar(
                    top_factors,
                    x='GHG Emission Factor',
                    y='Emission Source',
                    orientation='h',
                    title="Top 10 Scope 1 Emission Sources"
                )
                st.plotly_chart(fig)
            else:
                st.warning("Couldn't find column 'GHG Emission Factor' in the dataset.")

    # -------------------------------
    # Combined Analysis
    # -------------------------------
    elif viz_type == "Combined Analysis":
        st.write("### Upload Both Food and Scope 1 Data for Comparison")

        col1, col2 = st.columns(2)

        with col1:
            food_file = st.file_uploader("Upload Food Emissions CSV", type=["csv"], key="food_combined")

        with col2:
            ghg_file = st.file_uploader("Upload Scope 1 GHG Excel", type=["xlsx"], key="ghg_combined")

        if food_file and ghg_file:
            food_df = pd.read_csv(food_file)
            ghg_df = pd.read_excel(ghg_file)

            preds = predict_emissions(food_df)
            food_df["Predicted Total Emissions (kg CO2e/kg)"] = preds

            st.write("### Side-by-Side Comparison")
            col1, col2 = st.columns(2)

            with col1:
                st.write("**Food Emissions Predictions**")
                st.dataframe(food_df.head())

            with col2:
                st.write("**Scope 1 Emission Factors**")
                st.dataframe(ghg_df.head())

            st.write("### Comparison Chart")
            combined_chart = px.bar(
                x=["Food Emissions (Avg)", "Scope 1 Emissions (Avg)"],
                y=[food_df["Predicted Total Emissions (kg CO2e/kg)"].mean(),
                   ghg_df["GHG Emission Factor"].mean()],
                labels={'x': 'Category', 'y': 'Average Emissions (kg CO2e)'},
                title="Average Emissions Comparison"
            )
            st.plotly_chart(combined_chart)

# ================================
# Generate Report
# ================================
elif menu == "Generate Report":
    st.subheader("Generate Final Report")
    st.write("""
    Combine your predictions and visualizations into a downloadable PDF report for stakeholders.
    """)
    st.info("Upload the same CSV used for predictions.")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        preds = predict_emissions(df)
        df["Predicted Total Emissions (kg CO2e/kg)"] = preds

        pdf_buffer = generate_pdf_report(df, preds)
        st.download_button("Download PDF Report", pdf_buffer, "final_report.pdf")
