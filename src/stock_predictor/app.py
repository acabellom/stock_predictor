import pandas as pd
import streamlit as st

# Cargar CSV
df = pd.read_csv("data/AAPL_historical_data.csv", index_col="t", parse_dates=True)
print(df.head())

st.subheader("Historical Stock Data for AAPL")

# Mostrar línea de average_price
st.line_chart(df["average_price"])
