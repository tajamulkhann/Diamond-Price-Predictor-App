# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import pickle

# Safe dependency check
try:
    import sklearn
except ModuleNotFoundError as e:
    st.error(f"Missing dependency: {e.name}. Did requirements.txt include it?")
    st.stop()

# Set Streamlit config
st.set_page_config(page_title="💎 Diamond Price Predictor", page_icon="💎", layout="centered")

# Load model
with open("pipeline.pkl", "rb") as f:
    model = pickle.load(f)

# Load data
df = pd.read_csv("Diamond Price Prediction.csv")

# Rename columns to match model training
df.rename(columns={
    "Carat(Weight of Daimond)": "carat",
    "Cut(Quality)": "cut",
    "Color": "color",
    "Clarity": "clarity",
    "Depth": "depth",
    "Table": "table",
    "X(length)": "x",
    "Y(width)": "y",
    "Z(Depth)": "z"
}, inplace=True)

# Streamlit App
def main():
    # Sidebar
    with st.sidebar:
        st.image("https://avatars.githubusercontent.com/u/107360623?v=4", caption="Tajamul Khan", width=120)
        st.markdown("## 📘 About")
        st.markdown("""
        **App Name:** Diamond Price Predictor  
        **Author:** Tajamul Khan  
        **Model:** Random Forest Regressor Pipeline  
        **Connect:** [LinkedIn](https://www.linkedin.com/in/tajamulkhann/)
        """)

    # Title
    st.markdown("<h1 style='text-align: center; color: #1f77b4;'>💎 Diamond Price Predictor</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Estimate the price of a diamond based on its attributes</p>", unsafe_allow_html=True)
    st.markdown("---")

    # Input Form
    col1, col2 = st.columns(2)

    with col1:
        carat = st.selectbox("💠 Carat", sorted(df["carat"].unique()))
        cut = st.selectbox("✂️ Cut", sorted(df["cut"].unique()))
        color = st.selectbox("🎨 Color", sorted(df["color"].unique()))
        clarity = st.selectbox("🔍 Clarity", sorted(df["clarity"].unique()))

    with col2:
        depth = st.selectbox("📏 Depth", sorted(df["depth"].unique()))
        table = st.selectbox("📐 Table", sorted(df["table"].unique()))
        x = st.selectbox("📏 X (Length)", sorted(df["x"].unique()))
        y = st.selectbox("📏 Y (Width)", sorted(df["y"].unique()))
        z = st.selectbox("📏 Z (Depth)", sorted(df["z"].unique()))

    # Prediction Button
    if st.button("🔮 Predict Price"):
        try:
            input_df = pd.DataFrame({
                'carat': [carat],
                'cut': [cut],
                'color': [color],
                'clarity': [clarity],
                'depth': [depth],
                'table': [table],
                'x': [x],
                'y': [y],
                'z': [z]
            })

            price = model.predict(input_df)[0]
            st.success(f"💵 Estimated Diamond Price: ${price:,.2f}")
        except Exception as e:
            st.error(f"❌ Error: {e}")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; font-size: 12px;'>
        © 2025 <b>Tajamul Khan</b> | Built with ❤️ using Streamlit  
        <a href='https://www.linkedin.com/in/tajamulkhann/' target='_blank'>🔗 LinkedIn</a>
    </div>
    """, unsafe_allow_html=True)

# Run the App
if __name__ == '__main__':
    main()
