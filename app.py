# -*- coding: utf-8 -*-
import streamlit as st
import streamlit as st
try:
    import sklearn
    import pickle
except ModuleNotFoundError as e:
    st.error(f"Missing dependency: {e.name}. Did requirements.txt include it?")
    st.stop()

# Then load model
with open("pipeline.pkl", "rb") as f:
    model = pickle.load(f)
import pandas as pd
import pickle

# Set Streamlit config
st.set_page_config(page_title="💎 Diamond Price Predictor", page_icon="💎", layout="centered")

# Load the trained model pipeline
with open("pipeline.pkl", "rb") as f:
    model = pickle.load(f)

# Load data to extract dropdown options
df = pd.read_csv("Diamond Price Prediction.csv")

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
        carat = st.selectbox("💠 Carat", options=sorted(df["Carat(Weight of Daimond)"].unique()))
        cut = st.selectbox("✂️ Cut", options=sorted(df["Cut(Quality)"].unique()))
        color = st.selectbox("🎨 Color", options=sorted(df["Color"].unique()))
        clarity = st.selectbox("🔍 Clarity", options=sorted(df["Clarity"].unique()))

    with col2:
        depth = st.selectbox("📏 Depth", options=sorted(df["Depth"].unique()))
        table = st.selectbox("📐 Table", options=sorted(df["Table"].unique()))
        x = st.selectbox("📏 X (Length)", options=sorted(df["X(length)"].unique()))
        y = st.selectbox("📏 Y (Width)", options=sorted(df["Y(width)"].unique()))
        z = st.selectbox("📏 Z (Depth)", options=sorted(df["Z(Depth)"].unique()))

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
