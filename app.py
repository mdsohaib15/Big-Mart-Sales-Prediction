import streamlit as st
import numpy as np
import pandas as pd
import pickle
from xgboost import XGBRegressor

# Load model
model = pickle.load(open('model.pkl', 'rb'))

# Load dataset for Item Identifiers
df = pd.read_csv('Train.csv')
item_identifiers = df['Item_Identifier'].unique()

# Title and Image
st.title("🛒 Big Mart Sales Prediction 📊")
st.image("bigmart.jpg", caption="Big Mart Sales Data Analysis", use_container_width =True)

# Input options
item_fat_content_options = ['Low Fat', 'Regular', 'Non-Edible']
item_type_options = ['Dairy', 'Soft Drinks', 'Meat', 'Fruits and Vegetables', 'Household',
                     'Baking Goods', 'Snack Foods', 'Frozen Foods', 'Breakfast', 'Health and Hygiene',
                     'Hard Drinks', 'Canned', 'Breads', 'Starchy Foods', 'Others', 'Seafood']
outlet_identifiers = ['OUT049', 'OUT018', 'OUT010', 'OUT013', 'OUT027', 'OUT045', 'OUT017', 'OUT046', 'OUT035', 'OUT019']
outlet_years = [1985, 1987, 1997, 1998, 1999, 2002, 2004, 2007, 2009]
outlet_sizes = ['Small', 'Medium', 'High']
outlet_location_types = ['Tier 1', 'Tier 2', 'Tier 3']
outlet_types = ['Grocery Store', 'Supermarket Type1', 'Supermarket Type2', 'Supermarket Type3']

# Encoders
fat_content_map = {'Low Fat': 0, 'Regular': 1, 'Non-Edible': 2}
item_type_map = {v: i for i, v in enumerate(item_type_options)}
outlet_id_map = {v: i for i, v in enumerate(outlet_identifiers)}
outlet_size_map = {'Small': 0, 'Medium': 1, 'High': 2}
location_type_map = {'Tier 1': 0, 'Tier 2': 1, 'Tier 3': 2}
outlet_type_map = {'Grocery Store': 0, 'Supermarket Type1': 1, 'Supermarket Type2': 2, 'Supermarket Type3': 3}
item_id_map = {v: i for i, v in enumerate(item_identifiers)}

# === Layout: Two columns ===
col1, col2 = st.columns(2)

with col1:
    Item_Identifier = st.selectbox("Select Item Identifier", item_identifiers)
    Item_Fat_Content = st.selectbox("Select Item Fat Content", item_fat_content_options)
    Item_Type = st.selectbox("Select Item Type", item_type_options)
    Outlet_Identifier = st.selectbox("Select Outlet Identifier", outlet_identifiers)
    Outlet_Size = st.selectbox("Select Outlet Size", outlet_sizes)
    Outlet_Type = st.selectbox("Select Outlet Type", outlet_types)

with col2:
    Item_Weight = st.number_input("Enter Item Weight", min_value=0.0)
    Item_Visibility = st.number_input("Enter Item Visibility", min_value=0.0, max_value=1.0)
    Item_MRP = st.number_input("Enter Item MRP", min_value=0.0)
    Outlet_Establishment_Year = st.selectbox("Select Outlet Establishment Year", outlet_years)
    Outlet_Location_Type = st.selectbox("Select Outlet Location Type", outlet_location_types)

# === Predict Button ===
st.markdown("---")
if st.button("Predict"):
    features = np.array([[  
        item_id_map[Item_Identifier],
        Item_Weight,
        fat_content_map[Item_Fat_Content],
        Item_Visibility,
        item_type_map[Item_Type],
        Item_MRP,
        outlet_id_map[Outlet_Identifier],
        Outlet_Establishment_Year,
        outlet_size_map[Outlet_Size],
        location_type_map[Outlet_Location_Type],
        outlet_type_map[Outlet_Type],
    ]], dtype=np.float64)

    prediction = model.predict(features).reshape(1, -1)
    st.subheader("📈 Predicted Sales")
    st.success(f"{prediction[0][0]:.2f}")
