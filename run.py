import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px

def load_real_data():
    tdt = pd.read_csv("train.csv")
    tedt = pd.read_csv("test.csv")

    y = tdt['SalePrice']
    tdt.drop(['SalePrice'], axis=1, inplace=True)

    # Combine datasets for consistent preprocessing
    dt = pd.concat([tdt, tedt], axis=0, sort=False)
    dt.drop(['Id'], axis=1, inplace=True)
    dt.fillna(dt.median(numeric_only=True), inplace=True)
    dt.fillna("None", inplace=True)
    dt = pd.get_dummies(dt)

    scaler = StandardScaler()
    scdt = pd.DataFrame(scaler.fit_transform(dt), columns=dt.columns)

    xtrain = scdt.iloc[:y.shape[0], :]
    xtrain2, xval, ytrain, yval = train_test_split(xtrain, y, test_size=0.2, random_state=42)

    return xtrain2, xval, ytrain, yval, dt.columns, scaler

# Train model function
def model_tr(xtrain, ytrain):
    model = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42)
    model.fit(xtrain, ytrain)
    return model

# Main Streamlit App
def main():
    st.title("House Price Predictor with XGBoost")
    st.write("Enter your property details to predict the price using ML model")

    # Load data and train model
    xtrain2, xval, ytrain, yval, feature_names, scaler = load_real_data()
    model = model_tr(xtrain2, ytrain)

    # Show input fields for prediction (basic only)
    st.subheader(" Enter the required entries:")
    input_values = {
        'LotArea': st.number_input("Plot Size (LotArea in sq ft)", min_value=500.0, max_value=100000.0, value=7500.0),
        'YearBuilt': st.number_input("Year Built", min_value=1800, max_value=2025, value=2005),
        'OverallCond': st.number_input("Overall Condition (1-10)", min_value=1, max_value=10, value=5)
    }

    input_df = pd.DataFrame([np.zeros(len(feature_names))], columns=feature_names)
    for k, v in input_values.items():
        if k in input_df.columns:
            input_df[k] = v

    # Predict button
    if st.button("Predict Price"):
        prediction = model.predict(input_df)[0]
        st.success(f"Predicted Price: ₹{prediction:,.2f}")

        # Plot with Plotly
        y_pred = model.predict(xval)
        fig = px.scatter(x=yval, y=y_pred, labels={'x': 'Actual Price', 'y': 'Predicted Price'},
                         title="Actual vs Predicted House Prices")
        fig.add_scatter(x=[prediction], y=[prediction], mode='markers', marker=dict(size=12, color='red'), name='Your Prediction')
        st.plotly_chart(fig)

        # Show performance metrics
        st.subheader("Model Performance:")
        mse = mean_squared_error(yval, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(yval, y_pred)
        percent_diff = np.mean(np.abs((yval - y_pred) / yval)) * 100

        st.write(f"**R² Score:** {r2:.4f}")
        st.write(f"**RMSE:** {rmse:.2f}")
        st.write(f"**Mean % Error:** {percent_diff:.2f}%")

if __name__ == "__main__":
    main()
