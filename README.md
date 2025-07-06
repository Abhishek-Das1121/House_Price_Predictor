🔍 Features
Predict house prices based on:

  Plot size (LotArea)

  Year built (YearBuilt)

  Overall condition (OverallCond)

# Interactive input form for real-time predictions

# Visual comparison: actual vs predicted prices (scatter plot)

# Model performance metrics: R² Score, RMSE, % error

# Deployed via Streamlit (optional for cloud deployment)
 
# 🌟How to Run Locally
  Clone the repo:
    git clone https://github.com/your-username/house-price-predictor.git
    cd house-price-predictor
    
  Install dependencies:
    pip install -r requirements.txt
    
  Run the app:
    streamlit run app.py
    
  Upload the following files to the root directory:

  train.csv
  test.csv

sample_submission.csv (optional)

📦 requirements.txt
ini
Copy
Edit
streamlit==1.29.0
pandas==2.1.4
numpy==1.26.2
scikit-learn==1.3.2
xgboost==2.0.3
plotly==5.18.0
