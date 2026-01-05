# House Price Prediction

## 1. Problem Description

### Goal
Build a machine learning model to **predict house prices (INR)** from real-estate listing attributes.  
Given a listing (location, transaction type, ownership, facing, floor, number of bathrooms/balconies, etc.), the model returns an estimated **price in rupees**.

### Why this is useful
Online listings often have inconsistent pricing. A reliable price estimate can help:
- **Buyers** quickly compare listings and identify potentially overpriced properties
- **Sellers/agents** set a more reasonable price range for new listings
- **Platforms** improve search ranking, recommendations, and pricing consistency checks

### ML task
- **Type:** Supervised learning — **Regression**
- **Target variable (raw):** `price_in_rupees`
- **Prediction service output:** `predicted_price_in_rupees`

### How the model is used
The trained model is packaged into a small **Flask web service** with an endpoint:

- `POST /predict`  
  Input: JSON with listing features  
  Output: JSON with the predicted price in rupees

Example output:
```json
{"predicted_price_in_rupees": 12651.98}
```

### Data source
- **Source (Kaggle):** https://www.kaggle.com/datasets/juhibhojani/house-price
- **Local file used in this repo:** `data/house_prices.csv`
- **Development environment:** VS Code (local)

### Why log1p on the target
The raw target `price_in_rupees` is **highly skewed with a long tail** (very expensive properties are rare but extreme).  
To reduce the impact of outliers and stabilize the scale, the model is trained to predict `log1p(price_in_rupees)`.

At inference time, the service converts the prediction back to rupees using `expm1(pred_log)`.

