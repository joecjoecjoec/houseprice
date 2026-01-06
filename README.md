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

## 3. Exploratory data analysis (EDA)
The EDA is available in `notebook.ipynb`. It includes:
- target distribution analysis (`price_in_rupees`, log-transform via `log1p`)
- missing values analysis and cleaning strategy
- categorical feature exploration (location/transaction/facing/ownership, etc.)
- numerical feature ranges (bathroom/balcony/floor parsing)
- feature importance analysis (e.g., permutation importance / tree feature importance)

## 4. Model training and selection
Models were trained and evaluated in `notebook.ipynb`, including:
- baseline linear regression / ridge regression
- tree-based model(s) (e.g. RandomForest / XGBoost / CatBoost)

Model selection was based on validation performance (e.g., RMSE on log1p target).
The final model is trained in `train.py` and saved to `model.bin`.

## 5. How to run (reproducibility)

### 5.1 Create virtual environment and install dependencies
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
### 5.2 Train the model

```bash
python train.py
```
This creates model.bin.

### 5.3 Run the service locally (without Docker)

```bash 
gunicorn --bind 0.0.0.0:9696 predict:app
```
Test:
```bash
curl -i http://localhost:9696/health
```
```bash
curl -X POST http://localhost:9696/predict \
  -H "Content-Type: application/json" \
  -d '{
    "location": "thane",
    "transaction": "Resale",
    "facing": "East",
    "ownership": "Freehold",
    "floor": "1 out of 10",
    "bathroom": 2,
    "balcony": 1,
    "amountin_rupees": "42 Lac"
  }'
  ```

## 6. Run with Docker
### 6.1 Build the image

```bash
docker build -t houseprice .
```

### 6.2 Run the container
```bash
docker run -it --rm -p 9696:9696 houseprice
```

### 6.3 Test the container
```bash
curl -i http://localhost:9696/health
```

## 7. Cloud deployment (Render)
The service is deployed on Render:
	•	Base URL: https://houseprice-5aia.onrender.com
Test:
```bash
curl -i https://houseprice-5aia.onrender.com/health

curl -X POST https://houseprice-5aia.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "location": "thane",
    "transaction": "Resale",
    "facing": "East",
    "ownership": "Freehold",
    "floor": "1 out of 10",
    "bathroom": 2,
    "balcony": 1,
    "amountin_rupees": "42 Lac"
  }'
```
Example response:

```json
{"predicted_price_in_rupees": 12982.82743088388}
```

## 8. Repository structure
	•	notebook.ipynb — data preparation, EDA, feature importance, model selection
	•	train.py — train final model and save it to model.bin
	•	predict.py — Flask service for inference
	•	model.bin — serialized trained model
	•	requirements.txt — dependencies
	•	Dockerfile — container build for the service
	•	render.yaml — Render blueprint
	•	data/house_prices.csv — dataset (or download instructions above)
