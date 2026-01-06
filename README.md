# House Price Prediction

## 1. Problem Description

### Goal
Build a machine learning model to **predict house prices (INR)** from real-estate listing attributes.  
Given a listing (location, transaction type, ownership, facing, floor, number of bathrooms/balconies, etc.), the model returns an estimated **price in rupees**.

### Why this is useful
Online listings often have inconsistent pricing. A reliable price estimate can help:
- **Buyers** compare listings and spot potentially overpriced properties
- **Sellers/agents** set a more reasonable price range for new listings
- **Platforms** improve ranking, recommendations, and price consistency checks

### ML task
- **Type:** Supervised learning — **Regression**
- **Target variable (raw):** `price_in_rupees`
- **Prediction service output:** `predicted_price_in_rupees`

### Data source
- **Dataset:** House price listings (India)
- **Source (Kaggle):** https://www.kaggle.com/datasets/juhibhojani/house-price
- **File used in this repo:** `data/house_prices.csv`
- **Development environment:** VS Code (local)

### Target transformation (log1p)
The raw target `price_in_rupees` is **highly skewed with a long tail** (very expensive properties are rare but extreme).  
To reduce the impact of outliers and stabilize the scale, the model is trained to predict `log1p(price_in_rupees)`.

At inference time, the service converts the prediction back to rupees using `expm1(pred_log)`.

### How the model is used
The trained model is packaged into a small **Flask web service**:

- `POST /predict`  
  Input: JSON with listing features  
  Output: JSON with the predicted price in rupees

Example output:
```json
{"predicted_price_in_rupees": 12651.98}
```

## 2. Dependency & Environment Management

This project uses **Python virtual environment (venv)** and a `requirements.txt` file.

### Setup

1. Clone the repository:

```bash
git clone https://github.com/joecjoecjoec/houseprice.git
cd houseprice
```

2. Create and activate a virtual environment:
```bash 
python -m venv .venv
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 3. Exploratory data analysis (EDA)

The EDA is available in `notebook.ipynb`. It includes:
- target distribution analysis (`price_in_rupees`, log-transform via `log1p`)
- missing values analysis and cleaning strategy
- categorical feature exploration (location/transaction/facing/ownership, etc.)
- numerical feature ranges (bathroom/balcony/floor parsing)
- feature importance analysis (e.g., permutation importance / tree feature importance)

The notebook (`notebook.ipynb`) explores the dataset and the main drivers of price.

Key findings:

- **Target distribution:** `price_in_rupees` is extremely right-skewed with a long tail.  
  After applying `log1p(price_in_rupees)`, the distribution becomes much more stable and closer to bell-shaped, which is better suited for regression models.

- **Bathrooms vs price:** listings with more bathrooms tend to have higher `log1p(price)` (clear upward trend), so `bathroom` is an important numeric signal.

- **Status feature:** `status` is dominated by **"Ready to Move"** (169,260 rows), while **"unknown"** is rare (595 rows).  
  The `log1p(price)` distribution differs by status: "Ready to Move" shows wider spread and more high-price outliers, suggesting `status` can carry useful signal (but comparisons for "unknown" are less reliable due to small sample size).

(Plots and detailed exploration are in the notebook.)

## 4. Data Preparation

Data preparation is implemented in `notebook.ipynb` and reused in `train.py`.

- **Train/validation/test split:** the dataset is split into train/validation/test sets for model selection and final evaluation.

- **Target transformation:** the raw target `price_in_rupees` is transformed with:
  - `y = log1p(price_in_rupees)`
  This reduces skew and makes the regression problem more stable.

- **Feature selection:** the model uses a mix of numeric and categorical features (e.g. location, transaction, facing, ownership, floor, status, bathroom, balcony, amountin_rupees).

- **DictVectorizer (one-hot encoding):**
  - Convert each row to a Python dict (`to_dict(orient="records")`)
  - Fit `DictVectorizer(sparse=True)` on training data
  - Transform validation/test using the same fitted vectorizer
  This produces a sparse feature matrix suitable for linear models.

- **Baseline modeling setup:** the same prepared matrices are used to train and compare multiple models:
  - Linear Regression baseline
  - Decision Tree Regressor
  - Random Forest Regressor
  Evaluation is done with RMSE in log-space.

## 5. Model training and selection

Metric: **RMSE in log-space** (RMSE on `log1p(price_in_rupees)`), evaluated on validation and test sets.

Models tried:

- **Linear Regression (baseline)** with one-hot encoded categorical features (`DictVectorizer`)
  - Validation RMSE ≈ **0.186**
  - Test RMSE ≈ **0.181**
  - Baseline already generalizes well (val/test are close).

- **Decision Tree Regressor**
  - Validation RMSE ≈ **0.325**
  - Performs significantly worse than linear baseline.

- **Random Forest Regressor**
  - Validation RMSE ≈ **0.259**
  - Test RMSE ≈ **0.253**
  - Also worse than linear models under the current sparse one-hot feature representation.

Ridge regression tuning:

- Tried multiple values of the regularization strength `alpha` and selected the best one on the validation set.
- Best configuration: **Ridge(alpha=0.1)**
  - Validation RMSE ≈ **0.1856**
  - Test RMSE ≈ **0.1805**

Final model:

- Trained the final Ridge model on **train + validation** data and evaluated on the test set.
- Final test RMSE (log-space) ≈ **0.1779**.
- Ridge was chosen because it matches the sparse one-hot setup well and adds regularization for better stability without sacrificing accuracy.

## 6. How to run (reproducibility)

### 6.1 Create virtual environment and install dependencies
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
### 6.2 Train the model

```bash
python train.py
```
This creates model.bin.

### 6.3 Run the service locally (without Docker)

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

## 7. Run with Docker
### 7.1 Build the image

```bash
docker build -t houseprice .
```

### 7.2 Run the container
```bash
docker run -it --rm -p 9696:9696 houseprice
```

### 7.3 Test the container
```bash
curl -i http://localhost:9696/health
```

## 8. Cloud deployment (Render)
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

## 9. Repository structure
	•	notebook.ipynb — data preparation, EDA, feature importance, model selection
	•	train.py — train final model and save it to model.bin
	•	predict.py — Flask service for inference
	•	model.bin — serialized trained model
	•	requirements.txt — dependencies
	•	Dockerfile — container build for the service
	•	render.yaml — Render blueprint
	•	data/house_prices.csv — dataset (or download instructions above)
