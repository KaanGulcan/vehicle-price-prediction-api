# Vehicle Price Prediction API (XGBoost vs Linear Regression)

This is an API and Machine Learning (ML) project capable of predicting vehicle prices based on vehicle attributes (brand, model, transmission type, year, engine, etc.). To launch the API, you can run the `main.py` file using the command `uvicorn main:app --reload` in your terminal. You can access the Swagger interface at the `/docs` endpoint (http://127.0.0.1:8000/docs) to utilize the prediction models. The system includes two models: Linear Regression and XGBoost and you need to populate the request body as required to perform a prediction.After that you should execute the model and prediction will be apear on screen in json type

I made this project to learn the fundamentals of FastAPI and to improve my knowledge by comparing machine learning models.

## Technologies

- Python
- Numpy
- Pandas
- Scikit-Learn (Linear Regression)
- XGBoost
- Joblib
- FastAPI
- Uvicorn
- Pydantic

## Machine Learning

### XGBoost

- Data Reading
- Data Analysis
- Feature Engineering
- Data Cleaning
- Feature Selection
- Train/Test Split
- Preprocessor
- StandartScaler
- One-Hot Encoding
- Pipeline
- K-Fold Cross Validation
- GridSearchCV
- Train
- Prediction
- Evaluation
- Joblib

### Linear Regression

- Data Reading
- Data Analysis
- Feature Engineering
- Data Cleaning
- Feature Selection
- Train/Test Split
- Preprocessor
- StandartScaler
- One-Hot Encoding
- Pipeline
- Train
- Prediction
- Evaluation
- Joblib

## Dataset

The dataset is not included in this repository

Dataset source: https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho

## Evaluation Metrics

### XGBoost

- MAE: 82260.40
- RMSE: 154427.03
- R2: 0.97
- MAPE: 0.16

### Linear Regression

- MAE: 69512.38
- RMSE: 151643.64
- R2: 0.97
- MAPE: 0.16

## Türkçe Açıklama

Bu proje, araç özelliklerine (marka, model, şanzıman tipi, üretim yılı, motor vb.) dayanarak araç fiyatlarını tahmin edebilen bir API ve Makine Öğrenmesi (ML) uygulamasıdır. API'yi başlatmak için terminalinizde `uvicorn main:app --reload` komutunu kullanarak `main.py` dosyasını çalıştırabilirsiniz. Tahmin modellerini kullanmak üzere `/docs` uç noktası (http://127.0.0.1:8000/docs) üzerinden Swagger arayüzüne erişebilirsiniz. Sistem; Doğrusal Regresyon (Linear Regression) ve XGBoost olmak üzere iki model içermektedir; tahmin işlemi gerçekleştirmek için istek gövdesini (request body) gerekli bilgilerle doldurmanız gerekir. Ardından modeli çalıştırdığınızda, tahmin sonucu ekranda JSON formatında görüntülenecektir.

Bu projeyi, FastAPI'nin temellerini öğrenmek ve makine öğrenmesi modellerini karşılaştırarak bilgilerimi geliştirmek amacıyla hazırladım.