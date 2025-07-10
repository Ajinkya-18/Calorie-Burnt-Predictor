from utils import get_inference_data, preprocess_data, load_model

# get inference data
data = get_inference_data()

# transform and scale the data
data_scaled = preprocess_data(data, mode='infer')

# pass the data to a trained model for making inference
model = load_model('models/trained_LinearRegression.joblib')
y_hat = model.predict(data_scaled)

print(f'Calories burnt: {round(y_hat[0], 1)}')
