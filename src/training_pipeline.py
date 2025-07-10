from utils import load_data, preprocess_data, load_model, test_model#, train_model, save_model


# load data
df = load_data('data/exercise.csv', 'data/calories.csv')
# print(df.shape)


# split and preprocess data
x_train, x_test, y_train, y_test = preprocess_data(df, 'train')
# print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)


# train model
# from sklearn.ensemble import RandomForestRegressor
# rfr = RandomForestRegressor(random_state=42, n_jobs=5)
# rfr = train_model(rfr, x_train, y_train)


# save trained model
# save_model(rfr, 'models/rfr_fitted.joblib')


# test model performance
rfr = load_model('models/trained_RandomForestRegressor.joblib')

model_acc = test_model(rfr, x_test, y_test)
print(f'Model Accuracy: {round(model_acc*100, 2)} %')

