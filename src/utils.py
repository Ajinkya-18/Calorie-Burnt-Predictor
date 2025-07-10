def load_data(exercises_data_path:str, calories_data_path:str):
    import os
    from pathlib import Path

    cwd = os.getcwd()
    full_calories_data_path = os.path.join(cwd, Path(calories_data_path))
    full_exercises_data_path = os.path.join(cwd, Path(exercises_data_path))

    if os.path.exists(full_calories_data_path) and full_calories_data_path.endswith('.csv') and os.path.exists(full_exercises_data_path) and full_exercises_data_path.endswith('.csv'):
        import pandas as pd

        calories_df = pd.read_csv(full_calories_data_path)
        exercises_df = pd.read_csv(full_exercises_data_path)
        df = pd.merge(left=exercises_df, right=calories_df, how='outer')
        df.drop(['User_ID'], axis=1, inplace=True)

        return df
    
    else:
        raise ValueError('Invalid data path or file extension.')
    
#------------------------------------------------------------------------------------------------------------

def preprocess_data(df, mode:str='train'):
    df['BMI'] = df['Weight'] / ((df['Height']/100)**2)
    df.drop(['Weight', 'Height'], axis=1, inplace=True)
    
    col_transformer = load_model('models/column_transformer_fitted.joblib')

    if mode == 'train':
        x_train, x_test, y_train, y_test = split_data(df)

        x_train_scaled = col_transformer.transform(x_train)
        x_test_scaled = col_transformer.transform(x_test)

        x_train_scaled.drop(['Gender'], axis=1, inplace=True)
        x_test_scaled.drop(['Gender'], axis=1, inplace=True)

        return x_train_scaled, x_test_scaled, y_train, y_test
    
    if mode == 'infer':
        df_scaled = col_transformer.transform(df)
        df_scaled.drop(['Gender'], axis=1, inplace=True)

        return df_scaled
    
    else:
        raise ValueError('Invalid "mode" value passed.')


#-----------------------------------------------------------------------------------------------------------------

def load_model(saved_model_path:str):
    import os
    from pathlib import Path

    cwd = os.getcwd()
    full_saved_model_path = os.path.join(cwd, Path(saved_model_path))

    if os.path.exists(full_saved_model_path) and full_saved_model_path.endswith('.joblib'):
        from joblib import load
        with open(full_saved_model_path, 'rb') as f:
            model = load(f)

            return model
        
    else:
        raise ValueError('Invalid file path or file extension.')

#------------------------------------------------------------------------------------------------------------------

def save_model(model_object, model_save_path:str):
    import os
    from pathlib import Path

    cwd = os.getcwd()
    full_model_save_path = os.path.join(cwd, Path(model_save_path))

    if os.path.exists(full_model_save_path) and full_model_save_path.endswith('.joblib'):
        from joblib import dump
        with open(full_model_save_path, 'wb') as f:
            dump(model_object, f)
            print('Model saved successfully.')
    else:
        raise ValueError('Invalid file path or file extension.')

#-------------------------------------------------------------------------------------------------------------

def split_data(df):
    from sklearn.model_selection import train_test_split

    X, Y = df.drop(['Calories'], axis=1), df['Calories']
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

    return x_train, x_test, y_train, y_test

#-----------------------------------------------------------------------------------------------------------------

def get_best_features(best_features_path:str):

    return load_model(best_features_path)

#-----------------------------------------------------------------------------------------------------------------------

def train_model(model_instance, x_train, y_train):
    try:
        model_instance.fit(x_train, y_train)

        return model_instance
    
    except Exception as e:
        print(e)

        raise e
    
def test_model(model_instance, x_test, y_test):
    from sklearn.metrics import r2_score

    try:
        y_preds = model_instance.predict(x_test)
        score = r2_score(y_test.values.reshape(-1, 1), y_preds.reshape(-1, 1))
        return float(score)
    
    except Exception as e:
        print(e)

        raise e
    
#----------------------------------------------------------------------------------------------------------------------

def get_inference_data():
    from random import choice, uniform
    import pandas as pd

    inference_vals = {
        'Gender': [choice(['male', 'female'])], 
        'Age': [choice(range(18, 80))], 
        'Duration': [round(uniform(1.0, 30.0), 1)],
        'Heart_Rate': [round(uniform(60, 180), 0)],
        'Body_Temp': [round(uniform(35, 43), 2)],
        'Height': [round(uniform(120, 230), 2)],
        'Weight': [round(uniform(30, 135), 2)]
        }

    inference_df = pd.DataFrame.from_dict(inference_vals)
    
    return inference_df

#-------------------------------------------------------------------------------------------------------------------------

