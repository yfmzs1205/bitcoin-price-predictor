import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def train_xgb_model(df, target_column):
    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    return model, acc, classification_report(y_test, y_pred, output_dict=True)

def predict_trend(model, latest_features_df):
    prediction = model.predict(latest_features_df)
    proba = model.predict_proba(latest_features_df)[0]
    return prediction[0], proba
