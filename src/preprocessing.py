from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib

def build_preprocessor(numeric_features, categorical_features):
    num_pipe = Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('scale', RobustScaler())
    ])
    cat_pipe = Pipeline([
        ('impute', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False))
    ])
    preprocessor = ColumnTransformer([
        ('num', num_pipe, numeric_features),
        ('cat', cat_pipe, categorical_features)
    ], remainder='drop')
    return preprocessor

def get_feature_names(preprocessor):
    raw = preprocessor.get_feature_names_out()
    cleaned = [r.split('__',1)[-1] for r in raw]
    return cleaned

def save_preprocessor(preprocessor, path):
    joblib.dump(preprocessor, path)

def load_preprocessor(path):
    return joblib.load(path)
