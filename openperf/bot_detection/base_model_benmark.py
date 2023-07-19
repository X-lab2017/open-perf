import pandas as pd
import numpy as np
import time
import warnings

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, roc_curve, auc, precision_recall_curve, roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline as imbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")  # eliminate warning messages


def load_data(filepath):
    # read CSV file
    df = pd.read_csv(filepath)
    RANGE_COUNT = 1

    bot_mapping = {'Human': 0, 'Bot': 1}
    df['label'] = df['label'].replace(bot_mapping)

    df.dropna()
    # 分离特征与标签
    y = df.pop('label')
    X = df[["login", "name", "email", "bio", "tag", "followers", "following", "tfidf_similarity", "Number of Activity",
            "Number of Issue", "Number of Pull Request", "Number of Repository", "Number of Commit",
            "Number of Active day", "Periodicity of Activities", "Number of Connection Account",
            "Median Response Time"]]
    print(X.dtypes)

    for col in X.columns:
        col_type = X[col].dtype
        if col_type != np.float64 and col_type != np.int64:
            print(f'The column "{col}" has a non-numeric data type: {col_type}')

    features = X.columns
    return X, y, features


def preprocess_data(X, y, features):
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', MinMaxScaler())
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, features)
    ])

    lr_pipeline = imbPipeline([
        ("processor", preprocessor),
        ('sampler', RandomUnderSampler()),
        ("classifier", LogisticRegression(max_iter=10000))
    ])

    dt_pipeline = imbPipeline([
        ("processor", preprocessor),
        ('sampler', RandomUnderSampler()),
        ('classifier', DecisionTreeClassifier(max_leaf_nodes=None, min_impurity_decrease=0.0))
    ])

    svm_pipeline = imbPipeline([
        ("processor", preprocessor),
        ('sampler', RandomUnderSampler()),
        ('classifier', SVC(kernel='rbf', probability=True))
    ])

    nb_pipeline = imbPipeline([
        ("processor", preprocessor),
        ('sampler', RandomUnderSampler()),
        ("classifier", GaussianNB())
    ])

    knn_pipeline = imbPipeline([
        ("processor", preprocessor),
        ('sampler', RandomUnderSampler()),
        ("classifier", KNeighborsClassifier())
    ])

    rf_pipeline = imbPipeline([
        ("processor", preprocessor),
        ('sampler', RandomUnderSampler()),
        ("classifier", RandomForestClassifier())
    ])

    model_list = [
        lr_pipeline,
        dt_pipeline,
        svm_pipeline,
        nb_pipeline,
        knn_pipeline,
        rf_pipeline
    ]

    return model_list


def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(
        f'Model: {model} | Accuracy: {accuracy:.3f} | Precision: {precision:.3f} | Recall: {recall:.3f} | F1-score: {f1:.3f}')
    model_name = model.named_steps['classifier'].__class__.__name__
    y_pred_proba = model.predict_proba(X_test)[:,1]

    y_prob = model.predict_proba(X_test)[:, 1]

    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_prob)

    pr_auc = auc(recall_curve, precision_curve)

    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

    roc_df = pd.DataFrame({
        'False Positive Rate': fpr,
        'True Positive Rate': tpr
    })

    roc_df.to_csv(f'./result/{model_name}_roc_curve_data.csv', index=False)

    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_df = pd.DataFrame({
        'Precision': precision_curve,
        'Recall': recall_curve
    })



    pr_df.to_csv(f'./result/{model_name}_pr_curve_data.csv', index=False)


    return {'Model': model.named_steps['classifier'].__class__.__name__, 'Accuracy': accuracy, 'Precision': precision,
            'Recall': recall, 'F1-score': f1, 'AUC': pr_auc}


def train_and_evaluate_model(model_list, X_train, X_test, y_train, y_test):
    output_data = []
    for model in model_list:
        output_data.append(evaluate_model(model, X_train, X_test, y_train, y_test))
    return pd.DataFrame(output_data)


def main():
    filepath = "./data/bothawk_data.csv"
    X, y, features = load_data(filepath)
    model_list = preprocess_data(X, y, features)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)
    df_output = train_and_evaluate_model(model_list, X_train, X_test, y_train, y_test)
    df_output.to_csv('./result/model_evaluation_v1.csv', index=False)


if __name__ == '__main__':
    main()
