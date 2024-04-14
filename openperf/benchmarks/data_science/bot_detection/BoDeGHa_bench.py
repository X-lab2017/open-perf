import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.under_sampling import RandomUnderSampler

def run_model_analysis(filepath):
    # Load the data
    df = pd.read_csv(filepath, index_col=0)
    df['label'] = df['label'].replace({'Human': 0, 'Bot': 1}).astype(int)

    # Normalize only numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].apply(lambda x: (x - x.min()) / (x.max() - x.min()))


    # Check and handle NaNs after normalization
    if df.isna().any().any():
        print("NaNs found after normalization, filling with zeros")
        df.fillna(0, inplace=True)

    df = df[['comment_type_num','comment_num','empty_comment_num','Issuecomment_num','PullRequestReviewcomment_num','Commitcomment_num','label']]
    # Splitting data
    labels = df['label']
    features = df[['comment_type_num','comment_num','empty_comment_num','Issuecomment_num','PullRequestReviewcomment_num','Commitcomment_num']]
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.3, random_state=42)

    # Balance the dataset
    rus = RandomUnderSampler(random_state=42)
    train_features_rus, train_labels_rus = rus.fit_resample(train_features, train_labels)

    # Model training with hyperparameter tuning
    rfc = RandomForestClassifier()
    param_grid = {
        'n_estimators': [10, 50, 100],
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False],
        'criterion': ['gini', 'entropy']
    }
    grid_search = GridSearchCV(rfc, param_grid=param_grid, cv=5)
    grid_search.fit(train_features_rus, train_labels_rus)

    # Evaluating the model
    y_pred = grid_search.predict(test_features)
    y_prob = grid_search.predict_proba(test_features)[:, 1]
    auc_score = roc_auc_score(test_labels, y_prob)

    accuracy = accuracy_score(test_labels, y_pred)
    precision = precision_score(test_labels, y_pred, pos_label=1)
    recall = recall_score(test_labels, y_pred, pos_label=1)
    f1 = f1_score(test_labels, y_pred, pos_label=1)

    # Output evaluation results
    results = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'AUC': auc_score
    }

    return results

# Usage example
if __name__ == '__main__':
    filepath = './data/bothawk_BoDeGHa_data.csv'  # Update the file path
    results = run_model_analysis(filepath)
    print(results)
