import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.under_sampling import RandomUnderSampler

def run_model_analysis(filepath):
    """
    Run the model analysis on the specified CSV file path.

    Args:
        filepath (str): The file path to the CSV file containing the data.

    Returns:
        dict: A dictionary containing various evaluation metrics.
    """
    try:
        # Load and preprocess the data
        df = pd.read_csv(filepath, index_col=0)
        df['label'] = df['label'].replace({'Human': 0, 'Bot': 1}).astype(int)
        normalize = lambda x: (x - x.min()) / (x.max() - x.min())
        df = df[['name','login','bio','type','following','followers','issue_id_num','repo_id_num','unique_issue_num','unique_repo_num','unique_pr_num','issue_type_num','pr_type_num','repo_type_num','commit_type_num','label']]
        labels = df['label']

        # Handling non-numeric data and missing values
        if df.select_dtypes(include=[np.number]).empty:
            df = df.astype(float)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)

        # Splitting data
        features = df[['name','login','bio','type','following','followers','issue_id_num','repo_id_num','unique_issue_num','unique_repo_num','unique_pr_num','issue_type_num','pr_type_num','repo_type_num','commit_type_num']].apply(normalize)
        print(labels.shape)
        print(features.shape)
        train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.3, random_state=42)

        # Balancing dataset
        rus = RandomUnderSampler(random_state=42)
        train_features_rus, train_labels_rus = rus.fit_resample(train_features, train_labels)

        # Model training
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
        y_probs = grid_search.predict_proba(test_features)[:, 1]  # Probability estimates for the positive class

        accuracy = accuracy_score(test_labels, y_pred)
        precision = precision_score(test_labels, y_pred, pos_label=1)
        recall = recall_score(test_labels, y_pred, pos_label=1)
        f1 = f1_score(test_labels, y_pred, pos_label=1)
        auc = roc_auc_score(test_labels, y_probs)  # AUC score

        # Output evaluation results
        results = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'AUC': auc  # Added AUC
        }
        return results

    except Exception as e:
        print(f"An error occurred: {e}")
        return {}

# Example usage
if __name__ == '__main__':
    results = run_model_analysis('data/bothawk_bothunter_data.csv')
    print(results)
