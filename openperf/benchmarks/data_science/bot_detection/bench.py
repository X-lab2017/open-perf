import pandas as pd
import pickle
import os

from pathlib import Path

import pkg_resources
from imblearn.under_sampling import RandomUnderSampler
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, \
    precision_recall_curve, roc_curve, auc
import xgboost as xgb
import numpy as np
import openperf.benchmarks.data_science.bot_detection.bothunter_bench as bothunter_bench
import openperf.benchmarks.data_science.bot_detection.BoDeGHa_bench as BoDeGHa_bench

def load_data():
    # Define the path of the CSV file relative to the project root
    # csv_file = Path(__file__).parent / "data" / "bothawk_data.csv"
    import pkg_resources

    # 获取数据文件路径
    csv_file = pkg_resources.resource_filename('openperf',
                                               'benchmarks/data_science/bot_detection/data/bothawk_data.csv')

    # Try to read the CSV file
    try:
        df = pd.read_csv(csv_file)
    except pd.errors.ParserError:
        print(f"Error: The file {csv_file} could not be parsed as a CSV file.")
        return
    df.dropna()

    normalize = lambda x: (x - x.min()) / (x.max() - x.min())

    # Select the required features
    df[[
        "Number of following", "Number of followers", "Number of Activity",
        "Number of Issue", "Number of Pull Request", "Number of Repository", "Number of Commit",
        "Number of Active day", "Periodicity of Activities", "Number of Connection Account",
        "Median Response Time"
    ]] = df[[
        "Number of following", "Number of followers", "Number of Activity",
        "Number of Issue", "Number of Pull Request", "Number of Repository", "Number of Commit",
        "Number of Active day", "Periodicity of Activities", "Number of Connection Account",
        "Median Response Time"
    ]].apply(normalize)

    df = df[[
        "login", "name", "email", "bio", "tag",
        "Number of followers", "Number of following", "tfidf_similarity",
        "Number of Activity", "Number of Issue", "Number of Pull Request",
        "Number of Repository", "Number of Commit", "Number of Active day",
        "Periodicity of Activities", "Number of Connection Account",
        "Median Response Time", 'label'
    ]]

    # Map the labels "Bot" and "Human" to the values 0 and 1
    bot_mapping = {'Human': 0, 'Bot': 1}
    df['label'] = df['label'].replace(bot_mapping)

    # Screen out positive and negative samples
    pos_samples = df[df['label'] == 1]
    neg_samples = df[df['label'] == 0]

    # Random downsampling
    # neg_samples = neg_samples.sample(n=pos_samples.shape[0], replace=False, random_state=42)

    # Combine positive and negative samples
    df = pd.concat([pos_samples, neg_samples])

    # Define features and labels
    X = df.drop('label', axis=1)
    y = df['label']

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test


def get_evaluation_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    return accuracy, precision, recall, f1, cm
def train_model(X_train, y_train, base_clf, params, use_bagging=True):
    best_estimators = []
    for i, (name, model) in enumerate(base_clf):
        if use_bagging:
            rus = RandomUnderSampler(random_state=42)
            X_train_res, y_train_res = rus.fit_resample(X_train, y_train)
            if name == 'XGBoost':
                grid_search = GridSearchCV(model, params[i], cv=5, scoring='accuracy')
                grid_search.fit(X_train, y_train)
                best_estimators.append(grid_search.best_estimator_)
            else:
                print('training bothawk')
                bagging = BaggingClassifier(base_estimator=model, random_state=0)
                if params[i]:
                    grid_search = GridSearchCV(bagging, params[i], cv=5, scoring='accuracy')
                    grid_search.fit(X_train_res, y_train_res)
                    best_estimators.append(grid_search.best_estimator_)
                else:
                    # 如果没有特定参数，直接训练
                    bagging.fit(X_train_res, y_train_res)
                    best_estimators.append(bagging)
        else:
            if params[i] and name != "XGBoost":
                print(f'training {name}')
                print(f"Using parameters for {name}: {params[i]}")
                grid_search = GridSearchCV(model, params[i], cv=5, scoring='accuracy')
                grid_search.fit(X_train, y_train)
                best_estimators.append(grid_search.best_estimator_)
            else:
                # 如果没有参数，则直接训练
                model.fit(X_train, y_train)
                best_estimators.append(model)

    return best_estimators


def evaluate_model(X_test, y_test, best_estimators, base_clf, prefix=""):
    eval_df = pd.DataFrame(columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC'])
    best_score = 0
    best_model_name = ""
    data_frames = []

    for i, (name, model) in enumerate(base_clf):
        if best_estimators[i] is not None:  # 确保估计器有效
            y_pred = best_estimators[i].predict(X_test)
            y_prob = best_estimators[i].predict_proba(X_test)[:, 1]
            accuracy, precision, recall, f1, cm = get_evaluation_metrics(y_test, y_pred)
            precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_prob)
            pr_auc = auc(recall_curve, precision_curve)
            model_name = prefix + name
            current_df = pd.DataFrame({
                'Model': [model_name],
                'Accuracy': [accuracy],
                'Precision': [precision],
                'Recall': [recall],
                'F1 Score': [f1],
                'AUC': [pr_auc]
            })
            data_frames.append(current_df)
            if f1 > best_score:
                best_score = f1
                best_model_name = model_name

    eval_df = pd.concat(data_frames, ignore_index=True)
    return eval_df, best_model_name


def basic_bench():
    X_train, X_test, y_train, y_test = load_data()

    base_clf = [
        ("DecisionTree", DecisionTreeClassifier()),
        ("KNeighbors", KNeighborsClassifier()),
        ("RandomForest", RandomForestClassifier()),
        ("XGBoost", xgb.XGBClassifier()),
        ("LogisticRegression", LogisticRegression()),
        ("SVC", SVC(probability=True)),
        ("GaussianNB", GaussianNB())
    ]

    # 不使用Bagging的情况
    params_without_bagging = [
        {'max_depth': [3, 5, 7], 'min_samples_split': [2, 4, 8]},  # DecisionTree
        {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']},  # KNeighbors
        {'n_estimators': [10, 50, 100], 'max_depth': [3, 5, 7]},  # RandomForest
        {'n_estimators': [100, 200, 300], 'max_depth': [3, 5, 7], 'learning_rate': [0.1, 0.01, 0.001]},  # XGBoost
        {'C': [0.1, 1.0, 10.0]},  # LogisticRegression
        {'C': [0.1, 1.0, 10.0], 'kernel': ['linear', 'rbf', 'poly']},  # SVC
        {}  # GaussianNB
    ]

    # 使用Bagging的情况
    params_with_bagging = [
        {'base_estimator__max_depth': [3, 5, 7], 'base_estimator__min_samples_split': [2, 4, 8]},
        {'base_estimator__n_neighbors': [3, 5, 7], 'base_estimator__weights': ['uniform', 'distance']},
        {'base_estimator__n_estimators': [10, 50, 100], 'base_estimator__max_depth': [3, 5, 7]},
        {'base_estimator__n_estimators': [100, 200, 300], 'base_estimator__max_depth': [3, 5, 7], 'base_estimator__learning_rate': [0.1, 0.01, 0.001]},
        {'base_estimator__C': [0.1, 1.0, 10.0]},
        {'base_estimator__C': [0.1, 1.0, 10.0], 'base_estimator__kernel': ['linear', 'rbf', 'poly']},
        {}
    ]

    # 训练非bagging模型
    non_bagging_estimators = train_model(X_train, y_train, base_clf, params_without_bagging, use_bagging=False)
    non_bagging_results, _ = evaluate_model(X_test, y_test, non_bagging_estimators, base_clf)

    # 训练bagging模型
    bagging_estimators = train_model(X_train, y_train, base_clf, params_with_bagging, use_bagging=True)
    bagging_results, best_bagging_model = evaluate_model(X_test, y_test, bagging_estimators, base_clf, prefix="bagging")

    # 标记最佳的bagging模型为"bot_hawk"
    bagging_results.loc[bagging_results['Model'] == best_bagging_model, 'Model'] = "bot_hawk"

    final_results = pd.concat([non_bagging_results, bagging_results])

    return final_results

def run():
    bothunter_csv_file = pkg_resources.resource_filename('openperf',
                                                         'benchmarks/data_science/bot_detection/data/bothawk_bothunter_data.csv')
    BoDeGHa_csv_file = pkg_resources.resource_filename('openperf',
                                                       'benchmarks/data_science/bot_detection/data/bothawk_BoDeGHa_data.csv')

    # Run the basic bench to get initial results
    results_original = basic_bench()

    print("training bothunter")
    results_bo = bothunter_bench.run_model_analysis(bothunter_csv_file)
    print("training BoDeGHa")
    results_bh = BoDeGHa_bench.run_model_analysis(BoDeGHa_csv_file)
    print("finish training")

    # Convert results to DataFrame and specify the columns
    results_bo_df = pd.DataFrame([results_bo])
    results_bh_df = pd.DataFrame([results_bh])

    # Add 'Model' labels if not included in original output
    results_bo_df['Model'] = 'BotHunter'
    results_bh_df['Model'] = 'BoDeGHa'

    # Combine all results into a single DataFrame
    all_results = pd.concat([results_original, results_bo_df, results_bh_df], ignore_index=True)

    # Ensure all necessary columns are present and select them
    required_columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC']
    # Fill missing columns with NaN where necessary
    for column in required_columns:
        if column not in all_results.columns:
            all_results[column] = np.nan

    # Select only the required columns
    all_results = all_results[required_columns]

    all_results = all_results[~all_results['Model'].str.contains("bagging|XGBoost", case=False, regex=True)]

    return all_results

if __name__ == "__main__":
    final_results = run()
    print(final_results)
