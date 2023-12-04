import pandas as pd
import pickle
import os

from pathlib import Path
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

def load_data():
    # Define the path of the CSV file relative to the project root
    # csv_file = Path(__file__).parent / "data" / "bothawk_data.csv"
    import pkg_resources

    # 获取数据文件路径
    csv_file = pkg_resources.resource_filename('openperf', 'benchmarks/data_science/bot_detection/data/bothawk_data.csv')

    # Try to read the CSV file
    try:
        df = pd.read_csv(csv_file)
    except pd.errors.ParserError:
        print(f"Error: The file {csv_file} could not be parsed as a CSV file.")
        return
    # df['Number of Connection Account'].astype('int64')
    df.dropna()

    normalize = lambda x: (x - x.min()) / (x.max() - x.min())

    # normalize
    # normalize
    df[["following", "followers", "counts_of_activity",
        "counts_of_issue", "counts_of_pull_request", "counts_of_repository", "counts_of_commit",
        "counts_of_active_day", "periodicity_of_activities", "counts_of_connection_account",
        "median_response_time"]] = df[['following', 'followers', "counts_of_activity", "counts_of_issue",
                                    "counts_of_pull_request", "counts_of_repository", "counts_of_commit",
                                    "counts_of_active_day", "periodicity_of_activities",
                                    "counts_of_connection_account", "median_response_time"]].apply(normalize)

# Select the required features
    df = df[["login", "name", "email", "bio", "type", "followers", "following", "tfidf_similarity", "counts_of_activity",
            "counts_of_issue", "counts_of_pull_request", "counts_of_repository", "counts_of_commit",
            "counts_of_active_day", "periodicity_of_activities", "counts_of_connection_account",
            "median_response_time", 'label']]

    # Map the labels "Bot" and "Human" to the values 0 and 1
    bot_mapping = {'Human': 0, 'Bot': 1}
    df['label'] = df['label'].replace(bot_mapping)

    # Screen out positive and negative samples
    pos_samples = df[df['label'] == 1]
    neg_samples = df[df['label'] == 0]

    # Random downsampling
    neg_samples = neg_samples.sample(n=pos_samples.shape[0], replace=False, random_state=42)

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


def train_model(X_train, y_train, base_clf, params):
    best_estimators = []
    for i, (name, model) in enumerate(base_clf):
        print(f"Optimizing {name} with Bagging...")
        if i == 3:
            # XGBoost
            xgb_model = xgb.XGBClassifier()
            grid_search = GridSearchCV(xgb_model, params[i], cv=5, scoring='accuracy')
            grid_search.fit(X_train, y_train)
            # with open('model/baggingXGBoost.pickle', 'wb') as f:
            #     pickle.dump(grid_search.best_estimator_, f)
        else:
            # Define base estimator
            base_est = model
            # Define Bagging model with base estimator
            bagging = BaggingClassifier(base_est, random_state=0)
            # Define GridSearchCV object
            grid_search = GridSearchCV(bagging, params[i], cv=5, scoring='accuracy')
            # Fit the model
            grid_search.fit(X_train, y_train)
            # Save the best estimator
            # with open(f'model/bagging{name}.pickle', 'wb') as f:
            #     pickle.dump(grid_search.best_estimator_, f)
        best_estimators.append(grid_search.best_estimator_)
    return best_estimators

def evaluate_model(X_test, y_test, best_estimators, base_clf):
    eval_df = pd.DataFrame(columns=['Accuracy', 'Precision', 'Recall', 'F1-score', 'AUC'])
    for i, (name, model) in enumerate(base_clf):
        # Predict labels
        y_pred = best_estimators[i].predict(X_test)
        y_prob = best_estimators[i].predict_proba(X_test)[:, 1]
        # Compute evaluation metrics
        accuracy, precision, recall, f1, cm = get_evaluation_metrics(y_test, y_pred)
        # Calculate the point of the ROC curve
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_prob)
        pr_auc = auc(recall_curve, precision_curve)
        # Significance of statistical features
        # perm_importance = permutation_importance(best_estimators[i], X_test, y_test, n_repeats=5,
        #                                          random_state=0, n_jobs=-1)

        # sorted_idx = perm_importance.importances_mean.argsort()
        # Store the data in a DataFrame
        # feature_names = X_test.columns.tolist()
        # df_perm_imp = pd.DataFrame({'Feature': [feature_names[i] for i in sorted_idx],
        #                             'Feature Importance': perm_importance.importances_mean})
        # df_perm_imp.to_csv(f'result/bagging{name}_perm_imp.csv', index=False)
        # if not os.path.exists('result'):
        #     os.makedirs('result')
        # eval_df.loc['bagging' + name] = [accuracy, precision, recall, f1, pr_auc]
        # eval_df.to_csv(f'result/bagging{name}_metrics.csv')
        eval_df.loc['bagging' + name] = [accuracy, precision, recall, f1, pr_auc]

    # print(eval_df.head(10))
    return eval_df

def run():
    # load data
    X_train, X_test, y_train, y_test = load_data()

    base_clf = [
        ("DecisionTree", DecisionTreeClassifier()),
        ("KNeighbors", KNeighborsClassifier()),
        ("RandomForest", RandomForestClassifier()),
        ("XGBoost", xgb.XGBClassifier()),
        ("LogisticRegression", LogisticRegression()),
        ("SVC", SVC()),
        ("GaussianNB", GaussianNB())
    ]

    params = [
        {'base_estimator__max_depth': [3, 5, 7],
         'base_estimator__min_samples_split': [2, 4, 8]},
        {'base_estimator__n_neighbors': [3, 5, 7],
         'base_estimator__weights': ['uniform', 'distance']},
        {'base_estimator__n_estimators': [10, 50, 100],
         'base_estimator__max_depth': [3, 5, 7]},
        {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.1, 0.01, 0.001]
        },
        {'base_estimator__C': [0.1, 1.0, 10.0]},
        {},
        {}
    ]

    best_estimators = train_model(X_train, y_train, base_clf, params)
    return evaluate_model(X_test, y_test, best_estimators, base_clf)

if __name__ == "__main__":
    run()
