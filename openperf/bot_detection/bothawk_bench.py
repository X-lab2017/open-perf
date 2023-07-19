import pandas as pd
import pickle

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
    # read CSV file
    df = pd.read_csv("data/bothawk_data.csv")
    # df['Number of Connection Account'].astype('int64')
    df.dropna()

    normalize = lambda x: (x - x.min()) / (x.max() - x.min())

    # normalize
    df[["following", "followers", "Number of Activity",
        "Number of Issue", "Number of Pull Request", "Number of Repository", "Number of Commit",
        "Number of Active day", "Periodicity of Activities", "Number of Connection Account",
        "Median Response Time"]] = df[['following', 'followers', "Number of Activity", "Number of Issue",
                                       "Number of Pull Request", "Number of Repository", "Number of Commit",
                                       "Number of Active day", "Periodicity of Activities",
                                       "Number of Connection Account", "Median Response Time"]].apply(normalize)

    # Select the required features
    df = df[["login", "name", "email", "bio", "tag", "followers", "following", "tfidf_similarity", "Number of Activity",
             "Number of Issue", "Number of Pull Request", "Number of Repository", "Number of Commit",
             "Number of Active day", "Periodicity of Activities", "Number of Connection Account",
             "Median Response Time", 'label']]

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


def preprocess_data(X):
    return X


def get_evaluation_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    return accuracy, precision, recall, f1, cm


def train_and_evaluate_model(X_train, X_test, y_train, y_test):
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

    eval_df = pd.DataFrame(columns=['Accuracy', 'Precision', 'Recall', 'F1-score', 'AUC'])

    for i, (name, model) in enumerate(base_clf):
        print(f"Optimizing {name} with Bagging...")

        if i == 3:
            # XGBoost
            xgb_model = xgb.XGBClassifier()
            grid_search = GridSearchCV(xgb_model, params[i], cv=5, scoring='accuracy')
            grid_search.fit(X_train, y_train)

            with open('model/baggingXGBoost.pickle', 'wb') as f:
                pickle.dump(grid_search.best_estimator_, f)

            y_pred = grid_search.predict(X_test)
            y_prob = grid_search.predict_proba(X_test)[:, 1]
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
            with open(f'model/bagging{name}.pickle', 'wb') as f:
                pickle.dump(grid_search.best_estimator_, f)

            # Predict labels
            y_pred = grid_search.predict(X_test)
            y_prob = grid_search.predict_proba(X_test)[:, 1]

        # Compute evaluation metrics
        accuracy, precision, recall, f1, cm = get_evaluation_metrics(y_test, y_pred)

        # Calculate the point of the ROC curve
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)

        precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_prob)

        pr_auc = auc(recall_curve, precision_curve)

        # Significance of statistical features
        perm_importance = permutation_importance(grid_search.best_estimator_, X_test, y_test, n_repeats=5,
                                                 random_state=0, n_jobs=-1)
        sorted_idx = perm_importance.importances_mean.argsort()

        # Store the data in a DataFrame


        df_roc = pd.DataFrame({'False Positive Rate': fpr, 'True Positive Rate': tpr, 'Thresholds': thresholds})
        df_pr = pd.DataFrame({'Precision': precision_curve, 'Recall': recall_curve})
        # df_perm_imp = pd.DataFrame({'Sorted Index': sorted_idx, 'Feature Importance': perm_importance.importances_mean})
        feature_names = X_train.columns.tolist()
        df_perm_imp = pd.DataFrame({'Feature': [feature_names[i] for i in sorted_idx],
                                    'Feature Importance': perm_importance.importances_mean})
        df_roc.to_csv(f'bagging{name}_roc_curve_data.csv', index=False)
        df_pr.to_csv(f'bagging{name}_pr_curve_data.csv', index=False)
        df_perm_imp.to_csv(f'result/bagging{name}_perm_imp.csv', index=False)

        eval_df.loc['bagging' + name] = [accuracy, precision, recall, f1, pr_auc]
        eval_df.to_csv(f'result/bagging{name}_metrics.csv')

        y_pred = grid_search.predict(X_test)
        y_pred_proba = grid_search.predict_proba(X_test)[:, 1]

        results_df = pd.DataFrame({
            'y_test': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        })
        results_df.to_csv(f'./result/bagging{name}_test_results.csv', index=False)

        print(f"Best parameters for {name}: {grid_search.best_params_}")

        print(f"{name} Classifier Evaluation Metrics:")
        print("Accuracy: %.2f%%" % (accuracy * 100.0))
        print("Precision: %.2f%%" % (precision * 100.0))
        print("Recall: %.2f%%" % (recall * 100.0))
        print("F1-score: %.2f%%" % (f1 * 100.0))
        print("Confusion Matrix:")
        print(cm)


def main():
    # load data
    X_train, X_test, y_train, y_test = load_data()

    X_train = preprocess_data(X_train)
    X_test = preprocess_data(X_test)

    # Train and evaluate the model
    train_and_evaluate_model(X_train, X_test, y_train, y_test)


if __name__ == '__main__':
    main()
