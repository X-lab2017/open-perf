import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import MinMaxScaler
from imblearn.under_sampling import RandomUnderSampler

def evaluate_model(model, X_test, y_test):
    """
    Evaluate a trained model on test data using various classification metrics.

    Args:
    model (object): Trained classification model.
    X_test (array-like): Test features.
    y_test (array-like): True labels for test data.

    Returns:
    accuracy (float): Model accuracy.
    precision (float): Model precision.
    recall (float): Model recall.
    f1_score (float): Model F1 score.
    """

    # Make predictions on test data
    y_pred = model.predict(X_test)

    # Compute various classification metrics
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label=1)
    recall = recall_score(y_test, y_pred, pos_label=1)
    f1 = f1_score(y_test, y_pred, pos_label=1)

    print('Confusion matrix:\n', cm)
    print('Accuracy:', accuracy)
    print('Precision:', precision)
    print('Recall:', recall)
    print('F1 score:', f1)

    return accuracy, precision, recall, f1

df = pd.read_csv('data/bothawk_BoDeGHa_data.csv', index_col=0)
bot_mapping = {'Human': 0, 'Bot': 1}
df['label'] = df['label'].replace(bot_mapping)
df = df.drop('actor_id', axis=1)

print(df.head(10))
# df
labels = df['label']
features = df.drop('label', axis=1)

if df.select_dtypes(include=[np.number]).empty:
    df = df.astype(float)

if not np.isfinite(df.values).all():
    print(df[~np.isfinite(df)])
    df.replace([np.inf, -np.inf, np.nan], np.nan, inplace=True)
df.dropna()

train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.3, random_state=42)
rus = RandomUnderSampler(random_state=42)

train_features_rus, train_labels_rus = rus.fit_resample(train_features, train_labels)

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
grid_search.fit(train_features, train_labels)

print('Best parameters: ', grid_search.best_params_)

print('Best cross-validation score: {:.2f}'.format(grid_search.best_score_))
print('Test set score: {:.2f}'.format(grid_search.score(test_features, test_labels)))

accuracy, precision, recall, f1 = evaluate_model(grid_search, test_features, test_labels)
print(f'accuracy:{accuracy}, precision:{precision}, recall:{recall}, f1:{f1}')


y_pred = grid_search.predict(test_features)

pred_df = pd.concat([test_features, pd.DataFrame({'label': test_labels, 'predicted': y_pred})], axis=1)

print(pred_df.head(10))

pred_df.to_csv('result/BoDeGHa_predictions.csv', index=False)
