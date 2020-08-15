import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm_notebook as tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

def plot_history(data):
    epochs = len(data['val_loss'])
    val_accuracy = data['val_accuracy']
    val_loss = data['val_loss']
    accuracy = data['accuracy']
    loss = data['loss']
    
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14, 3.5))
    
    ax[0].plot(accuracy, label='Train Accuracy', linewidth=2, marker='x')
    ax[0].plot(val_accuracy, label='Validation Accuracy', linewidth=2, marker='o', linestyle='--')
    ax[1].plot(loss, label='Train Loss', linewidth=2, marker='x')
    ax[1].plot(val_loss, label='Validation Loss', linewidth=2, marker='o', linestyle='--')
    ax[0].grid()
    ax[1].grid()
    ax[0].legend()
    ax[1].legend()
    ax[0].set_title('Accuracy')
    ax[1].set_title('Loss')
    plt.show()
    
def plot_feature_importance(feature_names, feature_importance, number):
    fig, ax = plt.subplots(figsize=(6, int(number/3)))
    data = pd.DataFrame(np.array(list(zip(feature_names, feature_importance))), columns=['Feature', 'Importance'])
    data.Importance = data.Importance.astype(float)
    data = data.sort_values('Importance', ascending=False)
    data = data[:number]
    sns.barplot(data=data, x='Importance', y='Feature', palette='Blues', edgecolor='black', ax=ax)
    plt.show()
    
def label_encode_cols(data):
    encoders = []
    transformed_data = []
    for col in data.columns:
        encoder = LabelEncoder()
        encoded_data = pd.Series(encoder.fit_transform(data[col].astype(str)), name=col, index=data.index)
        encoders.append(encoder)
        transformed_data.append(encoded_data)
    encoders = dict(zip(mutation_cols, encoders))
    transformed_data = pd.concat(transformed_data, axis=1)
    return encoders, transformed_data




def data_imputer(data, columns=None):
    if columns is None:
        target_cols = data.loc[:, data.isna().any()].columns
    else:
        target_cols=columns
    
    X = data[[x for x in data.columns if x not in target_cols]]
    null_col_headers = X.columns[X.isna().any()]
    null_cols = X[null_col_headers]
    X = X.drop(null_col_headers, 1)
    targets = data[target_cols]
    X_numerical = X.select_dtypes(exclude=['object'])
    X_object = X.select_dtypes(include=['object'])
    if X_object.shape[0] > 0:
        X_object = pd.get_dummies(X_object)
    X = pd.concat([X_numerical, X_object], 1)
    dtc = DecisionTreeClassifier(max_depth=10, min_samples_leaf=10)
    metric_names = ['Accuracy', 'Precision', 'Recall']
    model_metrics = pd.DataFrame(columns=metric_names)
    for col in tqdm(target_cols):
        X_all, X_null, y_all = (X.loc[targets[col].notna(), :], 
                                X.loc[targets[col].isna(), :], 
                                targets[col].loc[targets[col].notna()])
        y_all_value_counts = y_all.value_counts()
        to_drop = y_all_value_counts[y_all_value_counts == 1].index
        to_drop_mask = y_all.isin(to_drop)
        X_all, y_all = X_all[~to_drop_mask], y_all[~to_drop_mask]
        X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.1)
        encoder = LabelEncoder()
        y_train = encoder.fit_transform(y_train)
        y_test = encoder.transform(y_test)
        dtc.fit(X_train, y_train)
        y_pred = dtc.predict(X_test)
        metrics = pd.DataFrame(np.array([accuracy_score(y_test, y_pred), 
                                         precision_score(y_test, y_pred, average='weighted'), 
                                         recall_score(y_test, y_pred, average='weighted')])).T
        metrics.columns = metric_names
        model_metrics = pd.concat([model_metrics, metrics])
        y_null = encoder.inverse_transform(dtc.predict(X_null))
        data.loc[data[col].isna(), col] = y_null
    model_metrics.index = target_cols
    data = pd.concat([data, null_cols], axis=1)
    return data, model_metrics.applymap(lambda x: round(x, 2))

def plot_confusion_matrix(cf_matrix):
    group_names = ["True Neg", "False Pos", "False Neg", "True Pos"]
    group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in cf_matrix.flatten() / np.sum(cf_matrix)]
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    sns.heatmap(cf_matrix, annot=labels, fmt="", cmap='Blues')
    
def plot_roc_curve(roc_curve, optimal_point=False):
    
    optimal_threshold = roc_curve[roc_curve['Distance From Optimal'] == roc_curve['Distance From Optimal'].min()]
    
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(roc_curve['FPR'], roc_curve['TPR'], linewidth=3)
    ax.plot(np.arange(0, 1, 0.001), np.arange(0, 1, 0.001), linestyle='--', linewidth=3)
    ax.grid()
    ax.set_title('Receiver Operator Characteristic Curve')
    ax.set_xlabel('FPR')
    ax.set_ylabel('TPR')
    
    if optimal_point:
        ax.scatter(optimal_threshold['FPR'], optimal_threshold['TPR'], color='C2', linewidth=3)
    
    return ax

def plot_feature_importance(feature_importance, n_features):
    feature_importance = feature_importance.iloc[:n_features, :]
    fig, ax = plt.subplots(figsize=(5, int(n_features/2)))
    sns.barplot(data=feature_importance, x='Importance', y='Feature', ax=ax)
    ax.grid()
    return ax

def get_distance_from_optimal(roc_curve):
    return ((1 - roc_curve['TPR'])**2 + (0 - roc_curve['FPR'])**2)**(1/2)
                  
def assess_model(model, X_test, y_test, features=None, feature_importance=True, binary_target=True, threshold=None):
    if threshold is not None:
        y_pred = (model.predict_proba(X_test)[:,1] >= threshold).astype(int)
    else:
        y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    metric_results = dict()
    metric_results['classification_report'] = classification_report(y_test, y_pred)
    metric_results['accuracy_score'] = accuracy_score(y_test, y_pred)
    metric_results['cf_matrix'] = confusion_matrix(y_test, y_pred)
    metric_results['roc_curve'] = roc_curve(y_test, y_prob)
    metric_results['roc_curve'] = np.concatenate([x.reshape(-1, 1) for x in metric_results['roc_curve']], axis=1) 
    metric_results['roc_curve'] = pd.DataFrame(metric_results['roc_curve'], columns=['FPR', 'TPR', 'Threshold'])
    metric_results['roc_curve']['Distance From Optimal'] = get_distance_from_optimal(metric_results['roc_curve'])
    metric_results['roc_curve_optimal_threshold'] = metric_results['roc_curve'][metric_results['roc_curve']['Distance From Optimal'] == metric_results['roc_curve']['Distance From Optimal'].min()].Threshold
    metric_results['auc_score'] = roc_auc_score(y_test, y_prob)
    if feature_importance:
        metric_results['feature_importance'] = pd.DataFrame(np.array(list(zip(features, model.feature_importances_))), columns=['Feature', 'Importance'])
        metric_results['feature_importance'].Importance = metric_results['feature_importance'].Importance.astype(float)
        metric_results['feature_importance'] = metric_results['feature_importance'].sort_values('Importance', ascending=False)
    
    return metric_results
                               
                               
