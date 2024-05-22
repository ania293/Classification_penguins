## Function used in the jupyter notebook

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV


def model_perform(model, model_name, params, X_train, y_train, X_test, y_test):
    gridsearch = GridSearchCV(model,
                             params,
                             scoring='f1_macro',
                             cv=5,
                             verbose=0, n_jobs=-1)

    gridsearch.fit(X_train, y_train)
    print('\nBest hyperparameter:', gridsearch.best_params_,"\nBest score:", gridsearch.best_score_)
    best_model = gridsearch.best_estimator_

    metrics_dataframe = calculate_metrics(best_model, model_name, X_test, y_test)
    return metrics_dataframe

metrics_dataframe = pd.DataFrame(columns = ['Model', 'F1_score', 'Accuracy'])
models = []
models_names = []
predictions_proba_list = []

def calculate_metrics(model, name, X_checked, y_checked):
    models.append(model)
    models_names.append(name)
    global metrics_dataframe
    predictions = model.predict(X_checked)

    # Precision, Recall, F1, Accuracy
    report = classification_report(y_checked, predictions, output_dict=True)

    # Confusion matrix
    plt.figure()
    cm = confusion_matrix(y_checked, predictions)
    ax = sns.heatmap(cm, annot=True, cmap='Blues', fmt='.0f')
    ax.set_title('Confusion Matrix\n\n')
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ')
    plt.show()

    metrics_dataframe = pd.concat([metrics_dataframe, pd.DataFrame.from_records([{'Model': name, 'F1_score': report['macro avg']['f1-score'], 'Accuracy': report['accuracy']}])])

    return metrics_dataframe


