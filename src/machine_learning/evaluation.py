import numpy as np
import pandas as pd
from IPython.core.display import display_markdown, display
from sklearn.metrics import confusion_matrix, classification_report, matthews_corrcoef, accuracy_score, roc_curve, \
    recall_score, f1_score, auc
from sklearn.preprocessing import binarize

from machine_learning.plot_helpers import plot_confusion_matrix, pandas_settings, plot_roc_curve


# if you want to see whole review, change max_coldwith setting to -1
@pandas_settings(display__max_colwidth=130, display__float_format='{:.6f}'.format)
def evaluate_and_report(model, X_test, y_test, show_top_n=0, preprocessed=False, plot=True):
    predict_fn = model.predict_proba if not preprocessed else model.model.predict_proba
    X_test = X_test[:len(X_test) - len(X_test) % 50]
    y_test = y_test[:len(y_test) - len(y_test) % 50]
    y_pred_proba = np.concatenate([predict_fn(x)[:, 1] for x in np.split(X_test, 50)])
    y_pred = binarize(y_pred_proba.reshape(len(y_pred_proba), 1), 0.5)

    if show_top_n > 0:
        dt = pd.DataFrame({
            'text': pd.Series(X_test),
            'prob': pd.Series(y_pred_proba),
            'pred': pd.Series(y_pred.reshape(len(y_pred))),
            'real': pd.Series(y_test.reshape(len(y_test)))
        })

        display_markdown("#### Highest {}".format(show_top_n), raw=True)
        display(dt.nlargest(show_top_n, 'prob'))

        display_markdown("#### Lowest {}".format(show_top_n), raw=True)
        display(dt.nsmallest(show_top_n, 'prob'))

        display_markdown("#### Highest {} mispredicted".format(show_top_n), raw=True)
        display(dt[dt.pred != dt.real].nlargest(show_top_n, 'prob'))

        display_markdown("#### Lowest {} mispredicted".format(show_top_n), raw=True)
        display(dt[dt.pred != dt.real].nsmallest(show_top_n, 'prob'))

    if plot:
        display_markdown("#### Classification report for {}".format(model.NAME), raw=True)
        classes = ['Negative', 'Positive']
        print(classification_report(y_test, y_pred, target_names=classes, digits=4))
        plot_confusion_matrix(confusion_matrix(y_test, y_pred), classes)

    fpr, tpr, threshold = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    if plot:
        plot_roc_curve(fpr, tpr, roc_auc)

    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'MCC': matthews_corrcoef(y_test, y_pred),
        'ROC AUC': roc_auc,
    }
    roc = {
        'fpr': fpr.tolist(),
        'tpr': tpr.tolist()
    }
    return metrics, roc
