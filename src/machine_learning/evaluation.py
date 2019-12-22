import pandas as pd
from IPython.core.display import display_markdown, display

from machine_learning.plot_helpers import plot_confusion_matrix, pandas_settings, plot_roc_curve 
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import binarize


# if you want to see whole review, change max_coldwith setting to -1
@pandas_settings(display__max_colwidth=130, display__float_format='{:.6f}'.format)
def evaluate_and_report(model, X_test, y_test, show_top_n=5):
    y_pred_proba = model.predict_proba(X_test)[:, 1]
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

    display_markdown("#### Classification report for {}".format(model.NAME), raw=True)
    classes = ['Negative', 'Positive']
    print(classification_report(y_test, y_pred, target_names=classes))
    plot_confusion_matrix(confusion_matrix(y_test, y_pred), classes)
    plot_roc_curve(y_test, y_pred_proba)
    score = accuracy_score(y_test, y_pred)
    return score, y_pred_proba
