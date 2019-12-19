import mlflow
import tempfile
import os
from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, cross_val_score, cross_validate
from sklearn.metrics import get_scorer


def get_random_string():
    import uuid
    return str(uuid.uuid4())


def save_model(clf):
    """ hack to save the trained model and save the model -path"""
    with tempfile.NamedTemporaryFile(delete=False) as f:
        pkl.dump(clf, f)
        f.close()
        mlflow.log_artifact(f.name)
        os.unlink(f.name)
        model_name = f.name

    model_path = list(Path(mlflow.get_artifact_uri()[8:]).iterdir())[0]
    mlflow.set_tag('model_path', model_path)


def save_dataframe(df):
    # save a pandas dataframe
    f_name = get_random_string() + '.csv'
    df.to_csv(f_name, index=False)
    mlflow.log_artifact(f_name)
    os.remove(f_name)


def run_experiment(experiment_name, clf, params, X_train, y_train, scoring,
                   eval_dataset, feature_cols=None):
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
        feature_cols = feature_cols or X_train.columns.tolist()
        cv = StratifiedShuffleSplit(n_splits=3, test_size=0.3, random_state=1024)

        clf.set_params(**params)

        # cross validation
        cv_result = cross_validate(clf, X_train[feature_cols], y_train,
                                   scoring=scoring, cv=cv,
                                   return_train_score=True)
        mlflow.log_metrics(pd.DataFrame(cv_result).mean(axis=0).to_dict())

        # save the trained model
        clf.fit(X_train[feature_cols], y_train)
        save_model(clf)

        # save feature importance
        summary = pd.DataFrame({'feature': feature_cols,
                                'importance': clf.feature_importances_})
        summary = summary[summary.importance > 0]
        summary = summary.sort_values('importance', ascending=False)
        mlflow.log_metric('n_feature', len(summary))
        save_dataframe(df)

        # validate model performance
        # `eval_dataset` is a dictionary of `dataset name` => (X, y, [a list of scorer])
        for tag, (X, y, score_fns) in eval_dataset.items():
            if not isinstance(score_fns, (list, tuple)):
                score_fns = [score_fns]

            for score_fn in score_fns:
                if isinstance(score_fn, str):
                    score_fn = get_scorer(score_fn)

                metric_name = score_fn._score_func.__name__
                metric_value = score_fn(clf, X[feature_cols], y)
                mlflow.log_metric(tag + '_' + metric_name, metric_value)
        
        return summary