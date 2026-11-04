"""Linear probe of the linguistic-probing experiment: class-balanced logistic regression scored by balanced accuracy."""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils.class_weight import compute_class_weight


def get_sample_weights(y_train):
    """Per-sample weights that balance the classes."""
    classes = np.unique(y_train)
    class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight_dict = dict(zip(classes, class_weights))
    return np.array([class_weight_dict[y] for y in y_train])


def remove_identical_points(activations, labels):
    """Keep the first occurrence of every distinct activation row."""
    if isinstance(labels, list):
        labels = np.array(labels)
    _, base_idx = np.unique(activations, axis=0, return_index=True)
    indices = np.sort(base_idx)
    return activations[indices], labels[indices]


def fit_sklearn_no_cv(X_train, y_train, X_test, y_test):
    """Balanced test accuracy of a logistic regression (C=0.01) trained on deduplicated activations."""
    X_train, y_train = remove_identical_points(X_train, y_train)

    # All training rows identical (one class left after deduplication): return chance level.
    if len(np.unique(y_train)) < 2:
        n_classes = max(2, len(np.unique(y_test)))
        return 1.0 / n_classes

    model = LogisticRegression(max_iter=1000, C=0.01, tol=2e-4)
    model.fit(X_train, y_train, sample_weight=get_sample_weights(y_train))

    X_test, y_test = remove_identical_points(X_test, y_test)
    y_pred = model.predict(X_test)
    return balanced_accuracy_score(y_test, y_pred)
