"""
metrics from the official scoring repository
"""

from typing import Tuple, Sequence

import numpy as np
from torch_ecg.components.outputs import ClassificationOutput  # noqa: F401

from cfg import BaseCfg


__all__ = [
    "compute_metrics",
    "compute_metrics_detailed",
]


######################################
# custom metrics computation functions
######################################


def compute_metrics(
    labels: np.ndarray,
    scalar_outputs: np.ndarray,
    binary_outputs: np.ndarray,
    classes: Sequence[str] = BaseCfg.classes,
) -> Tuple[float, float, float, float, float]:
    """
    Compute detailed metrics, modified from the function `evaluate_model`
    in `evaluate_model.py` in the official scoring repository.

    Parameters
    ----------
    labels: np.ndarray,
        binary labels, of shape: (n_samples, n_classes)
    scalar_outputs: np.ndarray,
        scalar outputs (probabilities), of shape: (n_samples, n_classes)
    binary_outputs: np.ndarray,
        binary outputs, of shape: (n_samples, n_classes)
    classes: sequence of str, default `BaseCfg.classes`,
        class names

    Returns
    -------
    auroc: float,
        the macro-averaged area under the receiver operating characteristic curve
    auprc: float,
        the macro-averaged area under the precision-recall curve
    accuracy: float,
        the accuracy
    f_measure: float,
        the macro-averaged F-measure
    challenge_score: float,
        the challenge score

    """
    (
        auroc,
        auprc,
        _,
        _,
        accuracy,
        f_measure,
        _,
        challenge_score,
    ) = compute_metrics_detailed(labels, scalar_outputs, binary_outputs, classes)
    return auroc, auprc, accuracy, f_measure, challenge_score


def compute_metrics_detailed(
    labels: np.ndarray,
    scalar_outputs: np.ndarray,
    binary_outputs: np.ndarray,
    classes: Sequence[str] = BaseCfg.classes,
) -> Tuple[float, float, np.ndarray, np.ndarray, float, float, np.ndarray, float]:
    """
    Compute detailed metrics, modified from the function `evaluate_model`
    in `evaluate_model.py` in the official scoring repository.

    Parameters
    ----------
    labels: np.ndarray,
        binary labels, of shape: (n_samples, n_classes)
    scalar_outputs: np.ndarray,
        scalar outputs (probabilities), of shape: (n_samples, n_classes)
    binary_outputs: np.ndarray,
        binary outputs, of shape: (n_samples, n_classes)
    classes: sequence of str, default `BaseCfg.classes`,
        class names

    Returns
    -------
    auroc: float,
        the macro-averaged area under the receiver operating characteristic curve
    auprc: float,
        the macro-averaged area under the precision-recall curve
    auroc_classes: np.ndarray,
        the area under the receiver operating characteristic curve for each class
    auprc_classes: np.ndarray,
        the area under the precision-recall curve for each class
    accuracy: float,
        the accuracy
    f_measure: float,
        the macro-averaged F-measure
    f_measure_classes: np.ndarray,
        the F-measure for each class
    challenge_score: float,
        the challenge score

    """
    # For each patient, set the 'Unknown' class to positive if no class is positive or if multiple classes are positive.
    labels = enforce_positives(labels, classes, "Unknown")
    binary_outputs = enforce_positives(binary_outputs, classes, "Unknown")

    # Evaluate the model by comparing the labels and outputs.
    auroc, auprc, auroc_classes, auprc_classes = compute_auc(labels, scalar_outputs)
    accuracy = compute_accuracy(labels, binary_outputs)
    f_measure, f_measure_classes = compute_f_measure(labels, binary_outputs)
    challenge_score = compute_challenge_score(labels, binary_outputs, classes)

    return (
        auroc,
        auprc,
        auroc_classes,
        auprc_classes,
        accuracy,
        f_measure,
        f_measure_classes,
        challenge_score,
    )


###########################################
# methods from the file evaluation_model.py
# of the official repository
###########################################


def enforce_positives(
    outputs: np.ndarray, classes: Sequence[str], positive_class: str
) -> np.ndarray:
    """
    For each patient, set a specific class to positive if no class is positive or multiple classes are positive.

    Parameters
    ----------
    outputs: np.ndarray,
        binary outputs, of shape: (n_samples, n_classes)
    classes: Sequence[str],
        class names
    positive_class: str,
        class name to be set to positive

    Returns
    -------
    outputs: np.ndarray,
        enforced binary outputs, of shape: (n_samples, n_classes)

    """
    num_patients, num_classes = np.shape(outputs)
    j = classes.index(positive_class)

    for i in range(num_patients):
        if np.sum(outputs[i, :]) != 1:
            outputs[i, :] = 0
            outputs[i, j] = 1
    return outputs


def compute_confusion_matrix(labels: np.ndarray, outputs: np.ndarray) -> np.ndarray:
    """
    Compute a binary confusion matrix, where the columns are expert labels and rows are classifier labels.

    Parameters
    ----------
    labels: np.ndarray,
        binary labels, of shape: (n_samples, n_classes)
    outputs: np.ndarray,
        binary outputs, of shape: (n_samples, n_classes)

    Returns
    -------
    A: np.ndarray,
        confusion matrix, of shape: (n_classes, n_classes)

    """
    assert np.shape(labels) == np.shape(outputs)
    assert all(value in (0, 1) for value in np.unique(labels))
    assert all(value in (0, 1) for value in np.unique(outputs))

    num_patients, num_classes = np.shape(labels)

    A = np.zeros((num_classes, num_classes))
    for k in range(num_patients):
        i = np.argmax(outputs[k, :])
        j = np.argmax(labels[k, :])
        A[i, j] += 1

    return A


def compute_one_vs_rest_confusion_matrix(
    labels: np.ndarray, outputs: np.ndarray
) -> np.ndarray:
    """
    Compute binary one-vs-rest confusion matrices, where the columns are expert labels and rows are classifier labels.

    Parameters
    ----------
    labels: np.ndarray,
        binary labels, of shape: (n_samples, n_classes)
    outputs: np.ndarray,
        binary outputs, of shape: (n_samples, n_classes)

    Returns
    -------
    A: np.ndarray,
        one-vs-rest confusion matrix, of shape: (n_classes, 2, 2)

    """
    assert np.shape(labels) == np.shape(outputs)
    assert all(value in (0, 1) for value in np.unique(labels))
    assert all(value in (0, 1) for value in np.unique(outputs))

    num_patients, num_classes = np.shape(labels)

    A = np.zeros((num_classes, 2, 2))
    for i in range(num_patients):
        for j in range(num_classes):
            if labels[i, j] == 1 and outputs[i, j] == 1:  # TP
                A[j, 0, 0] += 1
            elif labels[i, j] == 0 and outputs[i, j] == 1:  # FP
                A[j, 0, 1] += 1
            elif labels[i, j] == 1 and outputs[i, j] == 0:  # FN
                A[j, 1, 0] += 1
            elif labels[i, j] == 0 and outputs[i, j] == 0:  # TN
                A[j, 1, 1] += 1

    return A


# alias
compute_ovr_confusion_matrix = compute_one_vs_rest_confusion_matrix


def compute_auc(
    labels: np.ndarray, outputs: np.ndarray
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Compute macro AUROC and macro AUPRC, and AUPRCs, AUPRCs for each class.

    Parameters
    ----------
    labels: np.ndarray,
        binary labels, of shape: (n_samples, n_classes)
    outputs: np.ndarray,
        binary outputs, of shape: (n_samples, n_classes)

    Returns
    -------
    macro_auroc: float,
        macro AUROC
    macro_auprc: float,
        macro AUPRC
    auprc: np.ndarray,
        AUPRCs for each class, of shape: (n_classes,)
    auprc: np.ndarray,
        AUPRCs for each class, of shape: (n_classes,)

    """
    num_patients, num_classes = np.shape(labels)

    # Compute and summarize the confusion matrices for each class across at distinct output values.
    auroc = np.zeros(num_classes)
    auprc = np.zeros(num_classes)

    for k in range(num_classes):
        # We only need to compute TPs, FPs, FNs, and TNs at distinct output values.
        thresholds = np.unique(outputs[:, k])
        thresholds = np.append(thresholds, thresholds[-1] + 1)
        thresholds = thresholds[::-1]
        num_thresholds = len(thresholds)

        # Initialize the TPs, FPs, FNs, and TNs.
        tp = np.zeros(num_thresholds)
        fp = np.zeros(num_thresholds)
        fn = np.zeros(num_thresholds)
        tn = np.zeros(num_thresholds)
        fn[0] = np.sum(labels[:, k] == 1)
        tn[0] = np.sum(labels[:, k] == 0)

        # Find the indices that result in sorted output values.
        idx = np.argsort(outputs[:, k])[::-1]

        # Compute the TPs, FPs, FNs, and TNs for class k across thresholds.
        i = 0
        for j in range(1, num_thresholds):
            # Initialize TPs, FPs, FNs, and TNs using values at previous threshold.
            tp[j] = tp[j - 1]
            fp[j] = fp[j - 1]
            fn[j] = fn[j - 1]
            tn[j] = tn[j - 1]

            # Update the TPs, FPs, FNs, and TNs at i-th output value.
            while i < num_patients and outputs[idx[i], k] >= thresholds[j]:
                if labels[idx[i], k]:
                    tp[j] += 1
                    fn[j] -= 1
                else:
                    fp[j] += 1
                    tn[j] -= 1
                i += 1

        # Summarize the TPs, FPs, FNs, and TNs for class k.
        tpr = np.zeros(num_thresholds)
        tnr = np.zeros(num_thresholds)
        ppv = np.zeros(num_thresholds)
        for j in range(num_thresholds):
            if tp[j] + fn[j]:
                tpr[j] = float(tp[j]) / float(tp[j] + fn[j])
            else:
                tpr[j] = float("nan")
            if fp[j] + tn[j]:
                tnr[j] = float(tn[j]) / float(fp[j] + tn[j])
            else:
                tnr[j] = float("nan")
            if tp[j] + fp[j]:
                ppv[j] = float(tp[j]) / float(tp[j] + fp[j])
            else:
                ppv[j] = float("nan")

        # Compute AUROC as the area under a piecewise linear function with TPR/
        # sensitivity (x-axis) and TNR/specificity (y-axis) and AUPRC as the area
        # under a piecewise constant with TPR/recall (x-axis) and PPV/precision
        # (y-axis) for class k.
        for j in range(num_thresholds - 1):
            auroc[k] += 0.5 * (tpr[j + 1] - tpr[j]) * (tnr[j + 1] + tnr[j])
            auprc[k] += (tpr[j + 1] - tpr[j]) * ppv[j + 1]

    # Compute macro AUROC and macro AUPRC across classes.
    if np.any(np.isfinite(auroc)):
        macro_auroc = np.nanmean(auroc)
    else:
        macro_auroc = float("nan")
    if np.any(np.isfinite(auprc)):
        macro_auprc = np.nanmean(auprc)
    else:
        macro_auprc = float("nan")

    return macro_auroc, macro_auprc, auroc, auprc


def compute_accuracy(labels: np.ndarray, outputs: np.ndarray) -> float:
    """
    Compute accuracy.

    Parameters
    ----------
    labels: np.ndarray,
        binary labels, of shape: (n_samples, n_classes)
    outputs: np.ndarray,
        binary outputs, of shape: (n_samples, n_classes)

    Returns
    -------
    accuracy: float,
        the accuracy
    """
    A = compute_confusion_matrix(labels, outputs)

    if np.sum(A) > 0:
        accuracy = np.sum(np.diag(A)) / np.sum(A)
    else:
        accuracy = float("nan")

    return accuracy


def compute_f_measure(
    labels: np.ndarray, outputs: np.ndarray
) -> Tuple[float, np.ndarray]:
    """
    Compute macro F-measure, and F-measures for each class.

    Parameters
    ----------
    labels: np.ndarray,
        binary labels, of shape: (n_samples, n_classes)
    outputs: np.ndarray,
        binary outputs, of shape: (n_samples, n_classes)

    Returns
    -------
    macro_f_measure: float,
        macro F-measure
    f_measure: np.ndarray,
        F-measures for each class, of shape: (n_classes,)
    """
    num_patients, num_classes = np.shape(labels)

    A = compute_one_vs_rest_confusion_matrix(labels, outputs)

    f_measure = np.zeros(num_classes)
    for k in range(num_classes):
        tp, fp, fn, tn = A[k, 0, 0], A[k, 0, 1], A[k, 1, 0], A[k, 1, 1]
        if 2 * tp + fp + fn > 0:
            f_measure[k] = float(2 * tp) / float(2 * tp + fp + fn)
        else:
            f_measure[k] = float("nan")

    if np.any(np.isfinite(f_measure)):
        macro_f_measure = np.nanmean(f_measure)
    else:
        macro_f_measure = float("nan")

    return macro_f_measure, f_measure


def compute_challenge_score(
    labels: np.ndarray, outputs: np.ndarray, classes: Sequence[str]
) -> float:
    """
    Compute Challenge score.

    Parameters
    ----------
    labels: np.ndarray,
        binary labels, of shape: (n_samples, n_classes)
    outputs: np.ndarray,
        binary outputs, of shape: (n_samples, n_classes)
    classes: sequence of str,
        class names

    Returns
    -------
    mean_score: float,
        mean Challenge score
    """
    # Define costs. Better to load these costs from an external file instead of defining them here.
    c_algorithm = 1  # Cost for algorithmic prescreening.
    c_gp = 250  # Cost for screening from a general practitioner (GP).
    c_specialist = 500  # Cost for screening from a specialist.
    c_treatment = 1000  # Cost for treatment.
    c_error = 10000  # Cost for diagnostic error.
    alpha = 0.5  # Fraction of murmur unknown cases that are positive.

    num_patients, num_classes = np.shape(labels)

    A = compute_confusion_matrix(labels, outputs)

    idx_positive = classes.index("Present")
    idx_unknown = classes.index("Unknown")
    idx_negative = classes.index("Absent")

    n_pp = A[idx_positive, idx_positive]
    n_pu = A[idx_positive, idx_unknown]
    n_pn = A[idx_positive, idx_negative]
    n_up = A[idx_unknown, idx_positive]
    n_uu = A[idx_unknown, idx_unknown]
    n_un = A[idx_unknown, idx_negative]
    n_np = A[idx_negative, idx_positive]
    n_nu = A[idx_negative, idx_unknown]
    n_nn = A[idx_negative, idx_negative]

    n_total = n_pp + n_pu + n_pn + n_up + n_uu + n_un + n_np + n_nu + n_nn

    total_score = (
        c_algorithm * n_total
        + c_gp * (n_pp + n_pu + n_pn)
        + c_specialist * (n_pu + n_up + n_uu + n_un)
        + c_treatment * (n_pp + alpha * n_pu + n_up + alpha * n_uu)
        + c_error * (n_np + alpha * n_nu)
    )
    if n_total > 0:
        mean_score = total_score / n_total
    else:
        mean_score = float("nan")

    return mean_score
