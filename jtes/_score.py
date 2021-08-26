from itertools import chain
import numpy as np


def jaccard_timespan_event_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Jaccard-Timespan-Event-Score (JTES) is a timespan score based on the Jaccard index
    also known as Intersection over Union (IoU).

    This index is designed to score events that are defined by a timespan defined by two timestamps (t0, t1).
    Score Goals:
      * Score is in range [0-1], where 0 is lowest and 1 best.
      * False-Positives and False-Negatives are equally bad
      * If a true event spans over multiple predicted events, the result is averaged
      * If the predicted event spans over multiple true events, the result is accounted accordingly
      * If len(y_true) == 0 and len(y_pred) > 0, score = 0
      * If len(y_true) > 0 and len(y_pred) == 0, score = 0
      * If ordered(y_true) == ordered(y_pred), score = 1
      * Overlap in y_true is not allowed

    Jaccard index in general is defined as |A ∩ B| / |A ∪ B|
    For each element in y_pred a score for each y_true is calculated.

    sum(avg(non_zero_score_per_y_true))/(len(y_true) + len(False-Positive))

    returns sum-of-max-score / max(len(y_pred), len(y_true))
    Always divide by longer list to assure punishment of false-negatives

    :param y_true: np.ndarray([[t0, t1], [t0,t1], ...])
    :param y_pred: np.ndarray([[t0, t1], [t0,t1], ...])
    :return: jaccard_timespan_event_score as float
    """

    # Test t0 < t1
    for t0, t1 in chain(y_pred, y_true):
        if t0 > t1:
            raise ValueError(f"t0 must be before t1: {t0} > {t1}")

    if len(y_pred) == 0 and len(y_true) == 0:
        # 1 score for empty predictions
        return 1.0

    if len(y_pred) == 0 or len(y_true) == 0:
        # 0 score miss-matching empty predictions
        return 0.0

    # Test non overlapping events in y_true
    for i in range(1, len(y_true)):
        t00, t01 = y_true[i-1]
        t10, t11 = y_true[i]
        if not (t01 <= t10 or t00 >= t11):
            raise ValueError(f"Overlapping events in y_true are not allowed: {t00}-{t01} and {t10}-{t11}")

    # IoU
    scores = np.zeros((len(y_pred), len(y_true)))
    for ip, (tp0, tp1) in enumerate(y_pred):
        for it, (tt0, tt1) in enumerate(y_true):
            if tp1 <= tt0 or tp0 >= tt1:  # no intersection
                scores[ip][it] = 0
            else:  # intersection
                union = (max(tp0, tp1, tt0, tt1) - min(tp0, tp1, tt0, tt1)).astype('float64')
                intersection = (min(tp1, tt1) - max(tp0, tt0)).astype('float64')
                scores[ip][it] = intersection / union

    # Average prediction score per true event
    avg_col = np.zeros(scores.shape[1])
    for i, col in enumerate(scores.T):
        non_zero = col[col != 0]
        if non_zero.shape[0] > 0:
            avg_col[i] = np.average(col[col != 0])

    # count false positives to punish FP
    fp_count = len(np.where(~scores.any(axis=1))[0])

    return sum(avg_col) / (scores.shape[1] + fp_count)
