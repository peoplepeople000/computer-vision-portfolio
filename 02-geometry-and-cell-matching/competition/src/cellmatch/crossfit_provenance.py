"""Strict partition checks for out-of-sample development predictions."""
from .data import split_id


def validate_partition(outer_train_ids, outer_subject, training_ids, prediction_ids):
    train, predicted, expected = set(training_ids), set(prediction_ids), set(outer_train_ids)
    if not train or not predicted or train & predicted:
        raise ValueError('Training and prediction IDs must be nonempty and disjoint')
    if train | predicted != expected:
        raise ValueError('Crossfit sources must exactly cover the outer development partition')
    train_subjects = {split_id(s)[0] for s in train}
    predicted_subjects = {split_id(s)[0] for s in predicted}
    if len(train_subjects) != 1 or len(predicted_subjects) != 1 or train_subjects & predicted_subjects:
        raise ValueError('Crossfit separation must hold at mouse level')
    if outer_subject in train_subjects | predicted_subjects:
        raise ValueError('Outer mouse leaked into crossfit calibration')
