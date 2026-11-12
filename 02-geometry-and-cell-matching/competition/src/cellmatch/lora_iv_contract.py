"""Shared train/validation/deployment contract for the fixed Stage99 IV recipe."""
from .data import split_id
from .stage3 import FOLDS

TRAIN_RECIPE = dict(steps=1200, batch=1, rank=8, alpha=8, seed=2099,
                    adapter_lr=1e-4, head_lr=1e-4, weight_decay=.1,
                    eta_min=1e-6, rdrop=.1, train_patch=128)
INFERENCE_RECIPE = dict(batch_size=1, diameter=None, bsize=128,
                        flow_threshold=.4, cellprob_threshold=0., min_size=12)


def partitions(training_ids, fold, hidden_ids=()):
    """Select whole subjects; hidden IDs never become fitting records."""
    training_ids, hidden_ids = sorted(training_ids), sorted(hidden_ids)
    if len(training_ids) != len(set(training_ids)) or len(hidden_ids) != len(set(hidden_ids)):
        raise ValueError('Duplicate sample IDs')
    if set(training_ids) & set(hidden_ids):
        raise ValueError('Training and hidden IDs overlap')
    if fold == 'all-training':
        if not hidden_ids:
            raise ValueError('Final inference requires explicit hidden IDs')
        return None, training_ids, hidden_ids, 'hidden_test'
    if fold not in FOLDS:
        raise ValueError('Unknown fold')
    held = FOLDS[fold]
    fit_ids = [s for s in training_ids if split_id(s)[0] != held]
    infer_ids = [s for s in training_ids if split_id(s)[0] == held]
    if not fit_ids or not infer_ids:
        raise ValueError('Empty subject partition')
    return held, fit_ids, infer_ids, 'training'


def prediction_row(sample_id, instances):
    """Image inference output; never copy annotations or a baseline EX branch."""
    return dict(sample_id=sample_id, invivo_instances=instances,
                exvivo_instances={}, match_pairs=[])
