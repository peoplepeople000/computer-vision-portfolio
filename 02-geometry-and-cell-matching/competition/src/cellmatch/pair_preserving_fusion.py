"""Fixed late fusion: preserve baseline pairs and add nonconflicting expert pairs.
Only unpaired baseline contours may be removed to accommodate an accepted pair.
No ground truth, score threshold fitting, or per-subject policy enters fusion.
"""
import copy
from .preserve_pair_masks import pixels


def fuse(original, expert):
    assert original['sample_id'] == expert['sample_id']
    result = copy.deepcopy(original)
    names = ['invivo_instances', 'exvivo_instances']
    occupied = [set(), set()]
    paired_keys = [set(), set()]
    for pair in original['match_pairs']:
        for mode, key in enumerate(pair):
            value = pixels(original[names[mode]][key])
            assert not (value & occupied[mode])
            occupied[mode].update(value)
            paired_keys[mode].add(key)
    accepted = 0
    for pair in expert['match_pairs']:
        masks = [expert[name][key] for name, key in zip(names, pair)]
        regions = list(map(pixels, masks))
        if any(value & used for value, used in zip(regions, occupied)):
            continue
        new_keys = []
        for mode, (name, mask, region) in enumerate(zip(names, masks, regions)):
            for key, old in list(result[name].items()):
                if key not in paired_keys[mode] and pixels(old) & region:
                    del result[name][key]
            key = f'FUSION_{accepted:06d}'
            assert key not in result[name]
            result[name][key] = mask
            new_keys.append(key)
            paired_keys[mode].add(key)
            occupied[mode].update(region)
        result['match_pairs'].append(new_keys)
        accepted += 1
    return result, accepted
