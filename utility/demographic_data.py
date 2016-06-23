age_bin_correspondence = {  # age bin 2 age interval
    0: (0, 19),
    1: (20, 29),
    2: (30, 39),
    3: (40, 49),
    4: (50, float("inf")),
    5: 'unknown'
}

gend_bin_correspondence = {  # gender bin to gender
    0: 'male',
    1: 'female',
    2: 'unknown'
}

age_gend_abs_bin = {  # age bin, gender bin to absolute bin
    (0, 0): 0,
    (1, 0): 1,
    (2, 0): 2,
    (3, 0): 3,
    (4, 0): 4,
    (0, 1): 5,
    (1, 1): 6,
    (2, 1): 7,
    (3, 1): 8,
    (4, 1): 9,
    # 10 is unknown
}

old_age_bins = {
    '18-29': 0,
    '30-39': 2,
    '40-49': 3,
    '20-29': 1,
    '50-59': 4,
    '': 5,
    '60-69': 4,
    'None': 5,
    '0-19': 0,
    '70+': 4}

new_age_bins = {(0, 19): 0,
            (20, 29): 1,
            (30, 39): 2,
            (40, 49): 3,
            (50, 59): 4,
            (60, 69): 4,
            (70, 100): 4,
            (-7977, -7977): 1}


def new_age_to_bin(nage):
    """converts a new age to a bin"""
    if not nage:
        return 6
    else:
        k = int(nage)
        for a, b in new_age_bins:
            if a <= k and k <= b:
                return new_age_bins[(a, b)]


def old_age_to_bin(oage):
    """converts an old age to a bin"""
    return old_age_bins[oage]


def gender_to_bin(gend):
    """ works for both """
    if gend == 'male':
        return 0
    elif gend == 'female':
        return 1
    else:
        return 2


def age_gend_bin_to_comb(age_bin, gend_bin):
    """ combines an age bin and a gender bin """
    return age_gend_abs_bin.get((age_bin, gend_bin), 10)