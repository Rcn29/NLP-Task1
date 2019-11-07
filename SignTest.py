import math


def n_choose_r(n, r):
    f = math.factorial
    return f(n) // f(r) // f(n - r)


def compute_p_value(original_labels, model1_predictions, model2_predictions):
    ties = 0
    wins_2 = 0
    wins_1 = 0
    p_value = 0

    for name in model1_predictions.keys():
        if model1_predictions[name] == model2_predictions[name]:
            ties += 1
        elif model1_predictions[name] == original_labels[name]:
            wins_1 += 1
        elif model2_predictions[name] == original_labels[name]:
            wins_2 += 1

    n = 2 * math.ceil(ties / 2) + wins_1 + wins_2
    k = math.ceil(ties / 2) + min(wins_1, wins_2)

    for it in range(0, k + 1):
        p_value += n_choose_r(n, it)

    p_value = p_value / pow(2, n - 1)

    return p_value
