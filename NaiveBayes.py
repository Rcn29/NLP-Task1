import math


def generate_unigram_probs(training_set, original_labels, smoothed):
    probs = {}
    word_set = set()
    pos_count = {}
    neg_count = {}
    pos_total = 0
    neg_total = 0

    for review in training_set:
        for word in review.content:
            word_set.add(word)

    for word in word_set:
        pos_count[word] = 0
        neg_count[word] = 0

    for review in training_set:
        for word in review.content:
            if original_labels[review.name] == "positive":
                pos_count[word] += 1
            else:
                neg_count[word] += 1

    for word in word_set:
        pos_total += pos_count[word]
        neg_total += neg_count[word]

    if smoothed:

        for word in word_set:
            pos_prob = (pos_count[word] + 1) / (pos_total + len(word_set))
            neg_prob = (neg_count[word] + 1) / (neg_total + len(word_set))
            probs[word] = (pos_prob, neg_prob)

    else:

        for word in word_set:
            pos_prob = pos_count[word] / pos_total
            neg_prob = neg_count[word] / neg_total
            probs[word] = (pos_prob, neg_prob)

    return probs


def generate_bigram_probs(training_set, original_labels, smoothed):
    probs = {}
    bigram_set = set()
    pos_count = {}
    neg_count = {}
    pos_total = 0
    neg_total = 0

    for review in training_set:
        for word_1, word_2 in zip(review.content[:-1], review.content[1:]):
            bigram_set.add((word_1, word_2))

    for bigram in bigram_set:
        pos_count[bigram] = 0
        neg_count[bigram] = 0

    for review in training_set:
        for word_1, word_2 in zip(review.content[:-1], review.content[1:]):
            if original_labels[review.name] == "positive":
                pos_count[(word_1, word_2)] += 1
            else:
                neg_count[(word_1, word_2)] += 1

    for bigram in bigram_set:
        pos_total += pos_count[bigram]
        neg_total += neg_count[bigram]

    if smoothed:

        for bigram in bigram_set:
            pos_prob = (pos_count[bigram] + 1) / (pos_total + len(bigram_set))
            neg_prob = (neg_count[bigram] + 1) / (neg_total + len(bigram_set))
            probs[bigram] = (pos_prob, neg_prob)

    else:

        for bigram in bigram_set:
            pos_prob = pos_count[bigram] / pos_total
            neg_prob = neg_count[bigram] / neg_total
            probs[bigram] = (pos_prob, neg_prob)

    return probs


def generate_both_probs(training_set, original_labels, smoothed):
    probs = {}
    word_set = set()
    pos_count = {}
    neg_count = {}
    pos_total = 0
    neg_total = 0

    for review in training_set:
        for word in review.content:
            word_set.add(word)

    for word in word_set:
        pos_count[word] = 0
        neg_count[word] = 0

    for review in training_set:
        for word in review.content:
            if original_labels[review.name] == "positive":
                pos_count[word] += 1
            else:
                neg_count[word] += 1

    for word in word_set:
        pos_total += pos_count[word]
        neg_total += neg_count[word]

    if smoothed:

        for word in word_set:
            pos_prob = (pos_count[word] + 1) / (pos_total + len(word_set))
            neg_prob = (neg_count[word] + 1) / (neg_total + len(word_set))
            probs[word] = (pos_prob, neg_prob)

    else:

        for word in word_set:
            pos_prob = pos_count[word] / pos_total
            neg_prob = neg_count[word] / neg_total
            probs[word] = (pos_prob, neg_prob)

    bigram_set = set()
    pos_count = {}
    neg_count = {}
    pos_total = 0
    neg_total = 0

    for review in training_set:
        for word_1, word_2 in zip(review.content[:-1], review.content[1:]):
            bigram_set.add((word_1, word_2))

    for bigram in bigram_set:
        pos_count[bigram] = 0
        neg_count[bigram] = 0

    for review in training_set:
        for word_1, word_2 in zip(review.content[:-1], review.content[1:]):
            if original_labels[review.name] == "positive":
                pos_count[(word_1, word_2)] += 1
            else:
                neg_count[(word_1, word_2)] += 1

    for bigram in bigram_set:
        pos_total += pos_count[bigram]
        neg_total += neg_count[bigram]

    if smoothed:
        for bigram in bigram_set:
            pos_prob = (pos_count[bigram] + 1) / (pos_total + len(bigram_set))
            neg_prob = (neg_count[bigram] + 1) / (neg_total + len(bigram_set))
            probs[bigram] = (pos_prob, neg_prob)

    else:

        for bigram in bigram_set:
            pos_prob = pos_count[bigram] / pos_total
            neg_prob = neg_count[bigram] / neg_total
            probs[bigram] = (pos_prob, neg_prob)

    return probs


def naivebayes(probabilities, feature_vectors, test_set):
    predicted_labels = {}

    for review in test_set:
        pos_sum = 0
        neg_sum = 0

        for feature in feature_vectors[review.name].keys():
            if feature in probabilities.keys():
                if probabilities[feature][0] != 0:
                    pos_sum += math.log(probabilities[feature][0], 2) * \
                        feature_vectors[review.name][feature]
                if probabilities[feature][1] != 0:
                    neg_sum += math.log(probabilities[feature][1], 2) * \
                        feature_vectors[review.name][feature]

        if pos_sum > neg_sum:
            predicted_labels[review.name] = "positive"
        else:
            predicted_labels[review.name] = "negative"

    return predicted_labels
