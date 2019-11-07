def select_unigram_features(reviews, cutoff):
        word_count = {}
        unigram_vector = []
        unigram_position = {}

        for review in reviews:
            for word in review.content:
                if word not in word_count:
                    word_count[word] = 1
                else:
                    word_count[word] += 1

        for word in word_count.keys():
            if word_count[word] >= cutoff:
                unigram_vector.append(word)

        counter = 1
        for word in unigram_vector:
            unigram_position[word] = counter
            counter += 1

        return unigram_vector, unigram_position


def select_bigram_features(reviews, cutoff):
    bigram_count = {}
    bigram_vector = []
    bigram_positions = {}

    for review in reviews:
        for word_1, word_2 in zip(review.content[:-1], review.content[1:]):
            if (word_1, word_2) not in bigram_count:
                bigram_count[(word_1, word_2)] = 1
            else:
                bigram_count[(word_1, word_2)] += 1

    for bigram in bigram_count.keys():
        if bigram_count[bigram] >= cutoff:
            bigram_vector.append(bigram)

    counter = 1

    for bigram in bigram_vector:
        bigram_positions[bigram] = counter
        counter += 1

    return bigram_vector, bigram_positions


def compute_unigram_feature_vectors(reviews, unigram_features, non_zero, mode):
    unigram_feature_vectors = {}

    for review in reviews:
        feature_vector = {}

        for feature in unigram_features:
            feature_vector[feature] = 0

        for word in review.content:
            if word in feature_vector:
                feature_vector[word] += 1

        if mode == "Presence":
            for word in feature_vector:
                if feature_vector[word] > 1:
                    feature_vector[word] = 1

        if non_zero:
            unigram_feature_vectors[review.name] = {
                x: y for x, y in feature_vector.items() if y != 0}
        else:
            unigram_feature_vectors[review.name] = feature_vector

    return unigram_feature_vectors


def compute_bigram_feature_vectors(reviews, bigram_features, non_zero, mode):
    bigram_feature_vectors = {}

    for review in reviews:
        feature_vector = {}

        for feature in bigram_features:
            feature_vector[feature] = 0

        for word_1, word_2 in zip(review.content[:-1], review.content[1:]):
            if (word_1, word_2) in feature_vector:
                feature_vector[(word_1, word_2)] += 1

        if mode == "Presence":
            for bigram in feature_vector:
                if feature_vector[bigram] > 1:
                    feature_vector[bigram] = 1

        if non_zero:
            bigram_feature_vectors[review.name] = {
                x: y for x, y in feature_vector.items() if y != 0}
        else:
            bigram_feature_vectors[review.name] = feature_vector

    return bigram_feature_vectors


def process_feature_vectors_to_svm_input(feature_vectors, positions, original_labels, file_type):
    if file_type == "train":
        file = open("SVM-Train", "a")
    elif file_type == "test":
        file = open("SVM-Test", "a")
    file.truncate(0)

    for name in feature_vectors.keys():
        string_representation = ""

        if original_labels[name] == "positive":
            string_representation += "1 "
        else:
            string_representation += "-1 "

        for feature in feature_vectors[name].keys():
            string_representation += str(positions[feature]) + ":" + str(
                feature_vectors[name][feature]) + " "

        print(string_representation, file=file)

    return
