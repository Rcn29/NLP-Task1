import os
import porter2stemmer


class Review:
    def __init__(self, name, content):
        self.name = name
        self.content = content


def merge_dictionaries(dict_a, dict_b):
    merge = dict_a.copy()
    merge.update(dict_b)
    return merge


def create_review_list(path, stemmed):
    review_list = []
    file_names = os.listdir(path)
    stemmer = porter2stemmer.Porter2Stemmer()

    for name in file_names:
        with open(path + "/" + name, "r", encoding="utf-8") as file:
            content = []
            for line in file:
                if stemmed:
                    content.append(stemmer.stem(line))
                else:
                    content.append(line)
            review_list.append(Review(name, content))

    return review_list


def label_review_list(review_list, label):
    labels = {}

    for review in review_list:
        labels[review.name] = label

    return labels


def stratified_split(review_list, strats):
    splits = [[] for _ in range(strats)]
    counter = 0

    for review in review_list:
        splits[counter % strats].append(review)
        counter += 1

    return splits


def generate_training_set(splits, index):
    training_set = []

    for iterator in range(0, len(splits)):
        if iterator != index:
            training_set += splits[iterator]

    return training_set


def generate_test_set(splits, index):
    test_set = splits[index]
    return test_set
