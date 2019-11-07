from timeit import default_timer
from Evaluation import cross_validation
from SignTest import compute_p_value
from InputProcessing import *

path_pos_reviews = os.path.join(os.getcwd(), 'POS')
path_neg_reviews = os.path.join(os.getcwd(), 'NEG')
no_of_splits = 10

gram = ["Unigrams", "Bigrams", "Both"]
mode = ["Frequency", "Presence"]
model = ["Naive Bayes", "SVM"]
stemmed = [False, True]
smoothed = [False, True]

nb_labels = []
svm_labels = []

nb_results_file = open("NB-Results", "a")
nb_results_file.truncate(0)

svm_results_file = open("SVM-Results", "a")
svm_results_file.truncate(0)

sign_test_results_file = open("SignTest-Results", "a")
sign_test_results_file.truncate(0)

for mymode in mode:
    for mygram in gram:
        t0 = default_timer()
        nb_labels.append(cross_validation(model[0], smoothed[1], mygram, mymode, path_pos_reviews,
                                          path_neg_reviews, stemmed[1], no_of_splits, nb_results_file))
        t1 = default_timer()
        print(t1 - t0)

for mymode in mode:
    for mygram in gram:
        t0 = default_timer()
        svm_labels.append(cross_validation(model[1], smoothed[1], mygram, mymode, path_pos_reviews,
                                           path_neg_reviews, stemmed[1], no_of_splits, svm_results_file))
        t1 = default_timer()
        print(t1 - t0)

all_system_labels = {}
pos_reviews = create_review_list(path_pos_reviews, stemmed[1])
neg_reviews = create_review_list(path_neg_reviews, stemmed[1])
original_labels = merge_dictionaries(label_review_list(pos_reviews, "positive"),
                                     label_review_list(neg_reviews, "negative"))


print("p(SVM_frequency_unigrams, SVM_presence_unigrams) = " +
      str(round(compute_p_value(original_labels,
                                svm_labels[0], svm_labels[3]), 3)) + "\n",
      file=sign_test_results_file)


print("p(SVM_frequency_bigrams, SVM_presence_bigrams) = " +
      str(round(compute_p_value(original_labels,
                                svm_labels[1], svm_labels[4]), 3)) + "\n",
      file=sign_test_results_file)


print("p(SVM_frequency_both, SVM_presence_both) = " +
      str(round(compute_p_value(original_labels,
                                svm_labels[2], svm_labels[5]), 3)) + "\n",
      file=sign_test_results_file)


print("p(NB_presence_unigrams, SVM_presence_unigrams) = " +
      str(round(compute_p_value(original_labels,
                                nb_labels[3], svm_labels[3]), 3)) + "\n",
      file=sign_test_results_file)


print("p(NB_presence_bigrams, SVM_presence_bigrams) = " +
      str(round(compute_p_value(original_labels,
                                nb_labels[4], svm_labels[4]), 3)) + "\n",
      file=sign_test_results_file)


print("p(NB_presence_both, SVM_presence_both) = " +
      str(round(compute_p_value(original_labels,
                                nb_labels[5], svm_labels[5]), 3)) + "\n",
      file=sign_test_results_file)
