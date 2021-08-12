from numpy import random

import numpy as np
from strkernel.mismatch_kernel import MismatchKernel
from strkernel.mismatch_kernel import preprocess
from time import process_time;
from Bio import SeqIO as sio
from Bio.Seq import Seq
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.metrics import classification_report  # classfication summary
import matplotlib.pyplot as plt

import shutil
import os

class SimilarityStudies():

    def calculate_similarity_matrix_mismatch_kernel(self, sequenze, l = 9, k = 2, m = 1) -> np.ndarray:
        """
        Per ora implementa il mismatch kernel
        :param sequenze:
        :return: matrice di similarità
        """
        return MismatchKernel(l, k, m).get_kernel(sequenze).kernel

    def prepare_data(self, path_sequenze: str, casual_extraction: bool, number_of_extractions: int, zeros: bool = False):
        class_a_seqs = [seq.seq for seq in sio.parse(path_sequenze, 'fasta')]
        print(class_a_seqs)
        if casual_extraction:
            a_x = preprocess(random.choice(class_a_seqs, number_of_extractions))
        else:
            a_x = preprocess(class_a_seqs)

        if zeros:
            a_y = np.zeros(len(a_x))
        else:
            a_y = np.ones(len(a_x))

        return a_x, a_y

    def calculate_mismatch_kernel(self, path_class_a: str, path_class_b: str, casual_extraction: bool = False, number_of_extraction: int = 500, l=4, k=9, m=1):
        (a_x, a_y) = self.prepare_data(path_class_a, casual_extraction, number_of_extraction, False)
        (b_x, b_y) = self.prepare_data(path_class_b, casual_extraction, number_of_extraction, True)
        class_a_similarity_matrix = self.calculate_similarity_matrix_mismatch_kernel(a_x, l, k, m)
        class_b_similarity_matrix = self.calculate_similarity_matrix_mismatch_kernel(b_x, l, k, m)

        return class_a_similarity_matrix, class_b_similarity_matrix, a_y, b_y

    def train_svm(self, class_a_similarity_matrix, class_b_similarity_matrix, class_a_y, class_b_y):
        X = np.concatenate([class_a_similarity_matrix, class_b_similarity_matrix])
        Y = np.concatenate([class_a_y, class_b_y])

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
        classificatore = SVC()
        classificatore.fit(X_train, Y_train)

        Y_true, Y_pred = Y_test, classificatore.predict(X_test)
        Y_score = classificatore.decision_function(X_test)
        return Y_true, Y_pred, Y_score, Y_test, classificatore

    def __calculate_fpr_tpr_treshold(self, Y_test, Y_score):
        fpr, tpr, tresholds = roc_curve(Y_test, Y_score)
        roc_auc = auc(fpr, tpr)
        return fpr, tpr, tresholds, roc_auc

    def plot_true_positive(self, Y_test, Y_score, path: str):
        fpr, tpr, treshold, roc_auc = self.__calculate_fpr_tpr_treshold(Y_test, Y_score)
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating curve')
        plt.legend(loc="lower right")
        plt.savefig(path)
        plt.show()



