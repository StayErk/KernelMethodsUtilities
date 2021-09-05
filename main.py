import os

import numpy as np

from FastaGenerator import RandomGenerator, DatasetBuilder, FileWriter
from StringKernel import SimilarityStudies
import EditDistance
from sklearn.metrics import classification_report
import sys
import time


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    args = sys.argv

    if len(args) < 2:
        print("Usage:", "main.py [generate-sequences | similarity-matrix | svm-model | edit-distance] [options...]", sep="\n")
        exit()

    if args[1] == "generate-sequences" and len(args) < 6:
        print("Usage:", "main.py generate-sequences numero-sequenze lunghezza_seqeunza pattern-lenght gamma-value output-path")
        exit()

    if args[1] == "similarity-matrix" and len(args) < 4:
        print("Usage:", "main.py similarity-matrix path-file-sequenza-input k-max [casual | all] [optional: number-of-extractions]")
        exit()

    if args[1] == "svm-model" and len(args) < 5:
        print("Usage:", "main.py svm-model path-file-classe-a path-file-classe-b k-max [casual | all] [optional: number-of-extractions]")
        exit()

    if args[1] == "generate-sequences":
        # TODO rendere iterativo: stesso null model diversi alternate model con gamma diverso
        rg = RandomGenerator((0.25, 0.25, 0.25, 0.25), float(args[5]))
        db = DatasetBuilder(int(args[3]), int(args[2]), int(args[4]), rg)

        null_model_nomi, null_model_sequenze = db.build_null_model()
        fw = FileWriter(args[6], "Null-Model", null_model_sequenze, null_model_nomi, (int(args[3]), int(args[2]), int(args[4]), float(args[5])))
        fw.SaveDataset()

        increment_range = np.arange(0.1, float(args[5]), 0.1)
        print(increment_range)
        for i in increment_range:
            rg.g_value = i
            alternate_model_nomi, alternate_model_sequenze = db.build_alternate_model_pattern_transfer(null_model_sequenze[0], null_model_sequenze[1:])
            fw = FileWriter(args[6], "Alternate-Model", alternate_model_sequenze, alternate_model_nomi,
                            (int(args[3]), int(args[2]), int(args[4]), i))
            fw.SaveDataset()




    if args[1] == "similarity-matrix":
        ss = SimilarityStudies()

        if args[4] == 'casual':
            casual = True
        else:
            casual = False

        if casual:
            (seq_x, seq_y) = ss.prepare_data(args[2], casual, int(args[5]))
        else:
            (seq_x, seq_y) = ss.prepare_data(args[2], casual, 0)
        matrice_similarita = ss.calculate_similarity_matrix_mismatch_kernel(seq_x, k = int(args[3]))
        print(matrice_similarita)

    if args[1] == "svm-model":
        if args[5] == 'casual':
            casual = True
        else:
            casual = False

        ss = SimilarityStudies()
        start_time = time.process_time()
        if casual:
            ms_classe_a, ms_classe_b, a_y, b_y = ss.calculate_mismatch_kernel(args[2], args[3], casual, int(args[6]),
                                                                              k = int(args[4]))
        else:
            ms_classe_a, ms_classe_b, a_y, b_y = ss.calculate_mismatch_kernel(args[2], args[3], casual, 0,
                                                                              k=int(args[4]))
        Y_true, Y_pred, Y_score, Y_test, classificatore = ss.train_svm(ms_classe_a, ms_classe_b, a_y, b_y)

        score = classification_report(Y_true, Y_pred)
        parameters = args[2].split("_", 1)[1].split("/")[0]
        end_time = time.process_time()

        os.mkdir("./Report_"+parameters)
        cr = open("./Report_"+parameters + "/classification-report.txt", "w")

        cr.write("Info SVM: " + "\n" + str(score) + "\nTook: " + str(end_time - start_time) + "s\n\n" + "Matrice di Similarità elementi classe a: " + "\n" + str(ms_classe_a) + "\n\n" + "Matrice di Similarità elementi classe b: " + "\n" + str(ms_classe_b) + "\n\n")
        cr.close()

        ss.plot_true_positive(Y_test, Y_score, "./Report_"+parameters+"/plot.png")



    if args[1] == "edit-distance":

        if len(args) < 3:
            print("usage: python main.py edit-distance path-sequenze [path-sequenze]")
            exit(1)

        distanze = []

        for arg in args[2:]:
            distanze.append(EditDistance.preprocess_data(arg))

        print(distanze)

