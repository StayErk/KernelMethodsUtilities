import os
from copy import copy

import numpy as np

from FastaGenerator import RandomGenerator, DatasetBuilder, FileWriter
from StringKernel import SimilarityStudies
from PowerEvaluation import PowerEvaluation
import matplotlib.pyplot as plt
import EditDistance
from sklearn.metrics import classification_report
import sys
import time


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    args = sys.argv

    if len(args) < 2:
        print("Usage:", "main.py [generate-sequences | similarity-matrix | svm-model | edit-distance | power-evaluation] [options...]", sep="\n")
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

        #increment_range = np.arange(0.01, float(args[5]), 0.1)
        #print(increment_range)
        increment_range = [0.01, 0.05, 0.1,  0.2, 0.3, 0.5]
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
        distanza_rif_individui = matrice_similarita[0]

        print(distanza_rif_individui)

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


    if args[1] == "power-evaluation":
        if len(args) < 3:
            print("usage: python main.py power-evaluation k-mer-max alpha-max path-null-model path-alternate-model [**path-alternate-model]")
            exit(1)
        ss = SimilarityStudies()
        path_null_model = args[4]
        paths_alternate_models = [arg for arg in args[5:]]
        krange = np.arange(2, int(args[2]), 1)

        null_model_similarita_k = dict()
        for i in krange:
            (seq_x, seq_y) = ss.prepare_data(path_null_model, False, 0)
            matrice_similarita = ss.calculate_similarity_matrix_mismatch_kernel(seq_x, k = i+1)
            null_model_similarita_k[i] = matrice_similarita[0]
            print("Null Model Similarità con k:", i, null_model_similarita_k[i])

        alternate_model_gamma = dict()

        for path in paths_alternate_models:
            alternate_model_similarita_k = dict()
            print("\n\n", path)
            for i in krange:
                (seq_x, seq_y) = ss.prepare_data(path, False, 0)
                matrice_similarita = ss.calculate_similarity_matrix_mismatch_kernel(seq_x, k=i+1)
                alternate_model_similarita_k[i] = matrice_similarita[0]
                print("Alternate Model Similarità con k:", i, alternate_model_similarita_k[i])

            alternate_model_gamma[path.split("/")[2].split("_")[4]] = copy(alternate_model_similarita_k)

        print(alternate_model_gamma)

        results_for_gamma = dict()
        for gamma in alternate_model_gamma.keys():
            print("\n\n\ngamma:", gamma)
            results_for_k = dict()
            for k_nm, k_am in zip(null_model_similarita_k.keys(), alternate_model_gamma[gamma].keys()):
                pw = PowerEvaluation(10, null_model_similarita_k.get(k_nm), alternate_model_gamma[gamma].get(k_am))
                pw.type1_error()
                power = pw.power()
                results_for_k[k_nm] = power
                print("\nk=", k_nm, "Null Model:", null_model_similarita_k.get(k_nm), "\nAlternate Model:", alternate_model_gamma[gamma].get(k_am),  "\npower:", power)
            results_for_gamma[gamma] = copy(results_for_k)

        print(results_for_gamma)

        for gamma in alternate_model_gamma.keys():

            couples_k_power = list(results_for_gamma[gamma].items())
            y_values = list(results_for_gamma[gamma].keys())

            print("X_values", couples_k_power)
            print("Y_values", y_values)

            x = []
            y = []
            for k, power in couples_k_power:
                x.append(power)
                y.append(k)

            plt.plot(y, x)
            plt.ylabel("Power")
            plt.xlabel("k")
            plt.show()
            plt.savefig("./Simulazioni/" + "gamma_" + gamma + "_plot.png")