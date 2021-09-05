
from Bio.Seq import  Seq
from Bio import SeqIO as sio

def min_edit_dist_levenshtein(string1, string2):
    deletion_cost, insertion_cost, substitution_cost = 1, 1, 0

    length_string1, length_string2 = len(string1) + 1, len(string2) + 1

    levenshtein_matrix = [[0 for i in range(length_string2)] for j in range(length_string1)]

    for i in range(length_string1):
        levenshtein_matrix[i][0] = i

    for j in range(length_string2):
        levenshtein_matrix[0][j] = j

    for i in range(1, length_string1):
        for j in range(1, length_string2):
            #print("Kata pertama huruf ke-" + str(i) + ": " + string1[ i -1])
            #print("Kata kedua huruf ke-" + str(j) + ": " + string1[ j -1])

            if string1[ i -1] == string2[ j -1]:
                #print("Dua karakter ini SAMA")
                substitution_cost = 0
                levenshtein_matrix[i][j] = min(
                    levenshtein_matrix[ i -1][j] + deletion_cost,
                    levenshtein_matrix[ i -1][ j -1] + substitution_cost,
                    levenshtein_matrix[i][ j -1] + insertion_cost
                )
                #print(levenshtein_matrix)

            else:
                #print("Dua karakter ini BERBEDA")
                substitution_cost = 1
                levenshtein_matrix[i][j] = min(
                    levenshtein_matrix[ i -1][j] + deletion_cost,
                    levenshtein_matrix[ i -1][ j -1] + substitution_cost,
                    levenshtein_matrix[i][ j -1] + insertion_cost
                )
                #print(levenshtein_matrix)

            #print("\n")

    return (levenshtein_matrix[-1][-1])

def preprocess_data(path_seqeunze: str) -> list:
    sequenze = [seq.seq for seq in sio.parse(path_seqeunze, "fasta")]
    distanze = []

    for sequenza in sequenze[1:]:
        distanze.append(min_edit_dist_levenshtein(sequenze[0], sequenza))
    print(distanze)
    return distanze


if __name__ == "__main__":
    preprocess_data("terzoGruppoSim/Null-Model_500_2_5_0.7/Null-Model.fa")
    preprocess_data("terzoGruppoSim/Alternate-Model_500_2_5_0.1/Alternate-Model.fa")
    preprocess_data("terzoGruppoSim/Alternate-Model_500_2_5_0.2/Alternate-Model.fa")
    preprocess_data("terzoGruppoSim/Alternate-Model_500_2_5_0.4/Alternate-Model.fa")
    preprocess_data("terzoGruppoSim/Alternate-Model_500_2_5_0.5/Alternate-Model.fa")
    preprocess_data("terzoGruppoSim/Alternate-Model_500_2_5_0.6/Alternate-Model.fa")

