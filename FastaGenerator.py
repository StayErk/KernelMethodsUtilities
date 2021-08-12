from random import seed
from random import random
import os
import sys
import time

"""
Questo modulo implementa la generazione di Null Model e Alternate Model utilizzando il Pattern Transfer
"""

alphabet = ('A', 'C', 'G', 'T')


class RandomGenerator():
    """
    Questa classe si occupa della generazione casuale dei valori
    """

    def __init__(self, distribuzione: tuple, g_value: float):
        self.boundaries = [0.0, 0.0, 0.0, 1.0]

        if g_value <= 0 or g_value > 1:
            raise ValueError('Il g_value deve essere compreso tra (0, 1]')

        self.g_value = g_value
        tot = 0
        for i in range(3):
            tot = tot + distribuzione[i]
            self.boundaries[i] = tot

    def get_next_base(self) -> int:
        """
        restituisce un con probabilità data dalla tupla distribuzione usata per
        inizializzare l'oggetto di tipo RandomGenerator un carattere di alphabet
        :return: valore che rappresenta un carattere di alphabet compreso tra 0 e 3
        """

        seed(time.time() / 1000)
        random_value = random()

        if random_value < self.boundaries[0]:
            return 0
        elif random_value < self.boundaries[1]:
            return 1
        elif random_value < self.boundaries[2]:
            return 2
        elif random_value < self.boundaries[3]:
            return 3

    def get_next_bernulli_value(self) -> bool:
        """
        Restiruisce vero o falso basandosi sul g_value usato per inizializzare
        l'oggetto di tipo RandomGenerator. Simula il lancio di una moneta con probabilita che esca
        testa del g_value%
        :return: un valore booleano
        """

        random_value = random()
        if random_value < self.g_value:
            return True
        else:
            return False


class DatasetBuilder:
    """
    Questa classe si occupa della generazione del Null Model e dell'Alternate Model
    """

    def __init__(self, lunghezza_stringa: int, numero_stringhe: int, lunghezzaPattern: int,
                 random_generator: RandomGenerator):
        self.lunghezza_stringa = lunghezza_stringa
        self.numero_stringhe = numero_stringhe
        self.lunghezza_pattern = lunghezzaPattern
        self.random_generator = random_generator

    def __pattern_transfer(self, seqRiferimento: list, sequenzaAlternativa: list, lunghezzaPattern: int) -> list:
        """
        Questa funzione implementa la logica del pattern transfer
        :param seqRiferimento: sequenza dalla quale copiare un pattern di lunghezza lunghezzaPatter
        :param sequenzaAlternativa: sequenza nella quale sostituire il pattern proveniente dalla sequenza di rifermento
        :param lunghezzaPattern: lunghezza del pattern da copiare
        :return: sequeza alternativa modificata
        """

        testina = 0
        contatore_sottosequenza = 0

        while testina < (len(sequenzaAlternativa) - lunghezzaPattern):
            bernulli_value = self.random_generator.get_next_bernulli_value()
            if (
                    bernulli_value):  # se esce testa copia lunghezzaPattern caratteri da seqRiferimento a seqAlternativa nella stessa posizione
                for k in range(lunghezzaPattern):
                    sequenzaAlternativa[testina] = seqRiferimento[testina]
                    testina = testina + 1
                contatore_sottosequenza = contatore_sottosequenza + 1

            else:
                testina = testina + 1

        return sequenzaAlternativa

    def __generate_rif_sequence(self, model_name: str) -> tuple:
        sequenza = ''
        nome_sequenza = model_name + "_riferimento_l:" + str(self.lunghezza_stringa) + "_n:" + str(self.numero_stringhe)
        for i in range(self.lunghezza_stringa):
            valore_base = self.random_generator.get_next_base()
            sequenza = sequenza + alphabet[valore_base]

        return nome_sequenza, sequenza

    def build_null_model(self) -> tuple:
        """
        Genera il null model, uan tupla contenente due liste una per i nomi delle sequenze e una per le sequenze stesse
        :return:
        """
        # genera la sequenza di rifermento (quella dalla quale verrà effettuato il pattern transfer nell' Alternate Model)
        nomi_sequenze = []
        sequenze = []

        nome_seq_rif, seq_rif = self.__generate_rif_sequence("Null-Model")
        nomi_sequenze.append(nome_seq_rif)
        sequenze.append(seq_rif)

        self.sequenza_riferimento = seq_rif

        for i in range(self.numero_stringhe):
            sequenza_generata = ''
            rg = RandomGenerator((0.25, 0.25, 0.25, 0.25), 0.25)  # occorre un rg per ogni iterazione
            for j in range(self.lunghezza_stringa):
                valore_base = rg.get_next_base()
                sequenza_generata = sequenza_generata + alphabet[valore_base]

            nome_sequenza_generata = "Null-Model_individuo" + str(i) + "_l:" + str(
                self.lunghezza_stringa) + "_n:" + str(self.numero_stringhe)
            nomi_sequenze.append(nome_sequenza_generata)
            sequenze.append(sequenza_generata)

        return nomi_sequenze, sequenze

    def build_alternate_model_pattern_transfer(self, sequenza_riferimento: str, sequenze_individui: list) -> tuple:
        """
        Genera l'alternate model utilizzando la tecnica del pattern transfer.
        A partire dalla sequenza di riferimento e dagli individui creati quando si è costruiti
        il null model restituisce l'alternate model
        """

        sequenze = []
        nomi_sequenze = []
        sequenze.append(sequenza_riferimento)
        nomi_sequenze.append("Pattern-Transfer-riferimento" + "_l:" + str(self.lunghezza_stringa) + "_n:" + str(
                    self.numero_stringhe) + "_lp" + str(self.lunghezza_pattern) + "_g:" + str(
                    self.random_generator.g_value))

        sequenza_riferimento = [char for char in sequenza_riferimento]

        for i, sequenza in enumerate(sequenze_individui):
            nuova_sequenza_individuo = ''
            sequenza_list = [char for char in sequenza]
            nuova_sequenza_individuo = nuova_sequenza_individuo.join(
                self.__pattern_transfer(sequenza_riferimento, sequenza_list, self.lunghezza_pattern))

            nomi_sequenze.append(
                "Pattern-Transfer-sequenza" + str(i) + "_l:" + str(self.lunghezza_stringa) + "_n:" + str(
                    self.numero_stringhe) + "_lp" + str(self.lunghezza_pattern) + "_g:" + str(
                    self.random_generator.g_value))
            sequenze.append(nuova_sequenza_individuo)

        return nomi_sequenze, sequenze


class FileWriter:
    def __init__(self, path: str, model_type: str, sequenze: list, nomi_sequenze: list, parameter: tuple):
        """

        :param path: path di output
        :param model_type: [Null-Model | Alternate-Model]
        :param sequenze: lista di sequenze
        :param nomi_sequenze: lista dei nomi delle sequenze
        :param parameter: tupla compsta da (lunghezza-stringa, numero-stringhe, lunghezza-pattern, gamma-value)
        """
        self.path = path
        self.model_type = model_type
        self.sequenze= sequenze
        self.nomi_sequenze = nomi_sequenze
        self.parametri = parameter

    def __save_file(self, path: str,  suffisso: str, nome_sequenza, sequenza):
        """
        :param path: path di partenza
        :param suffisso: suffisso del file, da informazioni sui parametri utilizzati
        :param nome_sequenza: nome della sequenza
        :param sequenza: sequenza
        :return:
        """
        file_seq_rif = open(path + self.model_type + "-" + suffisso + ".fa", "w")
        file_seq_rif.write(">" + nome_sequenza + "\n" + sequenza)

    def __save_in_single_file(self, nomi_sequenze: list, sequenze: list, path: str):
        single_file = open(path, "w")
        for nome_sequenza, sequenza in zip(nomi_sequenze, sequenze):
            single_file.write(">" + nome_sequenza + "\n" + sequenza + "\n")

    def SaveDataset(self):

        model_path = self.path + "/" + self.model_type + "_" + str(self.parametri[0]) + "_" + str(self.parametri[1]) + "_" + str(self.parametri[2]) + "_" + str(self.parametri[3]) + "/"
        os.mkdir(model_path)
        self.__save_file(model_path, "riferimento", self.nomi_sequenze[0], self.sequenze[0])

        os.mkdir(model_path + "/individui")

        for i, (nome_sequenza, sequenza) in enumerate(zip(self.nomi_sequenze[1:], self.sequenze[1:])):
            self.__save_file(model_path + "/individui/", "individuo" + str(i), nome_sequenza, sequenza)

        self.__save_in_single_file(self.nomi_sequenze, self.sequenze, model_path + "/" + self.model_type + ".fa")






if __name__ == "__main__":
    rg = RandomGenerator((0.25, 0.25, 0.25, 0.25), 0.5)
    db = DatasetBuilder(100, 10, 5, rg)

    null_model_nomi, null_model_sequenze = db.build_null_model()
    alternate_model_nomi, alternate_model_sequenze= db.build_alternate_model_pattern_transfer(null_model_sequenze[0], null_model_sequenze[1:])

    fw = FileWriter(".", "Null-Model", null_model_sequenze, null_model_nomi, (100, 10, 5, 0.5))
    fw.SaveDataset()

