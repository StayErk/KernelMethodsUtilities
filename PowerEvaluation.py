from Bio import SeqIO as sio
import math
class PowerEvaluation:

    def __init__(self, alpha: float, nm, amodel):
        self.alpha = alpha
        self.amodels = sorted(amodel, reverse = True)
        self.null_model = sorted(nm, reverse=True)

    def type1_error(self):
        alphesimo_indice = len(self.null_model) / 100 * self.alpha
        self.alphesimo = self.null_model[math.ceil(alphesimo_indice)]

    def power(self):
        true_positive = 0
        for misura in self.amodels:
            if misura >= self.alphesimo:
                true_positive = true_positive + 1

        return true_positive / len(self.amodels)


