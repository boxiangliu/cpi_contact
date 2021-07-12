import sys

class Preprocessor():
    def __init__(self, measure="IC50"):
        self.get_measure()


    def get_measure(self, measure):
        assert measure in ["IC50", "KIKD"]
        self.measure = measure
        sys.stderr.write(f"Creating dataset for measurement: {measure}\n")

    def get_mol_dict(self):
        