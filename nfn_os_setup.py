import os
import sys

def setup():
    if not os.path.exists("nfn_os/kernel_condensed.npy"):
        print("Initialisation du noyau fractal (condensation)...")
        from nfn_os.condensation import generate_condensed_kernel
        generate_condensed_kernel()

    if not os.path.exists("nfn_os/universal_facts.db"):
        print("Création de l'espace de Hilbert des faits (base universelle)...")
        from nfn_os.universal_knowledge import generate_gauge_matrices, generate_universal_facts
        generate_gauge_matrices()
        generate_universal_facts()

if __name__ == "__main__":
    setup()
