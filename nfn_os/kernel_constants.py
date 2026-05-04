import math

# Dimensions
r = 65536          # rang du condensat (nombre de modes fondamentaux)
d = 256            # dimension de l'espace de plongement fractal

# Constantes fondamentales issues du Livre Blanc
lambda_ = 1.618033988749895    # facteur d'échelle (nombre d'or)
omega_0 = 2 * math.pi * 1.420405751e9  # raie à 21 cm normalisée
c = 299792458                  # vitesse de la lumière (m/s)
h_planck = 6.62607015e-34      # constante de Planck

# Constantes déduites pour le laplacien fractal
mass_gap = 1e-3                # gap de masse de l'univers condensé
fractal_dim = 1.584962500721156  # dimension de Hausdorff typique (ex. Sierpinski)

# Méta-données de l'univers
cosmic_time_0 = 13.8e9 * 365.25 * 24 * 3600  # temps cosmique au moment de la condensation (en secondes)
Lambda_0 = 1.0  # échelle de coupure fractale initiale
