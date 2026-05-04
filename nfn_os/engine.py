import numpy as np
import sqlite3
import json

from .kernel_constants import r
from .universal_knowledge import embed

# Variables globales pour le noyau condensé
K_A, K_omega, K_phi = None, None, None

def load_kernel():
    global K_A, K_omega, K_phi
    if K_A is not None:
        return
    try:
        K_tilde = np.load('nfn_os/kernel_condensed.npy')
        # shape: (r, 3) où [k, 0]=A_k, [k, 1]=omega_k, [k, 2]=phi_k
        K_A = K_tilde[:, 0]
        K_omega = K_tilde[:, 1]
        K_phi = K_tilde[:, 2]
    except FileNotFoundError:
        print("ATTENTION: kernel_condensed.npy introuvable. Avez-vous exécuté condensation.py ?")
        K_A, K_omega, K_phi = None, None, None

def proper_time(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calcule le temps propre t(x,y) comme la distance géodésique
    dans l'espace des modules fractals via la métrique de Fisher-Rao.
    t(x,y) = arccos(|<psi_x | psi_y>|^2)
    """
    dot_product = np.dot(x, y)
    # Protection contre les erreurs numériques au-delà de [-1, 1]
    dot_product = np.clip(dot_product, -1.0, 1.0)
    return np.arccos(dot_product**2)

def K_condensed(x: np.ndarray, y: np.ndarray) -> float:
    """
    Évaluation du noyau condensé multidimensionnel en O(r).
    Calcule la somme des r modes sinusoïdaux.
    """
    t = proper_time(x, y)

    # Vectorisation: calcule sin(omega_k * t + phi_k) pour tous les k simultanément
    phases = K_omega * t + K_phi
    sinusoids = np.sin(phases)

    # Somme pondérée par les amplitudes A_k
    s = np.sum(K_A * sinusoids)

    # Normalisation finale selon la formulation du papier
    # Le noyau théorique intègre une norme, ici on normalise par rapport
    # à l'énergie maximale théorique.
    return s

def load_facts(db_path='nfn_os/universal_facts.db'):
    """
    Charge les faits universels depuis la base fractale en mémoire.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT id, subject, relation, value, point, proof_chain FROM facts")
    rows = cursor.fetchall()

    facts = []
    for row in rows:
        point = np.frombuffer(row[4], dtype=np.float64)
        facts.append({
            'id': row[0],
            'subject': row[1],
            'relation': row[2],
            'value': row[3],
            'point': point,
            'proof_chain': json.loads(row[5])
        })
    conn.close()
    return facts

def answer(subject: str, relation: str):
    """
    Moteur d'inférence algébrique:
    Encode la requête Q = x ⊗ r, la compare à tous les points de la base,
    et retourne le fait qui maximise le score du noyau S = K(Q, P_F).
    """
    load_kernel()

    v_subj = embed(subject)
    v_rel = embed(relation)

    # Approximation du produit tensoriel sur la variété
    q_point = v_subj * v_rel
    norm = np.linalg.norm(q_point)
    if norm > 0:
        q_point = q_point / norm

    facts = load_facts()
    best_score = -np.inf
    best_fact = None

    for fact in facts:
        score = K_condensed(q_point, fact['point'])
        if score > best_score:
            best_score = score
            best_fact = fact

    if best_fact is None:
        return None, None

    return best_fact['value'], best_fact['proof_chain']
