import numpy as np
import os
import sqlite3
import json
from scipy.stats import ortho_group

from .kernel_constants import d, lambda_

def fractal_hash(concept: str) -> float:
    """
    Hachage sensible à l'échelle fractale (déterministe).
    Projette un concept string en un scalaire dans [0, 1].
    """
    h = 0
    for i, char in enumerate(concept):
        # Utilisation de la constante de proportionnalité fractale
        h += ord(char) * (lambda_ ** (- (i % 10)))

    # Ramener dans [0, 1] en prenant la partie fractionnaire
    return h - np.floor(h)

def embed(concept: str) -> np.ndarray:
    """
    Plonge un concept dans l'espace fractal R^d.
    La décroissance de l'énergie suit le facteur d'échelle lambda.
    """
    h = fractal_hash(concept)
    v = np.zeros(d, dtype=np.float64)
    # Expansion fractale
    for i in range(d):
        v[i] = np.sin(2.0 * np.pi * h * (lambda_ ** (-i)))

    # Normalisation L2 (pour l'espace de Hilbert)
    norm = np.linalg.norm(v)
    if norm > 0:
        v = v / norm
    return v

def generate_gauge_matrices(output_path='nfn_os/gauge_matrices.npz'):
    """
    Génère les matrices de jauge sémantique G_Sem pour chaque langue.
    Chaque jauge est une matrice orthogonale (d, d).
    """
    print("Génération des invariants de jauge sémantique...")
    np.random.seed(42) # Déterministe pour que la génération soit la même

    # Jauge neutre (espace pivot)
    g_pivot = np.eye(d)

    # Jauges pour quelques langues
    g_fr = ortho_group.rvs(dim=d)
    g_en = ortho_group.rvs(dim=d)

    matrices = {
        'pivot': g_pivot,
        'fr': g_fr,
        'en': g_en
    }

    np.savez(output_path, **matrices)
    print(f"Matrices de jauge sauvegardées dans {output_path}")

def generate_universal_facts(db_path='nfn_os/universal_facts.db'):
    """
    Génère la base de faits universelle en appliquant le théorème
    de représentation universelle.
    """
    print("Génération de l'espace de Hilbert fractal des faits...")

    # Une petite ontologie fondamentale pour démontrer le système
    raw_facts = [
        {"subject": "France", "relation": "capital", "value": "Paris", "proof": ["France is a country", "Paris is a city", "Paris is the capital of France"]},
        {"subject": "Earth", "relation": "shape", "value": "Oblate Spheroid", "proof": ["Earth is a planet", "Gravity pulls mass into a sphere", "Rotation causes bulging at the equator"]},
        {"subject": "Water", "relation": "boiling_point", "value": "100C", "proof": ["Water is H2O", "At 1 atm pressure, kinetic energy overcomes intermolecular forces at 100C"]},
        {"subject": "Bitcoin", "relation": "creator", "value": "Satoshi Nakamoto", "proof": ["Bitcoin is a cryptocurrency", "The whitepaper was published in 2008 by Satoshi Nakamoto"]},
    ]

    # Initialisation de la base de données
    if os.path.exists(db_path):
        os.remove(db_path)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE facts (
            id TEXT PRIMARY KEY,
            subject TEXT,
            relation TEXT,
            value TEXT,
            point BLOB,
            proof_chain TEXT
        )
    ''')

    for f in raw_facts:
        subj = f["subject"]
        rel = f["relation"]

        # Le point de requête est le produit tensoriel fractal: Q = x ⊗ r
        # Pour rester en dimension d, on utilise le produit d'Hadamard (qui préserve l'espace)
        # comme approximation du produit tensoriel sur la variété fractale de dimension finie.
        v_subj = embed(subj)
        v_rel = embed(rel)
        p_fact = v_subj * v_rel

        # Re-normaliser
        norm = np.linalg.norm(p_fact)
        if norm > 0:
            p_fact = p_fact / norm

        point_blob = p_fact.tobytes()
        proof_str = json.dumps(f["proof"])
        fact_id = f"{subj}_{rel}"

        cursor.execute('''
            INSERT INTO facts (id, subject, relation, value, point, proof_chain)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (fact_id, subj, rel, f["value"], point_blob, proof_str))

    conn.commit()
    conn.close()
    print(f"Base de connaissances universelle générée dans {db_path} avec {len(raw_facts)} entités fondamentales.")

if __name__ == "__main__":
    generate_gauge_matrices()
    generate_universal_facts()
