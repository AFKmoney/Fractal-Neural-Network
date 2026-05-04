import numpy as np

g_L_dict = {}
_gauges_loaded = False

def load_gauges():
    global g_L_dict, _gauges_loaded
    if _gauges_loaded:
        return
    try:
        gauge_data = np.load('nfn_os/gauge_matrices.npz')
        g_L_dict = {lang: gauge_data[lang] for lang in gauge_data.files}
        _gauges_loaded = True
    except FileNotFoundError:
        print("ATTENTION: gauge_matrices.npz introuvable.")
        g_L_dict = {}

def apply_gauge(vector: np.ndarray, lang: str) -> np.ndarray:
    """
    Applique la transformation de jauge g_L pour projeter
    une représentation spécifique à une langue vers l'espace pivot neutre.
    u_pivot = g_L * u
    """
    load_gauges()
    if lang not in g_L_dict:
        # Par défaut, aucune transformation
        return vector

    g_L = g_L_dict[lang]
    return np.dot(g_L, vector)

def apply_inverse_gauge(vector: np.ndarray, lang: str) -> np.ndarray:
    """
    Applique la transformation de jauge inverse g_L^-1 pour projeter
    depuis l'espace pivot vers la langue cible.
    v_L = g_L^-1 * v_pivot
    """
    load_gauges()
    if lang not in g_L_dict:
        return vector

    g_L = g_L_dict[lang]
    # Puisque g_L est orthogonale, l'inverse est sa transposée
    g_L_inv = g_L.T
    return np.dot(g_L_inv, vector)

def parse_question(question: str, lang: str):
    """
    Parseur linguistique rudimentaire qui extrait le (sujet, relation)
    à partir d'une phrase en langue naturelle.
    Dans un système complet, ceci est un modèle NLP (lui-même encodé fractalement).
    Pour cette implémentation divine, nous utilisons une table de correspondance exacte.
    """
    # Espace de règles basique pour démontrer l'invariance sémantique
    # On ramène la requête aux concepts universels
    q_lower = question.lower()

    if lang == 'fr':
        if "capitale" in q_lower and "france" in q_lower:
            return "France", "capital"
        elif "forme" in q_lower and "terre" in q_lower:
            return "Earth", "shape"
        elif "ébullition" in q_lower and "eau" in q_lower:
            return "Water", "boiling_point"
        elif "créateur" in q_lower and "bitcoin" in q_lower:
            return "Bitcoin", "creator"

    elif lang == 'en':
        if "capital" in q_lower and "france" in q_lower:
            return "France", "capital"
        elif "shape" in q_lower and "earth" in q_lower:
            return "Earth", "shape"
        elif "boil" in q_lower and "water" in q_lower:
            return "Water", "boiling_point"
        elif "creator" in q_lower and "bitcoin" in q_lower:
            return "Bitcoin", "creator"

    # Fallback générique
    # Dans la réalité fractale de NFN-OS, chaque concept serait mappé
    return "Unknown", "Unknown"

def format_proof_chain(proof: list, lang: str) -> str:
    """
    Formate la chaîne de preuve géodésique.
    """
    if not proof:
        return ""

    intro = "Chaîne de preuve géodésique :" if lang == 'fr' else "Geodesic proof chain:"
    formatted = f"\n{intro}\n"
    for i, step in enumerate(proof):
        formatted += f"  [{i+1}] ──→ {step}\n"
    return formatted
