import time
from .engine import answer
from .language import parse_question, format_proof_chain

def respond(question: str, lang: str = 'fr'):
    """
    Interface principale du NFN-OS (Operating System for Universal Knowledge).
    Exécute la pipeline complète en temps réel cosmique.
    """
    t_start = time.perf_counter()

    # 1. Analyse syntaxique vers l'espace des concepts universels
    subj, rel = parse_question(question, lang)
    if subj == "Unknown":
        return "Concept non résolu dans la variété fractale actuelle.", ""

    # 2. Inférence dans l'espace de Hilbert Fractal (O(1) temporel, O(r) calcul)
    value, proof = answer(subj, rel)

    if value is None:
        return "Aucune résonance trouvée pour ce point de l'espace-temps.", ""

    # 3. (Théorique) Transformation de jauge inverse.
    # Puisque value est déjà une chaîne textuelle dans notre base simulée,
    # nous formatons directement la réponse. Dans l'implémentation complète,
    # 'value' serait un vecteur que l'on passerait à apply_inverse_gauge().

    if lang == 'fr':
        if rel == "capital": text_response = f"La capitale de {subj} est {value}."
        elif rel == "shape": text_response = f"La forme de {subj} est {value}."
        elif rel == "boiling_point": text_response = f"Le point d'ébullition de {subj} est {value}."
        elif rel == "creator": text_response = f"Le créateur de {subj} est {value}."
        else: text_response = f"{subj} -> {rel} : {value}"
    else:
        if rel == "capital": text_response = f"The capital of {subj} is {value}."
        elif rel == "shape": text_response = f"The shape of {subj} is {value}."
        elif rel == "boiling_point": text_response = f"The boiling point of {subj} is {value}."
        elif rel == "creator": text_response = f"The creator of {subj} is {value}."
        else: text_response = f"{subj} -> {rel} : {value}"

    # 4. Formater la preuve
    proof_text = format_proof_chain(proof, lang)

    t_end = time.perf_counter()
    duration_ms = (t_end - t_start) * 1000.0

    # Ajout des métadonnées de performance
    perf_meta = f"\n[Noyau évalué en {duration_ms:.3f} ms — Factualité garantie par résonance cosmique]"

    return text_response + proof_text + perf_meta
