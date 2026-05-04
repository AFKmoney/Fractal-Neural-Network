import argparse
import sys
from nfn_os.interface import respond

from nfn_os_setup import setup

def main():
    # Initialisation de l'univers si nécessaire
    setup()

    print("=" * 60)
    print(" NFN-OS v1.0 — Système d'Exploitation de la Connaissance")
    print(" Noyau Condensé Multi-dimensionnel Activé")
    print("=" * 60)

    parser = argparse.ArgumentParser(description="Query the NFN-OS Universal Knowledge Base.")
    parser.add_argument("-q", "--question", type=str, help="La question à poser (en langage naturel)")
    parser.add_argument("-l", "--lang", type=str, default="fr", choices=["fr", "en"], help="Langue de la requête (fr/en)")
    args = parser.parse_args()

    if args.question:
        # Exécution unique
        answer = respond(args.question, args.lang)
        print(f"\nQ: {args.question}")
        print(f"R: {answer}\n")
    else:
        # Mode interactif
        print("Entrez dans le flux de renormalisation. Tapez 'exit' pour quitter.")
        lang = args.lang
        while True:
            try:
                question = input(f"\n[{lang}] Requête > ")
                if question.lower() in ['exit', 'quit', 'q']:
                    break
                if not question.strip():
                    continue

                answer = respond(question, lang)
                print(f"R: {answer}")

            except KeyboardInterrupt:
                break

    print("\nFermeture du lien de résonance.")

if __name__ == "__main__":
    main()
