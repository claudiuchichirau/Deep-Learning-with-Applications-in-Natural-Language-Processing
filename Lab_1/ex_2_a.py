from nltk.corpus import wordnet as wn


def ex_1_wordnet_relations(word):
    synsets = wn.synsets(word)
    if not synsets:
        print(f"Nu am gasit cuvantul '{word}' in WordNet")
        return

    print(f"\nCuvant analizat: '{word}'")
    print("=" * 50)

    for idx, syn in enumerate(synsets):
        print(f"\nSENS: {idx+1}: {syn.name()}")
        print(f"Def: {syn.definition()}")
        print(f"Ex: {syn.examples()}\n")

        #Sinonime 
        synonyms = set()
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())

        print(f"Sinonime: {', '.join(synonyms) if synonyms else '—'}")

        #Antonime
        antonyms = set()
        for lemma in syn.lemmas():
            for ant in lemma.antonyms():
                antonyms.add(ant.name())

        print(f"Antonime: {', '.join(antonyms) if antonyms else '—'}")

        #Hiperonime 
        hypernyms = syn.hypernyms()
        if hypernyms:
            print("Hiperonime:")
            for h in hypernyms[:5]:
                print("  -", h.name(), "=", h.definition())
        else:
            print("Hiperonime: —")

        #Hiponime
        hyponyms = syn.hyponyms()
        if hyponyms:
            print("Hiponime:")
            for h in hyponyms[:5]:
                print("  -", h.name(), "=", h.definition())
        else:
            print("Hiponime: —")

        #Meronime
        meronyms = syn.part_meronyms()
        if meronyms:
            print("Meronime:")
            for m in meronyms[:5]:
                print("  -", m.name(), "=", m.definition())
        else:
            print("Meronime: —")

        print("-" * 50)


ex_1_wordnet_relations("tree")