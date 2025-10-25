import nltk, spacy
from nltk import CFG, ChartParser, Nonterminal, PCFG, Production, ProbabilisticProduction
from nltk.grammar import induce_pcfg, Nonterminal
from nltk.parse.generate import generate
from spacy import displacy


# 1. Flying planes can be dangerous.
grammar_1 = """
S → NP VP
NP → Gerund N | N
VP → Aux V Adj
Adj → 'dangerous'
Gerund → 'Flying'
N → 'planes'
Aux → 'can'
V → 'be'
"""


# 2. The parents of the bride and the groom were flying.
grammar_2 = """
S → NP VP
NP → NP Conj NP | N
VP → V Gerund
Gerund → 'flying'
N → 'parents' | 'bride' | 'groom'
V → 'were'
Conj → 'and'
"""


# 3. The groom loves dangerous planes more than the bride.
grammar_3 = """
S -> NP VP
NP -> Det N | Adj N NP | PP Det N
PP -> COMP_ADV P
Det -> 'The'
N -> 'groom' | 'planes' | 'bride'

VP -> V NP
V -> 'loves'
Adj -> 'dangerous' 
P -> 'than' 
COMP_ADV -> 'more'
"""


# ex 1
def define_grammar():
    full_grammar = """
        S -> NP VP
        NP -> Det N | Det Adj N | Adj N | Gerund N | NP PP | NP CONJ NP | NP COMP | N
        VP -> Aux V Adj | Aux V | V NP | V NP COMP               
        PP -> P NP 
        COMP -> COMP_ADV P NP   
        Det -> 'the' | 'The'
        N -> 'parents' | 'bride' | 'groom' | 'planes'
        V -> 'loves' | 'were' | 'flying' | 'be'
        Adj -> 'dangerous'
        Gerund -> 'Flying'
        Aux -> 'can' | 'were'
        P -> 'of' | 'than'
        CONJ -> 'and'
        COMP_ADV -> 'more'
        """

    return CFG.fromstring(full_grammar)

# `ex 2
def parse_sentences(grammar):
    parser = ChartParser(grammar)

    sentences = [
        "Flying planes can be dangerous",
        "The parents of the bride and the groom were flying",
        "The groom loves dangerous planes more than the bride"
    ]

    for sent in sentences:
        print(f"\nSentence: {sent}")
        parses = list(parser.parse(sent.split()))

        for i, tree in enumerate(parses, 1):
            print(f"Tree {i}:")
            tree.pretty_print()

# ex 3
def load_dependency_parser():
    nlp = spacy.load("en_core_web_sm")
    return nlp

def dependency_analysis(nlp):
    sentences = [
        "Flying planes can be dangerous",
        "The parents of the bride and the groom were flying",
        "The groom loves dangerous planes more than the bride"
    ]

    for sent in sentences:
        doc = nlp(sent)
        print(f"\nSentence: {sent}")
        print("Word\t\t→ Dependency\t→ Head")
        print("-" * 45)
        for token in doc:
            print(f"{token.text:12}\t→ {token.dep_:15}\t→ {token.head.text}")


def visualize_dependency_trees(nlp):
    sentences = [
        "Flying planes can be dangerous",
        "The parents of the bride and the groom were flying",
        "The groom loves dangerous planes more than the bride"
    ]

    for sent in sentences:
        doc = nlp(sent)
        print(f"\nVisualizing dependency tree for: {sent}")
        displacy.serve(doc, style="dep", host="localhost", port=5000)

## Bonus
def CFG_to_Chomsky_normal_form(grammar):
    print("Original Grammar:\n", grammar)

    cnf_grammar = grammar.chomsky_normal_form()
    print("\nChomsky Normal Form Grammar:\n", cnf_grammar)
    return cnf_grammar


def binarize_rules(grammar):
    new_rules = []
    transformations = {} 
    counter = 1

    for prod in grammar.productions():
        lhs, rhs = prod.lhs(), prod.rhs()
        prob = prod.prob() if hasattr(prod, "prob") else None

        if len(rhs) <= 2:
            new_rules.append(prod)
            continue

        current_lhs = lhs
        generated_rules = []

        for i in range(len(rhs) - 2):
            new_nt = Nonterminal(f"X{counter}")
            counter += 1

            if prob is not None:
                new_rule = ProbabilisticProduction(current_lhs, [rhs[i], new_nt], prob=prob if i == 0 else 1.0)
            else:
                new_rule = Production(current_lhs, [rhs[i], new_nt])

            new_rules.append(new_rule)
            generated_rules.append(str(new_rule))
            current_lhs = new_nt

        # ultima regula
        if prob is not None:
            last_rule = ProbabilisticProduction(current_lhs, rhs[-2:], prob=1.0)
        else:
            last_rule = Production(current_lhs, rhs[-2:])

        new_rules.append(last_rule)
        generated_rules.append(str(last_rule))

        transformations[str(prod)] = generated_rules

    return new_rules, transformations


def replace_terminals(rules):
    new_rules = []
    transformations = {}
    counter = 100
    terminal_map = {}

    for prod in rules:
        lhs, rhs = prod.lhs(), prod.rhs()
        prob = prod.prob() if hasattr(prod, "prob") else None
        new_rhs = []
        generated_rules = []

        for symbol in rhs:
            if isinstance(symbol, str):
                if symbol not in terminal_map:
                    new_nt = Nonterminal(f"T{counter}")
                    counter += 1
                    terminal_map[symbol] = new_nt
                    if prob is not None:
                        new_rules.append(ProbabilisticProduction(new_nt, [symbol], prob=1.0))
                    else:
                        new_rules.append(Production(new_nt, [symbol]))
                    generated_rules.append(f"{new_nt} -> '{symbol}'")
                new_rhs.append(terminal_map[symbol])
            else:
                new_rhs.append(symbol)

        if prob is not None:
            new_prod = ProbabilisticProduction(lhs, new_rhs, prob=prob)
        else:
            new_prod = Production(lhs, new_rhs)

        new_rules.append(new_prod)
        if generated_rules:
            transformations[str(prod)] = generated_rules

    return new_rules, transformations


def remove_unit_productions(rules):
    new_rules = []
    transformations = {}
    unit_pairs = []

    for prod in rules:
        lhs, rhs = prod.lhs(), prod.rhs()
        prob = prod.prob() if hasattr(prod, "prob") else None

        if len(rhs) == 1 and isinstance(rhs[0], Nonterminal):
            unit_pairs.append((lhs, rhs[0]))
        else:
            new_rules.append(prod)

    for (A, B) in unit_pairs:
        generated_rules = []
        for prod in rules:
            if prod.lhs() == B and not (len(prod.rhs()) == 1 and isinstance(prod.rhs()[0], Nonterminal)):
                prob = prod.prob() if hasattr(prod, "prob") else None
                if prob is not None:
                    new_prod = ProbabilisticProduction(A, prod.rhs(), prob=prob)
                else:
                    new_prod = Production(A, prod.rhs())
                new_rules.append(new_prod)
                generated_rules.append(str(new_prod))
        transformations[f"{A} -> {B}"] = generated_rules

    return new_rules, transformations


def print_transformations(title, transformations):
    print(f"\n{title}:")
    if not transformations:
        print("No transformations performed.")
    for original, derived in transformations.items():
        print(f"\n\t - Original: {original}")
        for d in derived:
            print(f"\t      - CNF: {d}")

def print_grammar(rules, title):
    print(f"\n{title}:")
    for prod in rules:
        rhs = " ".join(str(x) for x in prod.rhs())
        prob = f" [{prod.prob():.2f}]" if hasattr(prod, "prob") else ""
        print(f"\t{prod.lhs()} -> {rhs}{prob}")


if __name__ == "__main__":
    grammar = define_grammar()

    parse_sentences(grammar)

    nlp = load_dependency_parser()

    dependency_analysis(nlp)
    visualize_dependency_trees(nlp)

    CFG_to_Chomsky_normal_form(grammar)


    ## bonus - cnf
    print_grammar(grammar.productions(), "Original Grammar")

    rules, t1 = binarize_rules(grammar)
    print_transformations("Step 1: Binarization", t1)

    rules, t2 = replace_terminals(rules)
    print_transformations("Step 2: Replace terminals", t2)

    rules, t3 = remove_unit_productions(rules)
    print_transformations("Step 3: Remove unit productions", t3)

    print_grammar(rules, "Final CNF Grammar")