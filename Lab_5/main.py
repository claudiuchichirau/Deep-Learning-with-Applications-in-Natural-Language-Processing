
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



# 1. Flying planes can be dangerous.
# 2. The parents of the bride and the groom were flying.
# 3. The groom loves dangerous planes more than the bride.


full_grammar = """
S -> NP VP
NP -> Det N | Adj N | Gerund N | N | NP PP | NP CONJ NP | Det Adj N | Adj N NP || PP Det N
VP -> V | V NP | V PP | Aux VP | Aux V Adj | V NP COMP
PP -> P NP | COMP_ADV P
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






import nltk
from nltk import CFG, ChartParser

grammar = CFG.fromstring(full_grammar)
parser = ChartParser(grammar)

sentences = [
    "Flying planes can be dangerous",
    "The parents of the bride and the groom were flying",
    "The groom loves dangerous planes more than the bride"
]

for sent in sentences:
    print(f"\nSentence: {sent}")
    for tree in parser.parse(sent.split()):
        print(tree)
        tree.pretty_print()
