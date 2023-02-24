import spacy
from spacy.tokenizer import Tokenizer
from spacy.lang.char_classes import ALPHA, ALPHA_LOWER, ALPHA_UPPER, CONCAT_QUOTES, LIST_ELLIPSES, LIST_ICONS
from spacy.symbols import ORTH
from spacy.util import compile_infix_regex

class SpacyTokenizer:
    def __init__(self   ):
        '''Tokenize sentences for disrpt. Works well for en and nld.
        *******Usage********
        Stok = SpacyTokenizer()
        print(Stok.tokenize_sents("(bijv. Hartsell-Gundy et al., 2015;"))
        print([Stok.spacify_punkts("(bijv. Hartsell-Gundy et al., 2015;")])
        '''
        self.nlp = spacy.load("en_core_web_sm")
        self.nlp.tokenizer = self.custom_tokenizer(self.nlp)

    def custom_tokenizer(self, nlp):
        infixes = (
            LIST_ELLIPSES
            + LIST_ICONS
            + [
                r"(?<=[0-9])[+\-\*^](?=[0-9-])",
                r"(?<=[{al}{q}])\.(?=[{au}{q}])".format(
                    al=ALPHA_LOWER, au=ALPHA_UPPER, q=CONCAT_QUOTES
                ),
                r"(?<=[{a}]),(?=[{a}])".format(a=ALPHA),
                #r"(?<=[{a}])(?:{h})(?=[{a}])".format(a=ALPHA, h=HYPHENS),
                r"(?<=[{a}0-9])[:<>=/](?=[{a}])".format(a=ALPHA),
            ]
        )

        infix_re = compile_infix_regex(infixes)
        special_cases = {"<*>": [{"ORTH": "<*>"}], 
                            '""': [{"ORTH": '"'}, {"ORTH": '"'}], 
                            "''": [{"ORTH": "'"}, {"ORTH": "'"}],
                            "'\"": [{"ORTH": "'"}, {"ORTH": '"'}],
                            "\"'": [{"ORTH": '"'}, {"ORTH": "'"}],
                            "'s": [{"ORTH": "'s"}],
                            "zo'n": [{"ORTH": "zo'n"}]} #disrpt rule
                            #manual rules:
                            # convert "" -> "
                            # 1. column search ." -" ," ?" )" "
                            # 2. column search "
                            # 3. column search [ ... ] to [...]
                            # 4. run word-colon spacer **********

        return Tokenizer(nlp.vocab, prefix_search=nlp.tokenizer.prefix_search,
                                    suffix_search=nlp.tokenizer.suffix_search,
                                    infix_finditer=infix_re.finditer,
                                    token_match=nlp.tokenizer.token_match,
                                    rules=special_cases)

    def tokenize_sents(self, sent):
        return [t.text for t in self.nlp(sent)]

    def spacify_punkts(self, sent):
        spacified_list = self.tokenize_sents(sent)
        spacified_string = ' '.join(spacified_list)
        return spacified_string

    def check_is_alpha_num_sym(self, c):
        return c.isalnum() or c in ['.', ',', '-', '!', '?', ';', ':', '<', '>']

    def spacify_inverted_commas(self, s):
        '''Ensures that inverted commas have spaces after them in case they are next to a sentence'''
        # s = '"It \'s so much better to wake up to the sound of somebody talking to you : your favorite radio station , some music , whatever , instead of that classic iPhone alarm , "" ba , ba , ba . """'
        t = ''

        if '"' in s:
            print("in . "+s)
        for index in range(len(s)):
            c = s[index]
            if c not in ['"', "'"]:
                t += c
            else:
                if index!=0:
                    if self.check_is_alpha_num_sym(s[index-1]):
                        t += ' '
                t += c
                if index!=len(s)-1:
                    if self.check_is_alpha_num_sym(s[index+1]):
                        t += ' '
        if '"' in s:
            print("out. "+t)
        return t

# Stok = SpacyTokenizer()
# print([Stok.spacify_punkts('"maar zich niet weerhouden van “ gewelddadige "" aanvallen in een poging"')])
# print(Stok.tokenize_sents("(bijv. Hartsell-Gundy et al., 2015; This ''\" is \"'\"something '\"'from (L2). This is \"'sam. \"Over zijn \"\" beschermeling \"\" schreef hij ,\""))
# print([Stok.spacify_inverted_commas('"dat het in het echte leven echt een beetje magisch was , """')])
