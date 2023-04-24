from lang2vec import syntactic_distance as sd
from lang2vec import geographic_distance as geod
from lang2vec import phonological_distance as pd
from lang2vec import genetic_distance as gend
from lang2vec import inventory_distance as id
from lang2vec import featural_distance as fd

from lang2vec import get_features

from sklearn.metrics.pairwise import cosine_similarity

# from lang2vec import LEARNED_LETTER_CODES as llc
# print(llc)

# https://en.wikipedia.org/wiki/Wikipedia:WikiProject_Languages/List_of_ISO_639-3_language_codes_(2019)
lang_codes = {
'deu.rst.pcc': "deu",
'eng.pdtb.pdtb': "eng",
'eng.rst.gum': "eng",
'eng.rst.rstdt': "eng",
'eng.sdrt.stac': "eng",
'fas.rst.prstc': "prq",
'fra.sdrt.annodis': "fra",
'nld.rst.nldt': "nld",
'por.rst.cstn': "por",
'rus.rst.rrt': "rus",
'spa.rst.rststb': "spa",
'spa.rst.sctb': "spa",
'tur.pdtb.tdb': "tur",
'zho.rst.sctb': "zho",
}

genres = {
'deu.rst.pcc': set(["news"]),
'eng.pdtb.pdtb': set(["news"]),
'eng.rst.gum': set(["political", "news", "science", "academic", "fiction", "chats"]),
'eng.rst.rstdt': set(["news"]),
'eng.sdrt.stac': set(["chats"]),
'fas.rst.prstc': set(["news"]),
'fra.sdrt.annodis': set(["news", "academic", "political"]),
'nld.rst.nldt': set(["science", "news", "academic"]),
'por.rst.cstn': set(["news"]),
'rus.rst.rrt': set(["news", "science"]),
'spa.rst.rststb': set(["science", "academic"]),
'spa.rst.sctb': set(["science", "academic", "news", "commercial"]),
'tur.pdtb.tdb': set(["fiction", "academic" "news"]),
'zho.rst.sctb': set(["science", "academic", "news", "commercial"]),
}

genres_rich = {
'deu.rst.pcc': {"news":100},
'eng.pdtb.pdtb': {"news":100}, 
'eng.rst.gum': {"political":13.276247, "news":7.706699, "science":13.204289, "academic":19.953947, "fiction":12.592646, "chats":33.266173},
'eng.rst.rstdt': {"news":100},
'eng.sdrt.stac': {"chats":100},
'fas.rst.prstc': {"news":100},
'fra.sdrt.annodis': {"news":39.633867, "academic":58.489703, "political":1.876430},
'nld.rst.nldt': {"science": 50.808458, "commercial":49.191542},
'por.rst.cstn': {"news":100},
'rus.rst.rrt': {"academic":37.492193, "news":33.817917, "science":28.68989},
'spa.rst.rststb': {"science":77.455357, "academic":22.544643},
'spa.rst.sctb': {'science':55.125285, 'academic':27.107062, 'news': 9.339408, 'political':8.428246},
'tur.pdtb.tdb': {"fiction":33.33, "academic":33.33, "news":33.33},
'zho.rst.sctb': {'science':55.125285, 'academic':27.107062, 'news': 9.339408, 'political':8.428246},
}

def get_genre_overlap(a, b):
    # 0 means no overlap, 1 means partial overlap and 2 means overlap
    if len(a.intersection(b)) == 0: return 0.0
    else:
        if a.intersection(b)==a and a.intersection(b)==b: return 2.0
        else: return 1.0

def get_rich_genre_overlap(a, b):
    genres = ['news', 'political', 'science', 'academic', 'fiction', 'chats']
    a_genre = [a[g] if g in a else 0.0 for g in genres]
    b_genre = [b[g] if g in b else 0.0 for g in genres]
    overlap = [min(x,y) for x,y in zip(a_genre,b_genre)]
    return overlap


formality = {
'deu.rst.pcc': set(["yes"]),
'eng.pdtb.pdtb': set(["yes"]),
'eng.rst.gum': set(["yes", "no"]),
'eng.rst.rstdt': set(["yes"]),
'eng.sdrt.stac': set(["no"]),
'fas.rst.prstc': set(["yes"]),
'fra.sdrt.annodis': set(["yes"]),
'nld.rst.nldt': set(["yes", "no"]),
'por.rst.cstn': set(["yes"]),
'rus.rst.rrt': set(["yes"]),
'spa.rst.rststb': set(["yes"]),
'spa.rst.sctb': set(["yes", "no"]),
'tur.pdtb.tdb': set(["yes"]),
'zho.rst.sctb': set(["yes", "no"]),
}


f = open('uriel_kaveri.tsv', 'w')
header = ['dataset1', 'dataset2', 'phonological', 'syntactic', 'geographic', 'genetic', 'inventory', 'featural', 'formalism', 'genre_overlap', 'formality_overlap']
header = ['dataset1', 'dataset2', 'genre_overlap', 'formality_overlap', 'genreoverlap_news', 'genreoverlap_political', 
          'genreoverlap_science', 'genreoverlap_academic', 'genreoverlap_fiction', 'genreoverlap_chats']
f.write('\n'+'\t'.join(header))
print(header)
for key1 in lang_codes:
    code1 = lang_codes[key1]
    for key2 in lang_codes:
        code2 = lang_codes[key2]
        formalism = float(key1.split('.')[1]==key2.split('.')[1])
        genre_overlap = get_genre_overlap(genres[key1], genres[key2])
        genre_rich_overlap = get_rich_genre_overlap(genres_rich[key1], genres_rich[key2])
        formality_overlap = get_genre_overlap(formality[key1], formality[key2])
        # row = [key1, key2, str(pd(code1, code2)), str(sd(code1, code2)), str(geod(code1, code2)), str(gend(code1, code2)), str(id(code1, code2)), str(fd(code1, code2)), str(formalism), str(genre_overlap), str(formality_overlap)]
        row = [key1, key2, str(genre_overlap), str(formality_overlap), 
               str(genre_rich_overlap[0]), str(genre_rich_overlap[1]), str(genre_rich_overlap[2]), str(genre_rich_overlap[3]), str(genre_rich_overlap[4]), str(genre_rich_overlap[5])]
        f.write('\n'+'\t'.join(row))
        print('\t'.join(row))