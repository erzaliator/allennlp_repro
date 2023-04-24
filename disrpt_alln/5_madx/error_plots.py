import matplotlib.pyplot as plt
import pandas as pd



####ERROR PLOT
# names = ["Intercept", "Geographic", "Syntactic", "Genetic"     , "Entropy_pretrain_train", "Finetune.Training.set.size", "UniqueLabels_pretrain_test", "Formalism"]
# coeffs = [-5.156022e+01, 1.652709e+01 , -1.264581e+01, 1.005534e+01, 5.740592e+00, -4.246788e-04, 7.796086e-01, -2.774576e+00]
# low = [-68.698819058, 10.461049358 , -25.978193880, 0.859880548, 4.469601497, -0.000559422, 0.198623827, -6.223440575]
# high =  [-3.442162e+01 ,  2.259313e+01, 6.865744e-01, 1.925080e+01, 7.011582e+00, -2.899356e-04, 1.360593e+00, 6.742883e-01]

names = ['Geographic', 'Phonological', 'Syntactic', 'Genetic', 'Entropy_pretrain_train ', 'Finetune.Training.set.size', 'Pretrain.Training.set.size', 'KL_train ', 'UniqueLabels_finetune_test', 'UniqueLabels_pretrain_test', 'Formalism', 'Genre.overlap', 'Formality.overlap', ]
coeffs = [ 8.825e-01,  5.114e+00, -2.023e+01,  7.656e+00, -2.115e-01,  1.690e-04,  7.306e-04, -2.634e-01, -1.993e-01, -1.374e+00,  9.769e-01,  1.634e-01,  2.999e-01, ]
low = [-4.510794e+00, -4.117381e+00, -3.252499e+01, -5.652238e-01, -2.390186e+00,  4.385321e-05,  4.941464e-04, -9.178855e-01, -7.313278e-01, -1.916775e+00, -4.792289e+00, -2.491754e+00, -2.310323e+00, ]
high =  [6.275764e+00,  1.434586e+01, -7.933692e+00,  1.587687e+01,  1.967214e+00,  2.941170e-04,  9.671253e-04,  3.910175e-01,  3.327863e-01, -8.316474e-01,  6.746064e+00,  2.818576e+00,  2.910190e+00, ]

x = names[1:]
y = coeffs[1:]

fig = plt.figure()
asymmetric_error = [low[1:], high[1:]]

plt.errorbar(x, y, yerr=asymmetric_error, fmt='o')
plt.xticks(rotation = 90)
plt.tight_layout()

plt.savefig('./errorplot.png')



# ###BOX PLOT
# import seaborn as sns

# fig, axs = plt.subplots(2, 2)
# df = pd.read_csv('lang2vec/lang2vec/uriel_kaveri.tsv', header=0, sep='\t')
# print(df)
# axs[0][0] = sns.pointplot(data=df, x="dataset1", y="genre_overlap", linestyle='')
# plt.xticks(rotation = 90)
# plt.savefig('./errorplot.png')