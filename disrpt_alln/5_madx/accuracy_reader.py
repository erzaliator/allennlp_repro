import json
import pandas as pd
from os import listdir
from os.path import isdir, join


# get_lang1_or_lang2 = 1 # set 1 for first lang1 or 2 for lang2
folder_name = '/home/VD/kaveri/bert_categorical_tutorial/allennlp_repro/disrpt_alln/5_madx/runs/full_shot/same_pretrain/'

lang_list = [
'deu.rst.pcc', 
'eng.pdtb.pdtb',
'eng.rst.gum', 
'eng.rst.rstdt', 
'eng.sdrt.stac', 
'fas.rst.prstc',
'fra.sdrt.annodis', 
'nld.rst.nldt', 
'por.rst.cstn', 
'rus.rst.rrt', 
'spa.rst.rststb', 
'spa.rst.sctb', 
'tur.pdtb.tdb', 
'zho.rst.sctb']

df_feats = pd.read_csv('./LMFeatures.tsv',sep='\t')
df_feats = df_feats[:-1]

def get_accuracy_from_json_file_name(file_name):
    dataset_name1 = file_name.split('_')[-2].replace('/metrics.json', '')
    dataset_name2 = file_name.split('_')[-1].replace('/metrics.json', '')
    json_obj = open(file_name, 'r').readlines()[0]
    json_obj = json_obj.replace("'", '"')
    file = json.loads(json_obj)
    # if (dataset_name2==lang2 and get_lang1_or_lang2==2) or (dataset_name1==lang1 and get_lang1_or_lang2==1):
    # result += '\n'+file_name.split('_')[-1].replace('/metrics.json', '')+'\t'+str(file['acc']*100)
    result_num = file['acc']*100
    return result_num

def get_feat_list(lang1, lang2, df_feats):
    header = df_feats.columns.to_list()
    for remove_col in ['Dataset1', 'Dataset2', 'Model absolute', 'Model gains', 'Model Percentage Gain']:
        header.remove(remove_col)
    feat_list = []
    for feature_name in header:
        feat_list.append(str(df_feats[df_feats['Dataset2']==lang2][df_feats['Dataset1']==lang1][feature_name].item()))
    return feat_list, header

# experiment_name = 'SingleEpochPass=v4_'
# experiment_name = 'FullShot=v4_'
df_experiment_list = []
for experiment_name in ['CheckFullShot=v4_finetune_', 'CheckFullShot=v5_finetune_', 'CheckFullShot=v6_finetune_']:
    df_experiment = pd.DataFrame(columns=lang_list+['lang1'])
    for lang1 in lang_list:
        row = {'lang1': lang1}
        for lang2 in lang_list:
            try:
                if lang1==lang2: 
                    search_prefix = experiment_name.replace('finetune', 'pretrain')+lang1
                    files = [f for f in listdir(folder_name) if isdir(join(folder_name, f)) and search_prefix in f]
                    file = files[0]
                    file_name = folder_name+file+'/metrics.json'
                    result_num = get_accuracy_from_json_file_name(file_name)
                    row[lang2] = result_num

                else:
                    
                        search_prefix = experiment_name+lang1+'_'+lang2
                        files = [f for f in listdir(folder_name) if isdir(join(folder_name, f)) and search_prefix in f]
                        result = ''

                        for file in files:
                            file_name = folder_name+file+'/metrics.json'
                            result_num = get_accuracy_from_json_file_name(file_name)
                            row[lang2] = result_num
            except:
                row[lang2] = -1
        df_experiment = pd.concat([df_experiment, pd.DataFrame.from_records([row])])

    df_experiment.set_index('lang1', inplace=True)
    print(df_experiment)
    df_experiment_list.append(df_experiment)


# f = open('latest_features_runfix.tsv', 'w')
# flag=1
# for run in range(3):
#     for lang1 in lang_list:
#         for lang2 in lang_list:
#             feat_list, header = get_feat_list(lang1, lang2, df_feats)
#             header = ['Run', 'Dataset1', 'Dataset2']+header+['Model Percentage Gain', 'Model gains', 'Model absolute']
#             # average = (df_experiment_list[0][lang2][lang1]+df_experiment_list[1][lang2][lang1]+df_experiment_list[2][lang2][lang1])/3
#             # baseline_average = (df_experiment_list[0][lang2][lang2]+df_experiment_list[1][lang2][lang2]+df_experiment_list[2][lang2][lang2])/3
#             average = (df_experiment_list[run][lang2][lang1])
#             baseline_average = (df_experiment_list[run][lang2][lang2])
#             absolute_gains = str(average-baseline_average)
#             relative_gains = str((average-baseline_average)*100/baseline_average)
#             if flag:
#                 f.write('\t'.join(header)+'\n')
#                 flag=0
#             print('\t'.join([str(run), lang1, lang2, *feat_list, relative_gains, absolute_gains, str(average)]))
#             f.write('\t'.join([str(run), lang1, lang2, *feat_list, relative_gains, absolute_gains, str(average)])+'\n')