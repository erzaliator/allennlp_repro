from preprocessing import construct_dataset
from logger import Logger
from iterators import get_iterators

BATCH_SIZE = 4
BERT_MODEL = 'bert-base-german-cased'
logger = Logger()
train_path = '../../processed/deu.rst.pcc_train_enriched.rels'
test_path = '../../processed/deu.rst.pcc_test_enriched.rels'
valid_path = '../../processed/deu.rst.pcc_valid_enriched.rels'
train_dataset, test_dataset, valid_dataset = construct_dataset(train_path, test_path, valid_path, logger)
get_iterators(train_dataset, test_dataset, valid_dataset, BATCH_SIZE, BERT_MODEL, logger)