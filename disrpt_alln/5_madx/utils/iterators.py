from transformers import AutoTokenizer, BertTokenizer
from datasets import ClassLabel
import torch



def get_iterators(train_dataset, test_dataset, valid_dataset, BATCH_SIZE, BERT_MODEL, logger):
    labels = ClassLabel(names=list(set(train_dataset['label'])|set(test_dataset['label'])|set(valid_dataset['label'])))

    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
    class Dataset(torch.utils.data.Dataset):
        def __init__(self, dataset, labels, raw_text=False):
            self.text = []
            self.raw_text = []
            self.raw_label = []
            self.raw_text_flag = raw_text
            for premise, hypothesis in zip(dataset['unit1_txt'], dataset['unit2_txt']):
                self.text.append(tokenizer.encode_plus(premise, hypothesis, padding="max_length", truncation=True, max_length=512))
                if raw_text: self.raw_text.append([premise, hypothesis])
            # self.labels = torch.tensor(labels.str2int(dataset['label'])).to(device)
            self.labels = labels.str2int(dataset['label'])
            if raw_text: self.raw_label = dataset['label']
            print('read ' + str(len(self.text)) + ' examples')

        def __getitem__(self, idx):
            if self.raw_text_flag:  
                return {'input_ids':self.text[idx]['input_ids'], 
                    'token_type_ids':self.text[idx]['token_type_ids'], 
                    'attention_mask':self.text[idx]['attention_mask'], 
                    'raw_text': self.raw_text[idx],
                    'label':self.labels[idx],
                    'raw_label': self.raw_label[idx]}

            return {'input_ids':self.text[idx]['input_ids'], 
                    'token_type_ids':self.text[idx]['token_type_ids'], 
                    'attention_mask':self.text[idx]['attention_mask'], 
                    'label':self.labels[idx]}

        def __len__(self):
            return len(self.text)
        
        def combine_with_another_dataset(self, dataset2):
            print('Adding datasets of sizes ', self.num_rows, ', ', dataset2.num_rows)
            self.text = self.text + dataset2.text
            self.raw_text = self.raw_text + dataset2.raw_text
            self.raw_label = self.raw_label + dataset2.raw_label
            self.num_rows = len(self.text) #update texts is now longer
            assert self.raw_text_flag == dataset2.raw_text_flag
            print('To new dataset size: ', self.num_rows)


    def load_data_snli(batch_size, labels):
        """Download the SNLI dataset and return data iterators and vocabulary."""
        train_data = train_dataset
        valid_data = valid_dataset
        test_data = test_dataset
        train_set = Dataset(train_data, labels, raw_text=False)
        valid_set = Dataset(valid_data, labels, raw_text=False)
        test_set = Dataset(test_data, labels, raw_text=False)
        train_iter = torch.utils.data.DataLoader(train_set, batch_size,
                                                shuffle=True)
        valid_iter = torch.utils.data.DataLoader(valid_set, batch_size,
                                                shuffle=False)
        test_iter = torch.utils.data.DataLoader(test_set, batch_size,
                                                shuffle=False)
        
        return train_set, valid_set, test_set, labels

    train_dataset, valid_dataset, test_dataset, labels = load_data_snli(BATCH_SIZE, labels)
    

    return train_dataset, valid_dataset, test_dataset, labels