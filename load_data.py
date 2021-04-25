import pickle as pickle
import os
import pandas as pd
import torch
from tqdm.auto import tqdm


# Dataset 구성.
class RE_Dataset(torch.utils.data.Dataset):
    def __init__(self, tokenized_dataset, labels):
        self.tokenized_dataset = tokenized_dataset
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.tokenized_dataset.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


class XLM_Dataset(torch.utils.data.Dataset):
    def __init__(self, tokenized_dataset, labels):
        self.tokenized_dataset = tokenized_dataset
        self.labels = labels

    def __getitem__(self, idx):
        item = {'input_ids': self.tokenized_dataset['input_ids'][idx],
                'attention_mask': self.tokenized_dataset['attention_mask'][idx],
                'labels': torch.tensor(self.labels[idx])}

        return item

    def __len__(self):
        return len(self.labels)


# 처음 불러온 tsv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다.
# 변경한 DataFrame 형태는 baseline code description 이미지를 참고해주세요.
def preprocessing_dataset_base(dataset, label_type):
    label = []
    for i in dataset[8]:
        if i == 'blind':
            label.append(100)
        else:
            label.append(label_type[i])
    out_dataset = pd.DataFrame(
        {'sentence': dataset[1], 'entity_01': dataset[2], 'entity_02': dataset[5], 'label': label, })
    return out_dataset


# tsv 파일을 불러옵니다.
def load_data(dataset_dir):
    # load label_type, classes
    with open('/opt/ml/input/data/label_type.pkl', 'rb') as f:
        label_type = pickle.load(f)
    # load dataset
    dataset = pd.read_csv(dataset_dir, delimiter='\t', header=None)
    # preprecessing dataset
    dataset = preprocessing_dataset(dataset, label_type)

    return dataset


# bert input을 위한 tokenizing.
# tip! 다양한 종류의 tokenizer와 special token들을 활용하는 것으로도 새로운 시도를 해볼 수 있습니다.
# baseline code에서는 2가지 부분을 활용했습니다.

def tokenized_dataset(dataset, tokenizer):
    concat_entity = []
    for e01, e02 in zip(dataset['entity_01'], dataset['entity_02']):
        temp = ''
        temp = e01 + '[SEP]' + e02
        concat_entity.append(temp)
    tokenized_sentences = tokenizer(
        concat_entity,
        list(dataset['sentence']),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=100,
        add_special_tokens=True,
    )
    print(tokenizer.tokenize(concat_entity[0]))
    print(tokenized_sentences[0])
    return tokenized_sentences


def QA_dataset(dataset, tokenizer):
    start_token = '[ENT]'
    end_token = '[/ENT]'
    concat_entity = []
    for e01, e02 in zip(dataset['entity_01'], dataset['entity_02']):
        temp = e01 + '(와/과) ' + e02 + \
               '(은/는) 무슨 관계일까?'
        concat_entity.append(temp)

    print(f'{concat_entity[0]} \n {dataset.sentence[0]}')
    tokenized_sentences = tokenizer(
        list(dataset['sentence']),
        concat_entity,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=100,
        add_special_tokens=True,
    )
    return tokenized_sentences

def preprocessing_dataset(dataset, label_type):
    label = []
    for i in dataset[8]:
        if i == 'blind':
            label.append(100)
        else:
            label.append(label_type[i])
    out_dataset = pd.DataFrame({'sentence':dataset[1],'entity_01':dataset[2], 'e1s':dataset[3],'e1e':dataset[4],
                              'entity_02':dataset[5], 'e2s':dataset[6],'e2e':dataset[7],'label':label})
    return out_dataset

def convert_to_features(train_dataset, tokenizer, max_len):
    max_seq_len = max_len
    cls_token = tokenizer.cls_token
    # cls_token_segment_id=tokenizer.cls_token_id
    cls_token_segment_id = 0
    sep_token = tokenizer.sep_token
    pad_token = 1
    pad_token_segment_id = 0
    sequence_a_segment_id = 0
    add_sep_token = False
    mask_padding_with_zero = True

    all_input_ids = []
    all_attention_mask = []
    all_token_type_ids = []
    all_e1_mask = []
    all_e2_mask = []
    # all_sep_mask = []
    total_length = []
    for idx in tqdm(range(len(train_dataset))):
        sorted_entity_index = sorted([train_dataset['e1s'][idx], train_dataset['e1e'][idx], train_dataset['e2s'][idx],
                                      train_dataset['e2e'][idx]])
        sentence = train_dataset['sentence'][idx]

        if train_dataset['e1s'][idx] > train_dataset['e2s'][idx]:
            train_dataset['sentence'][idx] = (sentence[:sorted_entity_index[0]] +
                                              ' <e2> ' +
                                              sentence[sorted_entity_index[0]: sorted_entity_index[1] + 1] +
                                              ' </e2> ' +
                                              sentence[sorted_entity_index[1] + 1: sorted_entity_index[2]] +
                                              ' <e1> ' +
                                              sentence[sorted_entity_index[2]: sorted_entity_index[3] + 1] +
                                              ' </e1> ' +
                                              sentence[sorted_entity_index[3] + 1:]
                                              # sep_token +
                                              # ' <e2> ' +
                                              # sentence[sorted_entity_index[0]: sorted_entity_index[1] + 1] +
                                              # ' </e2> ' +
                                              # sentence[sorted_entity_index[1] + 1: sorted_entity_index[2]] +
                                              # ' <e1> ' +
                                              # sentence[sorted_entity_index[2]: sorted_entity_index[3] + 1] +
                                              # ' </e1> ' +
                                              # sep_token
                                              )
        else:
            train_dataset['sentence'][idx] = (sentence[:sorted_entity_index[0]] +
                                              ' <e1> ' +
                                              sentence[sorted_entity_index[0]: sorted_entity_index[1] + 1] +
                                              ' </e1> ' +
                                              sentence[sorted_entity_index[1] + 1: sorted_entity_index[2]] +
                                              ' <e2> ' +
                                              sentence[sorted_entity_index[2]: sorted_entity_index[3] + 1] +
                                              ' </e2> ' +
                                              sentence[sorted_entity_index[3] + 1:]
                                              # sep_token +
                                              # ' <e1> ' +
                                              # sentence[sorted_entity_index[0]: sorted_entity_index[1] + 1] +
                                              # ' </e1> ' +
                                              # sentence[sorted_entity_index[1] + 1: sorted_entity_index[2]] +
                                              # ' <e2> ' +
                                              # sentence[sorted_entity_index[2]: sorted_entity_index[3] + 1] +
                                              # ' </e2> ' +
                                              # sep_token
                                              )

        token = tokenizer.tokenize(train_dataset['sentence'][idx])
        total_length.append(len(token))

        e11_p = token.index("<e1>")  # the start position of entity1
        e12_p = token.index("</e1>")  # the end position of entity1
        e21_p = token.index("<e2>")  # the start position of entity2
        e22_p = token.index("</e2>")  # the end position of entity2

        # s_start_p = token.index(sep_token) # the start position of sep token


        token[e11_p] = "$"
        token[e12_p] = "$"
        token[e21_p] = "#"
        token[e22_p] = "#"

        # extra_e11_p = token.index("<e1>")
        # extra_e12_p = token.index("</e1>")
        #
        # extra_e21_p = token.index("<e2>")
        # extra_e22_p = token.index("</e2>")
        #
        # token[extra_e11_p] = "$"
        # token[extra_e12_p] = "$"
        # token[extra_e21_p] = "#"
        # token[extra_e22_p] = "#"


        e11_p += 1
        e12_p += 1
        e21_p += 1
        e22_p += 1

        special_tokens_count = 1

        if len(token) > max_seq_len - special_tokens_count:
            token = token[: (max_seq_len - special_tokens_count)]

        if add_sep_token:
            token += [sep_token]

        token_type_ids = [sequence_a_segment_id] * len(token)

        token = [cls_token] + token
        token_type_ids = [cls_token_segment_id] + token_type_ids

        input_ids = tokenizer.convert_tokens_to_ids(token)

        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        padding_length = max_seq_len - len(input_ids)
        input_ids = input_ids + ([pad_token] * padding_length)
        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        e1_mask = [0] * len(attention_mask)
        e2_mask = [0] * len(attention_mask)
        # sep_mask = [0] * len(attention_mask)

        for i in range(e11_p, e12_p + 1):
            e1_mask[i] = 1

        for i in range(e21_p, e22_p + 1):
            e2_mask[i] = 1

        # for i in range(s_start_p, len(token)):
        #     sep_mask[i] = 1

        assert len(input_ids) == max_seq_len, "Error with input length {} vs {}".format(len(input_ids), max_seq_len)
        assert len(attention_mask) == max_seq_len, "Error with attention mask length {} vs {}".format(
            len(attention_mask), max_seq_len
        )
        assert len(token_type_ids) == max_seq_len, "Error with token type length {} vs {}".format(
            len(token_type_ids), max_seq_len
        )

        all_input_ids.append(input_ids)
        all_attention_mask.append(attention_mask)
        all_token_type_ids.append(token_type_ids)
        all_e1_mask.append(e1_mask)
        all_e2_mask.append(e2_mask)
        # all_sep_mask.append(sep_mask)

    all_features = {
        'input_ids': torch.tensor(all_input_ids),
        'attention_mask': torch.tensor(all_attention_mask),
        'token_type_ids': torch.tensor(all_token_type_ids),
        'e1_mask': torch.tensor(all_e1_mask),
        'e2_mask': torch.tensor(all_e2_mask),
        # 'sep_mask': torch.tensor(all_sep_mask)
    }
    train_label = train_dataset['label'].values
    print(f'average length : {float(sum(total_length))/float(len(total_length))}')
    print(f'max length : {max(total_length)}')
    return RE_Dataset(all_features, train_label)

def convert_to_QA(train_dataset, tokenizer, max_len):
    max_seq_len = max_len
    cls_token = tokenizer.cls_token
    # cls_token_segment_id=tokenizer.cls_token_id
    cls_token_segment_id = 0
    sep_token = tokenizer.sep_token
    pad_token = 1
    pad_token_segment_id = 0
    sequence_a_segment_id = 0
    add_sep_token = False
    mask_padding_with_zero = True

    all_input_ids = []
    all_attention_mask = []
    all_token_type_ids = []
    all_e1_mask = []
    all_e2_mask = []
    all_sep_mask = []
    total_length = []
    for idx in tqdm(range(len(train_dataset))):
        sorted_entity_index = sorted([train_dataset['e1s'][idx], train_dataset['e1e'][idx], train_dataset['e2s'][idx],
                                      train_dataset['e2e'][idx]])
        sentence = train_dataset['sentence'][idx]

        if train_dataset['e1s'][idx] > train_dataset['e2s'][idx]:
            train_dataset['sentence'][idx] = (sentence[:sorted_entity_index[0]] +
                                              ' <e2> ' +
                                              sentence[sorted_entity_index[0]: sorted_entity_index[1] + 1] +
                                              ' </e2> ' +
                                              sentence[sorted_entity_index[1] + 1: sorted_entity_index[2]] +
                                              ' <e1> ' +
                                              sentence[sorted_entity_index[2]: sorted_entity_index[3] + 1] +
                                              ' </e1> ' +
                                              sentence[sorted_entity_index[3] + 1:] +
                                              sep_token +
                                              ' <e2> ' +
                                              sentence[sorted_entity_index[0]: sorted_entity_index[1] + 1] +
                                              ' </e2> ' +
                                              '(와/과)' +
                                              ' <e1> ' +
                                              sentence[sorted_entity_index[2]: sorted_entity_index[3] + 1] +
                                              ' </e1> ' +
                                              '(은/는) 무슨 관계일까?' +
                                              sep_token
                                              )
        else:
            train_dataset['sentence'][idx] = (sentence[:sorted_entity_index[0]] +
                                              ' <e1> ' +
                                              sentence[sorted_entity_index[0]: sorted_entity_index[1] + 1] +
                                              ' </e1> ' +
                                              sentence[sorted_entity_index[1] + 1: sorted_entity_index[2]] +
                                              ' <e2> ' +
                                              sentence[sorted_entity_index[2]: sorted_entity_index[3] + 1] +
                                              ' </e2> ' +
                                              sentence[sorted_entity_index[3] + 1:] +
                                              sep_token +
                                              ' <e1> ' +
                                              sentence[sorted_entity_index[0]: sorted_entity_index[1] + 1] +
                                              ' </e1> ' +
                                              '(와/과)' +
                                              ' <e2> ' +
                                              sentence[sorted_entity_index[2]: sorted_entity_index[3] + 1] +
                                              ' </e2> ' +
                                              '(은/는) 무슨 관계일까?' +
                                              sep_token)

        token = tokenizer.tokenize(train_dataset['sentence'][idx])
        total_length.append(len(token))

        e11_p = token.index("<e1>")  # the start position of entity1
        e12_p = token.index("</e1>")  # the end position of entity1
        e21_p = token.index("<e2>")  # the start position of entity2
        e22_p = token.index("</e2>")  # the end position of entity2

        s_start_p = token.index(sep_token) # the start position of sep token


        token[e11_p] = "$"
        token[e12_p] = "$"
        token[e21_p] = "#"
        token[e22_p] = "#"

        extra_e11_p = token.index("<e1>")
        extra_e12_p = token.index("</e1>")

        extra_e21_p = token.index("<e2>")
        extra_e22_p = token.index("</e2>")

        token[extra_e11_p] = "$"
        token[extra_e12_p] = "$"
        token[extra_e21_p] = "#"
        token[extra_e22_p] = "#"


        e11_p += 1
        e12_p += 1
        e21_p += 1
        e22_p += 1

        special_tokens_count = 1

        if len(token) > max_seq_len - special_tokens_count:
            token = token[: (max_seq_len - special_tokens_count)]

        if add_sep_token:
            token += [sep_token]

        token_type_ids = [sequence_a_segment_id] * len(token)

        token = [cls_token] + token
        token_type_ids = [cls_token_segment_id] + token_type_ids

        input_ids = tokenizer.convert_tokens_to_ids(token)

        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        padding_length = max_seq_len - len(input_ids)
        input_ids = input_ids + ([pad_token] * padding_length)
        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        e1_mask = [0] * len(attention_mask)
        e2_mask = [0] * len(attention_mask)
        sep_mask = [0] * len(attention_mask)

        for i in range(e11_p, e12_p + 1):
            e1_mask[i] = 1

        for i in range(e21_p, e22_p + 1):
            e2_mask[i] = 1

        for i in range(s_start_p, len(token)):
            sep_mask[i] = 1

        assert len(input_ids) == max_seq_len, "Error with input length {} vs {}".format(len(input_ids), max_seq_len)
        assert len(attention_mask) == max_seq_len, "Error with attention mask length {} vs {}".format(
            len(attention_mask), max_seq_len
        )
        assert len(token_type_ids) == max_seq_len, "Error with token type length {} vs {}".format(
            len(token_type_ids), max_seq_len
        )

        all_input_ids.append(input_ids)
        all_attention_mask.append(attention_mask)
        all_token_type_ids.append(token_type_ids)
        all_e1_mask.append(e1_mask)
        all_e2_mask.append(e2_mask)
        all_sep_mask.append(sep_mask)

    all_features = {
        'input_ids': torch.tensor(all_input_ids),
        'attention_mask': torch.tensor(all_attention_mask),
        'token_type_ids': torch.tensor(all_token_type_ids),
        'e1_mask': torch.tensor(all_e1_mask),
        'e2_mask': torch.tensor(all_e2_mask),
        'sep_mask': torch.tensor(all_sep_mask)
    }
    train_label = train_dataset['label'].values
    print(f'average length : {float(sum(total_length))/float(len(total_length))}')
    print(f'max length : {max(total_length)}')
    return RE_Dataset(all_features, train_label)

