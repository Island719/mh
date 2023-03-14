# -*- coding: utf-8 -*-
import bunch
# torch==1.12.1
# transformers==2.24.0
# bunch
# sklearn

"""### 数据处理部分，从最原始文件夹集合到一个文件中。"""

# import os
import random
import numpy as np
from bunch import Bunch
import os

DEP = "./Identifying-depression-master/Data_Collector/mixed_depression"
NON_DEP = "./Identifying-depression-master/Data_Collector/mixed_non_depression"

def process_post(file_path):
    """
    Reads a file and extracts the raw text
    :param file_path: the path to the file
    :return: the processed string of raw text
    """
    # open and read the file
    post = open(file_path, 'rb')
    text = ''

    # read the post line by line
    for line in post:
        if line.strip():  # check if the line isn't empty
            # decode the line and append to the text string
            line = line.strip()
            text += line.decode('unicode-escape') + ' '

    return text

def construct_data(dep_fnames, non_dep_fnames, dep_dir, non_dep_dir):
    """
    Constructs the data bunch that contains file names, file paths, raw file texts, and targets of files
    :param dep_fnames: list of file names in depression directory
    :param non_dep_fnames: list of file names in non-depression directory
    :return: the constructed data bunch
    """
    # instantiate the data bunch
    data = Bunch()

    # join the 2 arrays of file names
    file_names = np.concatenate((dep_fnames, non_dep_fnames))

    # shuffle the data and add to the data bunch
    random.shuffle(file_names)

    # assign the shuffled file names array to data
    data.filenames = file_names

    # instantiate the lists for data attributes
    data.filepath = []  # path to files
    data.data = []  # raw texts
    data.target = []  # target category index

    # iterate the file names
    for index in range(len(file_names)):
        fn = file_names[index]

        # if the file belongs to depression cat
        if file_names[index] in dep_fnames:

            # append the corresponding index to the target attribute
            data.target.append(0)

            # find and append the path of the file to path attribute
            data.filepath.append(os.path.join(dep_dir, fn))

        # repeat for the other category
        else:
            data.target.append(1)
            data.filepath.append(os.path.join(non_dep_dir, fn))

        # get the path of the current file
        f_path = data.filepath[index]

        # read the file and pre-process the text
        post_text = process_post(f_path)

        # append it to the data attribute
        data.data.append(post_text)

    return data

print("\nProcessing data")

# lists of file names in both directories
dep_fnames = np.array(os.listdir(DEP))
non_dep_fnames = np.array(os.listdir(NON_DEP))

print("number of depression files: ", len(dep_fnames))
print("number of non-depression files: ", len(non_dep_fnames))

# Construct the data
data = construct_data(dep_fnames, non_dep_fnames, DEP, NON_DEP)

print("number of texts in data ", len(data.data))
print("targets for the first 10 files: ", data.target[:10])
print("number of targets of files in data", len(data.target))
print("first 2 files: ", data.data[:2])

"""### 开始构建模型部分，该部分基本上是所有数据集通用的。"""

### 设置不输出warning消息，与模型构建无关，只是不想输出太多包里的中间提示信息
import warnings
warnings.filterwarnings('ignore')

### 设置随机种子，防止模型随机初始化的不稳定性
import torch
def seed_everything(seed=7):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

### 定义一个数据处理的类
from torch.utils.data import Dataset, DataLoader
class BertDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length, is_testing=False):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_testing = is_testing

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        query = self.texts[idx]
        input_ids_query = self.tokenizer.encode(query)

        input_ids = input_ids_query[:-1]
        input_ids = input_ids[:self.max_length - 1] + tokenizer.convert_tokens_to_ids(['[SEP]'])
        attn_mask = [1] * len(input_ids)
        token_type_ids = [0] * len(input_ids)

        pad_len = self.max_length - len(input_ids)
        input_ids += [1] * pad_len
        attn_mask += [0] * pad_len
        token_type_ids += [0] * pad_len

        input_ids, attn_mask, token_type_ids = map(torch.LongTensor, [input_ids, attn_mask, token_type_ids])

        encoded_dict = {
            'input_ids': input_ids,
            'attn_mask': attn_mask,
            'token_type_ids': token_type_ids,
        }

        if not self.is_testing:
            label = self.labels[idx]
            encoded_dict['label'] = torch.tensor(label, dtype=torch.long)
        return encoded_dict

### 定义模型
from transformers import AutoTokenizer, AutoModel, AutoConfig
from torch import nn, optim
class DepressionBertModel(nn.Module):
    def __init__(self, bert_path):
        super().__init__()

        bert_config = AutoConfig.from_pretrained(bert_path)
        self.bert = AutoModel.from_pretrained(bert_path, config=bert_config)
        self.dropout = nn.Dropout(0.3) ### dropout 参数
        self.classifier = nn.Linear(bert_config.hidden_size, 2) ### label_num

    def forward(self, input_ids, attn_mask, token_type_ids):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attn_mask,
            token_type_ids=token_type_ids)

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits, pooled_output

def dev_eval(val_iter, model, device):
    model.eval()
    logits_list=[]
    label_list=[]
    avg_loss=0
    for batch in val_iter:
        input_ids, attn_mask, token_type_ids, label = batch['input_ids'], batch['attn_mask'], batch['token_type_ids'], batch['label']
        input_ids, attn_mask, token_type_ids, label = input_ids.to(device), attn_mask.to(device), token_type_ids.to(device), label.to(device)
        with torch.no_grad():
            logits, pooled_ = model(input_ids, attn_mask, token_type_ids)
        loss = nn.CrossEntropyLoss()
        loss=loss(logits,label)
        avg_loss += loss.item()
        logits = torch.max(logits.data, 1)[1].cpu()
        logits.tolist()
        label = label.cpu().tolist()
        logits_list.extend(logits)
        label_list.extend(label)
    f1 = f1_score(logits_list, label_list, average='macro')
    model.train()
    return f1

SAVE_MODEL_PATH = 'test' # 模型存储路径
SEED = 0 # 随即种子
ALL_LABELS = ['depression', 'not depression'] # 定义类别
BERT_PATH = 'mental/mental-bert-base-uncased' # BERT模型的路径
SPLIT = 0.8 # 训练集和验证集划分
EPOCH = 10 # 训练epoch
MAX_LENGTH = 128 # 模型可处理的最长样本长度
BATCH_SIZE = 32 # batch_size
SETP_SHOW = 100 # 每多少个step输出一次结果
LEARNING_RATE = 1e-5 # 学习率
WARMUP_STEPS = 0.1 # warmup比率

import os
from transformers import AutoTokenizer, AutoModel
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import f1_score,accuracy_score,hamming_loss, balanced_accuracy_score

if not os.path.exists(SAVE_MODEL_PATH):
    os.mkdir(SAVE_MODEL_PATH)

seed_everything(SEED)

tokenizer = AutoTokenizer.from_pretrained(BERT_PATH)
model = AutoModel.from_pretrained(BERT_PATH)

train_texts = data.data[:int(len(data.data) * SPLIT)]
train_labels = data.target[:int(len(data.target) * SPLIT)]

dev_texts = data.data[int(len(data.data) * SPLIT):]
dev_labels = data.target[int(len(data.target) * SPLIT):]

train_ds = BertDataset(train_texts, train_labels, tokenizer, MAX_LENGTH, is_testing=False)
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
train_steps = (len(train_dl) * EPOCH)
valid_ds = BertDataset(dev_texts, dev_labels, tokenizer, MAX_LENGTH, is_testing=False)
valid_dl = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DepressionBertModel(BERT_PATH)
# print(model)
model.to(DEVICE)
### 训练模型
param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.weight']
optimizer_parameters = [
    {
        'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
        'weight_decay': 1e-3
    },
    {
        'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
        'weight_decay': 0.
    }
]

optimizer = optim.AdamW(optimizer_parameters, lr=LEARNING_RATE)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(WARMUP_STEPS * train_steps),
    num_training_steps=train_steps
)

steps = 0
best_f1 = 0
model.train()
for epoch in range(EPOCH):
    for batch in train_dl:
        input_ids, attn_mask, token_type_ids, label = batch['input_ids'], batch['attn_mask'], batch['token_type_ids'], batch['label']
        input_ids, attn_mask, token_type_ids, label = input_ids.to(DEVICE), attn_mask.to(DEVICE), token_type_ids.to(DEVICE), label.to(DEVICE)
        logits, pooled_output= model(input_ids, attn_mask, token_type_ids)
        loss = nn.CrossEntropyLoss()
        loss = loss(logits,label)
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        steps += 1
        logits = torch.max(logits.data, 1)[1].cpu()
        label = label.cpu()
        if steps % SETP_SHOW == 0:
            f1 = f1_score(logits, label, average='macro')
            print('epoch:%d\t\t\tsteps:%d\t\t\tloss:%.6f\t\t\tf1_score:%.4f'%(epoch, steps, loss.item(), f1))
    
    dev_f1 = dev_eval(valid_dl, model, DEVICE)
    print('dev\nf1:%.6f' % (dev_f1))
    if dev_f1 > best_f1:
        best_f1 = dev_f1
        torch.save(model, SAVE_MODEL_PATH + '/best.pth')
        print('save best model\t\tf1:%.6f'%best_f1)

### 加载训练好的模型
model = torch.load(SAVE_MODEL_PATH + '/best.pth')
print(dev_eval(valid_dl, model))