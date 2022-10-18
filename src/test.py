"""
Test PatentSBERTa on USPTO-2M dataset

"""
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import MultiLabelBinarizer
import pickle
from os.path import exists

train_dataset = pd.read_csv("../../data/USPTO-2M/train.tsv").dropna()
test_dataset = pd.read_csv("../../data/USPTO-2M/test.tsv").dropna()

train_dataset["group_ids"] = train_dataset["group_ids"].apply(lambda x:str(x).split(","))
test_dataset["group_ids"] = test_dataset["group_ids"].apply(lambda x:str(x).split(","))
train_dataset["id"] = train_dataset["id"].apply(lambda x:str(x).replace("US",""))
test_dataset["id"] = test_dataset["id"].apply(lambda x:str(x).replace("US",""))

for col in ["date", "text"]:
    train_dataset[col] = train_dataset[col].apply(str)
    test_dataset[col] = test_dataset[col].apply(str)

df_claim_cpc_train = train_dataset[["id", "date", "text"]]
df_claim_cpc_test = test_dataset[["id", "date", "text"]]

mlb = MultiLabelBinarizer()
mlb.fit(train_dataset['group_ids'])

train_labels = pd.DataFrame(mlb.transform(train_dataset['group_ids']), columns=mlb.classes_)
test_labels = pd.DataFrame(mlb.transform(test_dataset['group_ids']), columns=mlb.classes_)
assert len(train_labels.columns) == len(test_labels.columns)

df_claim_cpc_train = pd.concat([df_claim_cpc_train, train_labels], axis=1)
df_claim_cpc_test = pd.concat([df_claim_cpc_test, test_labels], axis=1)

# order pandas row by length of text (save more time for model.encode)
df_claim_cpc_train = df_claim_cpc_train.sort_values(by="text", key=lambda x: x.str.len()).dropna().reset_index(drop=True)
df_claim_cpc_test = df_claim_cpc_test.sort_values(by="text", key=lambda x: x.str.len()).dropna().reset_index(drop=True)

print(df_claim_cpc_test.head())
print(df_claim_cpc_test.shape)
print(df_claim_cpc_train.shape)

model = SentenceTransformer('AI-Growth-Lab/PatentSBERTa')

#Sentences are encoded by calling model.encode()
if exists('./uspto-2m/test_embeddings.pkl'):
    test_embeddings = pickle.load('./uspto-2m/test_embeddings.pkl', 'rb')
else:
    test_embeddings = model.encode(df_claim_cpc_test.text.values.tolist(), convert_to_tensor=True, show_progress_bar=True)
    with open('./uspto-2m/test_embeddings.pkl', 'wb') as outf1:
        pickle.dump(test_embeddings,outf1)

if exists('./uspto-2m/claim_embeddings.pkl'):
    claim_embeddings = pickle.load('./uspto-2m/claim_embeddings.pkl', 'rb')
else:
    claim_embeddings = model.encode(df_claim_cpc_train.text.values.tolist(), convert_to_tensor=True, show_progress_bar=True)
    with open('./uspto-2m/claim_embeddings.pkl', 'wb') as outf2:
        pickle.dump(claim_embeddings,outf2)

stored_patent_test_embeddings_id = df_claim_cpc_test['id']
stored_patent_train_embeddings_id = df_claim_cpc_train['id']

# get_ipython().system('pip install sentence_transformers')
import numpy as np
import torch
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from datetime import datetime
import torch
import torch.nn as nn

torch.manual_seed(1)
import pandas as pd
import torch
import time
from tqdm import tqdm, trange

start = time.time()

F1Measure_list = []
Recall_list = []
Accuracy_list = []
Precision_list = []
Hamming_Loss_list = []


def get_top_n_similar_patents_df(new_claim, claim_embeddings):
    # returns a list with one entry for each query. 
    # Each entry is a list of dictionaries with the keys 'corpus_id' and 'score', sorted by decreasing cosine similarity scores.
    search_hits = util.semantic_search(new_claim, claim_embeddings)#, 10000, 5000000, 20)
    
    top_claim_order = []
    top_claim_ids = []
    top_similarity_scores = []
    for item in range(len(search_hits[0])):
        top_claim_order = search_hits[0][item]['corpus_id']
        top_claim_ids.append(stored_patent_train_embeddings_id.iloc[top_claim_order])
        top_similarity_scores.append(search_hits[0][item].get('score'))
        
    top_100_similar_patents_df = pd.DataFrame({
        'top_claim_ids': top_claim_ids,
        'cosine_similarity': top_similarity_scores,
    })  
    return top_100_similar_patents_df

def F1Measure(y_true, y_pred):
    save_F1 = []
    temp = 0
    for i in range(y_true.shape[0]):
        if (sum(y_true[i]) == 0) and (sum(y_pred[i]) == 0):
            continue
        temp_save = (2*sum(np.logical_and(y_true[i], y_pred[i])))/ (sum(y_true[i])+sum(y_pred[i]))
        save_F1.append(temp_save)
        temp += temp_save

    save_F1 = pd.DataFrame(save_F1)
    save_F1_ids = pd.concat([result, save_F1], axis=1, ignore_index=True)
    return temp/ y_true.shape[0]

def Recall(y_true, y_pred):
    temp = 0
    for i in range(y_true.shape[0]):
        if sum(y_pred[i]) == 0:
            continue
        temp+= sum(np.logical_and(y_true[i], y_pred[i]))/ sum(y_pred[i])
    return temp/ y_true.shape[0]

def Precision(y_true, y_pred):
    temp = 0
    for i in range(y_true.shape[0]):
        if sum(y_true[i]) == 0:
            continue
        temp += sum(np.logical_and(y_true[i], y_pred[i]))/ sum(y_true[i])
    return temp/ y_true.shape[0]

def Hamming_Loss(y_true, y_pred):
    temp=0
    for i in range(y_true.shape[0]):
        temp += np.size(y_true[i] == y_pred[i]) - np.count_nonzero(y_true[i] == y_pred[i])
    return temp/(y_true.shape[0] * y_true.shape[1])

def Accuracy(y_true, y_pred):
    temp = 0
    for i in range(y_true.shape[0]):
        temp += sum(np.logical_and(y_true[i], y_pred[i])) / sum(np.logical_or(y_true[i], y_pred[i]))
    return temp / y_true.shape[0]

claims = list(df_claim_cpc_train.text)
patent_id = list(df_claim_cpc_train.id)

listofpredictdfs = []

start = time.time()

for i in trange(len(df_claim_cpc_test)):
    get_top_n_similar_patents_df_predict = get_top_n_similar_patents_df(np.array(test_embeddings[i].cpu()).reshape(1,-1), claim_embeddings)
    result = pd.merge(get_top_n_similar_patents_df_predict, df_claim_cpc_train, left_on='top_claim_ids', right_on='id', how='left', suffixes=('_left','_right'))
    locals()["predict_n"+str(i)] = result.copy()
    listofpredictdfs.append("predict_n"+str(i))

df = pd.concat(map(lambda x: eval(x), listofpredictdfs),keys= listofpredictdfs ,axis=0)

top_k = 10
for k in range(top_k):
    top_n = k
    predict = pd.DataFrame(columns= df_claim_cpc_test.columns[6:])
    for item in range(len(listofpredictdfs)):
        k_similar_patents = df.xs(listofpredictdfs[item]).nlargest(top_n, ['cosine_similarity'])
        result_k_similar_patents = pd.DataFrame(0, index=np.arange(1),columns= k_similar_patents.columns[8:])
        for i in range(top_n):
            result_k_similar_patents  = result_k_similar_patents + k_similar_patents.iloc[i, 8:].values
        result_k_similar_patents_df = pd.DataFrame(result_k_similar_patents, columns= k_similar_patents.columns[8:])
        result_k_similar_patents_df.insert(0, "input_patent_id", df_claim_cpc_test.id.iloc[item], True)
        locals()["predict"+str(item)] = result_k_similar_patents_df.copy()
        predict = pd.concat([predict, locals()["predict"+str(item)]], ignore_index=True)
        result_k_similar_patents_df = result_k_similar_patents_df[0:0]

    data = torch.tensor((predict.to_numpy()).astype(float), dtype=torch.float32)
    m = nn.Sigmoid()
    output = m(data)
    output = (output>0.9).float()
    output_df = pd.DataFrame(output, columns=predict.columns).astype(float) 
    y_pred = output_df.iloc[:, :-1].to_numpy()
    y_true = df_claim_cpc_test.iloc[:, 6:].to_numpy()
    result = pd.concat([output_df, df_claim_cpc_test], axis=1, ignore_index=True)    
    F1Measure_list.append(F1Measure(y_true,y_pred))
    Recall_list.append(Recall(y_true,y_pred))
    Accuracy_list.append(Accuracy(y_true, y_pred))
    Precision_list.append(Precision(y_true,y_pred))
    Hamming_Loss_list.append(Hamming_Loss(y_true, y_pred))
    end = time.time()
    print("==================================")
    print(f"Runtime of the program is {end - start}")
    print(f"K={k}")
    print("F1Measure: ", F1Measure_list[top_n])
    print("Recall: ", Recall_list[top_n])
    print("Accuracy: ", Accuracy_list[top_n])
    print("Precision: ", Precision_list[top_n])
    print("Hamming_Loss: ", Hamming_Loss_list[top_n])
    
output_d_metrics = {'F1Measure':F1Measure_list,'Recall_list':Recall_list, 'Accuracy_list':Accuracy_list,'Precision_list':Precision_list,'Hamming_Loss_list':Hamming_Loss_list}
output_df_metrics = pd.DataFrame(output_d_metrics)

