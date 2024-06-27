
import codecs
import json
import pickle
import pathlib
import gensim
from nltk.tokenize import word_tokenize
import numpy as np
from collections import defaultdict as ddict
from pytorch_transformers.tokenization_bert import BertTokenizer
from pytorch_transformers.modeling_bert import (
    BertPreTrainedModel,
    BertConfig,
    BertModel,
)
import torch

def getEmbeddings(model, phr_list, embed_dims):
    embed_list = []
    all_num, oov_num, oov_rate = 0, 0, 0
    for phr in phr_list:
        if phr in model.vocab:
            embed_list.append(model.word_vec(phr))
            all_num += 1
        else:
            vec = np.zeros(embed_dims, np.float32)
            wrds = word_tokenize(phr)
            for wrd in wrds:
                all_num += 1
                if wrd in model.vocab:
                    vec += model.word_vec(wrd)
                else:
                    vec += np.random.randn(embed_dims)
                    oov_num += 1
            if len(wrds) == 0:
                embed_list.append(vec / 10000)
            else:
                embed_list.append(vec / len(wrds))
    oov_rate = oov_num / all_num
    print('oov rate:', oov_rate, 'oov num:', oov_num, 'all num:', all_num)
    return np.array(embed_list)

def pad_sequence(x, max_len, type=np.int):
    padded_x = np.zeros((len(x), max_len), dtype=type)
    for i, row in enumerate(x):
        padded_x[i][:len(row)] = row
    padded_x = torch.LongTensor(padded_x).cuda(7)

    return padded_x

def get_initial_embedding(dataset, triples_list):
    sub_list = []
    rel_list = []
    obj_list = []
    ent_list = []
    ent_fname = './file/' + dataset + '/ent_list'
    rel_fname = './file/' + dataset + '/rel_list'
    if not pathlib.Path(ent_fname).is_file() or not pathlib.Path(rel_fname).is_file():
        for trp in triples_list:
            sub, rel, obj = map(str, trp['triple'])
            if sub not in sub_list:
                sub_list.append(sub)
            if rel not in rel_list:
                rel_list.append(rel)
            if obj not in obj_list:
                obj_list.append(obj)
        for ent in sub_list:
            ent_list.append(ent)
        for ent in obj_list:
            if ent in sub_list:
                continue
            ent_list.append(ent)
        pickle.dump(rel_list, open(rel_fname, 'wb'))
        pickle.dump(ent_list, open(ent_fname, 'wb'))
    else:
        ent_list = pickle.load(open(ent_fname, 'rb'))
        rel_list = pickle.load(open(rel_fname, 'rb'))

    ent2id_fname = './file/' + dataset + '/ent2id'
    rel2id_fname = './file/' + dataset + '/rel2id'
    if not pathlib.Path(ent2id_fname).is_file() or not pathlib.Path(rel2id_fname).is_file():
        ent2id = dict([(v, k) for k, v in enumerate(ent_list)])
        rel2id = dict([(v, k) for k, v in enumerate(rel_list)])
        pickle.dump(ent2id, open(ent2id_fname, 'wb'))
        pickle.dump(rel2id, open(rel2id_fname, 'wb'))
    else:
        ent2id = pickle.load(open(ent2id_fname, 'rb'))
        rel2id = pickle.load(open(rel2id_fname, 'rb'))

    issub_fname = './file/' + dataset + '/issub'
    if not pathlib.Path(issub_fname).is_file():
        subs = []
        for trp in triples_list:
            sub, rel, obj = map(str, trp['triple'])
            sub_id = ent2id[sub]
            if sub_id not in subs:
                subs.append(sub_id)
        pickle.dump(subs, open(issub_fname, 'wb'))
    else:
        subs = pickle.load(open(issub_fname, 'rb'))

    triples_id_fname = './file/' + dataset + '/triples_id'
    triples = []
    if not pathlib.Path(triples_id_fname).is_file():
        for trp in triples_list:
            sub, rel, obj = map(str, trp['triple'])
            trp = (ent2id[sub], rel2id[rel], ent2id[obj])
            triples.append(trp)
        pickle.dump(triples, open(triples_id_fname, 'wb'))


    E_fname = './file/' + dataset + '/E_init'
    R_fname = './file/' + dataset + '/R_init'
    embed_loc_fname = './file/crawl-300d-2M.vec'
    if not pathlib.Path(E_fname).is_file() or not pathlib.Path(R_fname).is_file():
        model = gensim.models.KeyedVectors.load_word2vec_format(embed_loc_fname, binary=False)
        E_init = getEmbeddings(model, ent_list, 300)
        R_init = getEmbeddings(model, rel_list, 300)

        pickle.dump(E_init, open(E_fname, 'wb'))
        pickle.dump(R_init, open(R_fname, 'wb'))

    true_ent2cluster_fname = './file/' + dataset + '/true_ent2cluster'
    true_ent2clust = ddict(set)
    if not pathlib.Path(true_ent2cluster_fname).is_file():
        true_mark = None
        if dataset == "reverb45k_change":
            true_mark = 'true_sub_link'
        if dataset == "OPIEC59K":
            true_mark = 'subject_wiki_link'
        for trp in triples_list:
            sub = trp['triple_unique'][0]
            true_ent2clust[sub].add(trp[true_mark])
        pickle.dump(true_ent2clust, open(true_ent2cluster_fname, 'wb'))
    else:
        true_ent2clust = pickle.load(open(true_ent2cluster_fname, 'rb'))

    triples_fname = './file/' + dataset + '/triples'
    if not pathlib.Path(triples_fname).is_file():
        pickle.dump(triples_list, open(triples_fname, 'wb'))

    sentences_frame = './file/' + dataset + '/sentences_all_entities'
    if not pathlib.Path(sentences_frame).is_file():
        max_length = 128
        sentences = []
        for _ in range(len(ent_list)):
            sentences.append("")
        for trp in triples_list:
            sentences_ = trp["src_sentences"]
            sub, rel, obj = map(str, trp['triple'])
            for sentence in sentences_:
                if (len(sentences[ent2id[sub]]) == 0 or len(sentence.split(" ")) > len(
                        sentences[ent2id[sub]].split(" "))) and len(sentence.split(" ")) < max_length:
                    sub_sentence = sentence.replace(sub, "[unused0]")
                    sentences[ent2id[sub]] = sub_sentence
                if (len(sentences[ent2id[obj]]) == 0 or len(sentence.split(" ")) > len(sentences[ent2id[obj]].split(" "))) and len(sentence.split(" ")) < max_length:
                    obj_sentence = sentence.replace(obj, "[unused0]")
                    sentences[ent2id[obj]] = obj_sentence

        pickle.dump(sentences, open(sentences_frame, 'wb'))
    else:
        sentences = pickle.load(open(sentences_frame, 'rb'))

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
    special_tokens_dict = {
        "additional_special_tokens": [
            "[unused0]",
        ],
    }
    tokenizer.add_special_tokens(special_tokens_dict)
    context_input_ids_fname = './file/' + dataset + '/context_input_ids_remove_nopadding_all_entities'
    if not pathlib.Path(context_input_ids_fname).is_file():
        max_length = 128
        total_input_ids, total_masks, total_segments = [], [], []
        for i in range(len(sentences)):
            sentence = sentences[i]
            wps = tokenizer.tokenize(sentence)
            max_len = max_length - 2
            if len(wps) > max_len:
                wps = wps[:max_len]
            tokens_id = tokenizer.convert_tokens_to_ids(wps)
            total_input_ids.append(tokens_id)
        pickle.dump(total_input_ids, open(context_input_ids_fname, 'wb'))
    else:
        total_input_ids = pickle.load(open(context_input_ids_fname, 'rb'))
    print(len(total_input_ids))

    context_embedding_fname = './file/' + dataset + '/context_embedding_all_entities'
    if not pathlib.Path(context_embedding_fname).is_file():
        bert = BertModel.from_pretrained("bert-base-uncased")
        for param in bert.parameters():
            param.requires_grad = False
        context_embedding = None
        for i in range(len(total_input_ids)):
            input_id = total_input_ids[i]
            input_id = torch.LongTensor(input_id)
            input_id = torch.unsqueeze(input_id, dim=0)
            with torch.no_grad():
                output_bert, output_pooler = bert(input_id)

            embedding = None
            input_id = torch.squeeze(input_id)
            for i in range(len(input_id)):
                if input_id[i] == 1:
                    embedding = output_bert[0][i]
                    break

            if embedding == None:
                embedding = output_bert[0][0]

            if context_embedding == None:
                context_embedding = embedding
            else:
                context_embedding = torch.cat((context_embedding, embedding))
        context_embedding = context_embedding.view(-1, 768)

        pickle.dump(context_embedding, open(context_embedding_fname, 'wb'))



dataset = "OPIEC59K"
triples_list = []
if dataset == 'reverb45k_change':
    fname = './data/' + dataset + '/reverb45k_change_test_read'
    with codecs.open(fname, encoding='utf-8', errors='ignore') as f:
        for line in f:
            trp = json.loads(line.strip())
            triples_list.append(trp)
if dataset == 'OPIEC59K':
    fname = './data/' + dataset + '/OPIEC59k_test'
    triples_list = pickle.load(open(fname, 'rb'))

get_initial_embedding(dataset, triples_list)
