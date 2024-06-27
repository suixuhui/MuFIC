
import numpy as np
import pickle

from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
import itertools
import scipy.sparse as sp


def totol_cluster2pair(cluster_list):
    seed_pair_list, id_list = [], []
    for i in range(len(cluster_list)):
        id = cluster_list[i]
        if id not in id_list:
            id_list.append(id)
            index_list = [i for i, x in enumerate(cluster_list) if x == id]
            if len(index_list) > 1:
                iter_list = list(itertools.combinations(index_list, 2))
                seed_pair_list += iter_list
    return seed_pair_list


def seed_pair2cluster(seed_pair_list, ent_list):
    pair_dict = dict()
    for i in range(len(ent_list)):
        pair_dict[i] = [i]
    for seed_pair in seed_pair_list:
        a, b = seed_pair
        for j in pair_dict[a]:
            for k in pair_dict[b]:
                if j not in pair_dict[k]:
                    pair_dict[k].append(j)
        for j in pair_dict[b]:
            for k in pair_dict[a]:
                if j not in pair_dict[k]:
                    pair_dict[k].append(j)
    cluster_list = []
    for i in range(len(ent_list)):
        cluster_list.append(i)
    for ent_id in pair_dict:
        rep = pair_dict[ent_id]
        cluster_list[ent_id] = rep
    return cluster_list


def load_data(dataset):
    ent2id_fname = './file/' + dataset + '/ent2id'
    rel2id_fname = './file/' + dataset + '/rel2id'
    ent2id, rel2id = pickle.load(open(ent2id_fname, 'rb')), pickle.load(open(rel2id_fname, 'rb'))

    triples_id_fname = './file/' + dataset + '/triples_id'
    triples_id = pickle.load(open(triples_id_fname, 'rb'))

    EL_seed_fname = './file/' + dataset + '/EL_seed'
    EL_seed = pickle.load(open(EL_seed_fname, 'rb'))

    WEB_Entity_seed_fname = './file/' + dataset + '/WEB_seed/entity/cluster_list_threshold_0.015_url_max_length_all'
    WEB_Entity_seed = pickle.load(open(WEB_Entity_seed_fname, 'rb'))
    WEB_Entity_seed = totol_cluster2pair(WEB_Entity_seed)

    RP_seed_fname = './file/' + dataset + '/amie_rp_seed'
    RP_seed = pickle.load(open(RP_seed_fname, 'rb'))

    WEB_relation_seed_fname = './file/' + dataset + '/WEB_seed/relation/cluster_list_threshold_0.015_url_max_length_all'
    WEB_relation_seed = pickle.load(open(WEB_relation_seed_fname, 'rb'))
    WEB_relation_seed = totol_cluster2pair(WEB_relation_seed)

    all_entity_seed_pair_list = []
    for pair in WEB_Entity_seed:
        if pair not in all_entity_seed_pair_list:
            all_entity_seed_pair_list.append(pair)
    for pair in EL_seed:
        if pair not in all_entity_seed_pair_list:
            all_entity_seed_pair_list.append(pair)
    all_relation_seed_pair_list = []
    for pair in WEB_relation_seed:
        if pair not in all_relation_seed_pair_list:
            all_relation_seed_pair_list.append(pair)
    for pair in RP_seed:
        if pair not in all_relation_seed_pair_list:
            all_relation_seed_pair_list.append(pair)

    ent_list_fname = './file/' + dataset + '/ent_list'
    ent_list = pickle.load(open(ent_list_fname, 'rb'))
    entity_cluster_list = seed_pair2cluster(all_entity_seed_pair_list, ent_list)

    rel_list_fname = './file/' + dataset + '/rel_list'
    rel_list = pickle.load(open(rel_list_fname, 'rb'))
    relation_cluster_list = seed_pair2cluster(all_relation_seed_pair_list, rel_list)

    return triples_id, entity_cluster_list, relation_cluster_list


def inverse_sum(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -1).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    return d_inv_sqrt.reshape((-1, 1))


def HAC_getClusters(params, embed, cluster_threshold_real, embed_dim):
    dist = pdist(embed, metric=params.metric)
    if params.dataset == 'reverb45k_change':
        if not np.all(np.isfinite(dist)):
            for i in range(len(dist)):
                if not np.isfinite(dist[i]):
                    dist[i] = 0
    clust_res = linkage(dist, method=params.linkage)
    labels = fcluster(clust_res, t=cluster_threshold_real, criterion='maxclust') - 1
    # maxclust distance

    clusters = [[] for i in range(max(labels) + 1)]
    for i in range(len(labels)):
        clusters[labels[i]].append(i)

    clusters_center = np.zeros((len(clusters), embed_dim), np.float32)
    for i in range(len(clusters)):
        cluster = clusters[i]
        # if  ave:
        clusters_center_embed = np.zeros(embed_dim, np.float32)
        for j in cluster:
            embed_ = embed[j]
            clusters_center_embed += embed_
        clusters_center_embed_ = clusters_center_embed / len(cluster)
        clusters_center[i, :] = clusters_center_embed_
    return labels, clusters_center


def cluster_test(params, cluster_predict_list):
    clust2ent = {}
    sub_cluster_predict_list = []
    true_ent2cluster_fname = './file/' + params.dataset + '/true_ent2cluster'
    true_ent2clust = pickle.load(open(true_ent2cluster_fname, 'rb'))
    true_clust2ent = invertDic(true_ent2clust)

    issub_fname = './file/' + params.dataset + '/issub'
    issub = pickle.load(open(issub_fname, 'rb'))

    triples_fname = './file/' + params.dataset + '/triples'
    triples = pickle.load(open(triples_fname, 'rb'))

    ent2id_fname = './file/' + params.dataset + '/ent2id'
    ent2id = pickle.load(open(ent2id_fname, 'rb'))

    for eid in issub:
        sub_cluster_predict_list.append(cluster_predict_list[eid])

    for sub_id, cluster_id in enumerate(sub_cluster_predict_list):
        if cluster_id in clust2ent.keys():
            clust2ent[cluster_id].append(sub_id)
        else:
            clust2ent[cluster_id] = [sub_id]
    cesi_clust2ent = {}
    for rep, cluster in clust2ent.items():
        # cesi_clust2ent[rep] = list(cluster)
        cesi_clust2ent[rep] = set(cluster)
    cesi_ent2clust = invertDic(cesi_clust2ent)

    cesi_ent2clust_u = {}
    for trp in triples:
        sub_u, sub = trp['triple_unique'][0], trp['triple'][0]
        cesi_ent2clust_u[sub_u] = cesi_ent2clust[ent2id[sub]]
    cesi_clust2ent_u = invertDic(cesi_ent2clust_u)

    eval_results = evaluate(cesi_ent2clust_u, cesi_clust2ent_u, true_ent2clust, true_clust2ent)
    macro_prec, micro_prec, pair_prec = eval_results['macro_prec'], eval_results['micro_prec'], eval_results['pair_prec']
    macro_recall, micro_recall, pair_recall = eval_results['macro_recall'], eval_results['micro_recall'], eval_results['pair_recall']
    macro_f1, micro_f1, pair_f1 = eval_results['macro_f1'], eval_results['micro_f1'], eval_results['pair_f1']
    ave_prec = (macro_prec + micro_prec + pair_prec) / 3
    ave_recall = (macro_recall + micro_recall + pair_recall) / 3
    ave_f1 = (macro_f1 + micro_f1 + pair_f1) / 3
    model_clusters = len(cesi_clust2ent_u)
    model_Singletons = len([1 for _, entity in cesi_clust2ent_u.items() if len(entity) == 1])
    gold_clusters = len(true_clust2ent)
    gold_Singletons = len([1 for _, entity in true_clust2ent.items() if len(entity) == 1])

    return ave_prec, ave_recall, ave_f1, macro_prec, micro_prec, pair_prec, macro_recall, micro_recall, pair_recall, \
           macro_f1, micro_f1, pair_f1, model_clusters, model_Singletons, gold_clusters, gold_Singletons


def invertDic(my_map):
    inv_map = {}
    for k, v in my_map.items():
        for ele in v:
            inv_map[ele] = inv_map.get(ele, set())
            inv_map[ele].add(k)
    return inv_map


def macroPrecision(C_clust2ele, E_ele2clust):
    num_prec = 0

    for _, cluster in C_clust2ele.items():
        isFirst = True
        res = set()
        for ele in cluster:
            if ele not in E_ele2clust:
                # sys.stdout.write('.')
                continue
            if isFirst:
                res = E_ele2clust[ele]
                isFirst = False
                continue

            res = res.intersection(E_ele2clust[ele])

        if len(res) == 1:
            num_prec += 1
        # else:print('res:', len(res), res)
        # elif len(res) > 1:
        #     print('ERROR In Clustering micro!!!')

    if len(C_clust2ele) == 0: return 0
    return float(num_prec) / float(len(C_clust2ele))

def microPrecision(C_clust2ele, E_ele2clust):
    num_prec = 0
    total = 0

    for _, cluster in C_clust2ele.items():
        freq_map = {}
        total += len(cluster)

        for ent in cluster:
            if ent not in E_ele2clust:
                # sys.stdout.write('.')
                continue
            for ele in E_ele2clust[ent]:
                freq_map[ele] = freq_map.get(ele, 0)
                freq_map[ele] += 1
        max_rep = 0
        for k, v in freq_map.items(): max_rep = max(max_rep, v)

        num_prec += max_rep

    if total == 0: return 0
    return float(num_prec) / float(total)

def pairPrecision(C_clust2ele, E_ele2clust):
    num_hit = 0
    num_pairs = 0

    for _, cluster in C_clust2ele.items():
        all_pairs = list(itertools.combinations(cluster, 2))
        num_pairs += len(all_pairs)

        for e1, e2 in all_pairs:
            if e1 not in E_ele2clust or e2 not in E_ele2clust:
                # sys.stdout.write('.')
                continue

            res = E_ele2clust[e1].intersection(E_ele2clust[e2])
            if len(res) == 1: num_hit += 1
        # elif len(res) > 1: print( 'ERROR In Clustering pairwise!!!')

    if num_pairs == 0: return 0
    return float(num_hit) / float(num_pairs)

def pairwiseMetric(C_clust2ele, E_ele2clust, E_clust2ent):
    num_hit = 0
    num_C_pairs = 0
    num_E_pairs = 0

    for _, cluster in C_clust2ele.items():
        all_pairs = list(itertools.combinations(cluster, 2))
        num_C_pairs += len(all_pairs)

        for e1, e2 in all_pairs:
            if e1 in E_ele2clust and e2 in E_ele2clust and len(
                E_ele2clust[e1].intersection(E_ele2clust[e2])) > 0: num_hit += 1

    for rep, cluster in E_clust2ent.items():
        num_E_pairs += len(list(itertools.combinations(cluster, 2)))

    if num_C_pairs == 0 or num_E_pairs == 0:
        return 1e-6, 1e-6

    # print( num_hit, num_C_pairs, num_E_pairs)
    return float(num_hit) / float(num_C_pairs), float(num_hit) / float(num_E_pairs)


def calcF1(prec, recall):
    if prec + recall == 0: return 0
    return 2 * (prec * recall) / (prec + recall)

def microF1(C_ele2clust, C_clust2ele, E_ele2clust, E_clust2ent):
    micro_prec = microPrecision(C_clust2ele, E_ele2clust)
    micro_recall = microPrecision(E_clust2ent, C_ele2clust)
    micro_f1 = calcF1(micro_prec, micro_recall)
    return micro_f1

def macroF1(C_ele2clust, C_clust2ele, E_ele2clust, E_clust2ent):
    macro_prec = macroPrecision(C_clust2ele, E_ele2clust)
    macro_recall = macroPrecision(E_clust2ent, C_ele2clust)
    macro_f1 = calcF1(macro_prec, macro_recall)
    return macro_f1

def pairF1(C_ele2clust, C_clust2ele, E_ele2clust, E_clust2ent):
    pair_prec, pair_recall = pairwiseMetric(C_clust2ele, E_ele2clust, E_clust2ent)
    pair_f1 = calcF1(pair_prec, pair_recall)
    return pair_f1

def evaluate(C_ele2clust, C_clust2ele, E_ele2clust, E_clust2ent):
    macro_prec = macroPrecision(C_clust2ele, E_ele2clust)
    macro_recall = macroPrecision(E_clust2ent, C_ele2clust)
    macro_f1 = calcF1(macro_prec, macro_recall)

    micro_prec = microPrecision(C_clust2ele, E_ele2clust)
    micro_recall = microPrecision(E_clust2ent, C_ele2clust)
    micro_f1 = calcF1(micro_prec, micro_recall)

    pair_prec, pair_recall = pairwiseMetric(C_clust2ele, E_ele2clust, E_clust2ent)
    pair_f1 = calcF1(pair_prec, pair_recall)

    return {
        'macro_prec': round(macro_prec, 4),
        'macro_recall': round(macro_recall, 4),
        'macro_f1': round(macro_f1, 4),

        'micro_prec': round(micro_prec, 4),
        'micro_recall': round(micro_recall, 4),
        'micro_f1': round(micro_f1, 4),

        'pair_prec': round(pair_prec, 4),
        'pair_recall': round(pair_recall, 4),
        'pair_f1': round(pair_f1, 4),
    }
