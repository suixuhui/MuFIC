import random
import sys
from utils import load_data, HAC_getClusters, cluster_test
from models import pick_model
from torch.utils.data import DataLoader
import copy
from torch.optim import Adam
import pickle
from dataloader import *

if __name__ == "__main__":


    if args.random_seed != 0:
        random.seed(args.random_seed)
        np.random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)

    print(args)
    E_fname = args.FILE_DIR + '/' + args.dataset + '/E_init'
    R_fname = args.FILE_DIR + '/' + args.dataset + '/R_init'
    E_init = pickle.load(open(E_fname, 'rb'))
    R_init = pickle.load(open(R_fname, 'rb'))

    entity_context_tokens_id_fname = './file/' + args.dataset + '/context_input_ids_remove_nopadding_all_entities'
    entity_context_embedding_fname = './file/' + args.dataset + '/context_embedding_all_entities'
    entity_context_tokens_id = pickle.load(open(entity_context_tokens_id_fname, 'rb'))
    entity_context_embedding = pickle.load(open(entity_context_embedding_fname, 'rb'))

    issub_fname = './file/' + args.dataset + '/issub'
    issub = pickle.load(open(issub_fname, 'rb'))

    triples_id, entity_cluster_list, relation_cluster_list = load_data(args.dataset)
    origin_entity_cluster_list = copy.deepcopy(entity_cluster_list)

    nentity = len(E_init)
    nrelation = len(R_init)

    train_dataloader_head = DataLoader(
        TrainDataset(triples_id, nentity, nrelation, args.negative_number, 'head-batch'),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=TrainDataset.collate_fn
    )

    train_dataloader_tail = DataLoader(
        TrainDataset(triples_id, nentity, nrelation, args.negative_number, 'tail-batch'),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=TrainDataset.collate_fn
    )

    train_iterator = BidirectionalOneShotIterator(train_dataloader_head, train_dataloader_tail)

    model = pick_model(args, E_init, R_init, entity_context_tokens_id, entity_context_embedding)

    test_only = args.test_model is not None

    bert_params = list(map(id, model.bert.parameters()))
    other_params = filter(lambda p: id(p) not in bert_params, model.parameters())
    optimizer_grouped_parameters = [
        {"params": other_params, "lr": args.lr},
        {"params": model.bert.parameters(), "lr": args.bert_lr},
    ]

    optimizer = Adam(
        optimizer_grouped_parameters
    )
    model.train()

    for step in range(args.max_steps):

        positive_sample, negative_sample, mode, idxs = next(train_iterator)
        positive_sample = positive_sample.cuda(args.gpu)
        negative_sample = negative_sample.cuda(args.gpu)
        loss, outputs = model(positive_sample, negative_sample, mode, entity_cluster_list, relation_cluster_list, "train")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 200 == 0:
            print("step: ", step)
            print("loss: ", loss)
            with torch.no_grad():
                loss, structure_outputs, context_outputs, outputs = model(positive_sample, negative_sample, mode, entity_cluster_list,
                                      relation_cluster_list, "test")

            # threshold_or_cluster = 'threshold'
            threshold_or_cluster = 'cluster'
            if threshold_or_cluster == 'threshold':
                if args.dataset == 'OPIEC59K':
                    cluster_threshold_real = 0.29
                else:
                    cluster_threshold_real = 0.35
            else:
                if args.dataset == 'OPIEC59K':
                    cluster_threshold_real = 490
                else:
                    cluster_threshold_real = 6700

            if args.dataset == 'OPIEC59K':
                margin_step = 20000
            else:
                margin_step = 20000

            structure_ent_embedding = structure_outputs
            structure_ent_embedding = structure_ent_embedding.cpu().detach().numpy()
            structure_embeddings = []
            for id in issub:
                structure_embeddings.append(structure_ent_embedding[id])
            structure_labels, clusters_center = HAC_getClusters(args, structure_embeddings, cluster_threshold_real, args.embed_size)
            structure_cluster_predict_list = list(structure_labels)

            ave_prec, ave_recall, ave_f1, macro_prec, micro_prec, pair_prec, macro_recall, micro_recall, \
            pair_recall, macro_f1, micro_f1, pair_f1, model_clusters, model_Singletons, gold_clusters, gold_Singletons \
                = cluster_test(args, structure_cluster_predict_list)
            print('Ave-prec=', ave_prec, 'macro_prec=', macro_prec, 'micro_prec=', micro_prec,
                  'pair_prec=', pair_prec)
            print('Ave-recall=', ave_recall, 'macro_recall=', macro_recall, 'micro_recall=', micro_recall,
                  'pair_recall=', pair_recall)
            print('Ave-F1=', ave_f1, 'macro_f1=', macro_f1, 'micro_f1=', micro_f1, 'pair_f1=', pair_f1)
            print('Model: #Clusters: %d, #Singletons %d' % (model_clusters, model_Singletons))
            print('Gold: #Clusters: %d, #Singletons %d' % (gold_clusters, gold_Singletons))
            print()

            context_ent_embedding = context_outputs
            context_ent_embedding = context_ent_embedding.cpu().detach().numpy()
            context_embeddings = []
            for id in issub:
                context_embeddings.append(context_ent_embedding[id])
            context_labels, clusters_center = HAC_getClusters(args, context_embeddings, cluster_threshold_real, args.embed_size)
            context_cluster_predict_list = list(context_labels)

            ave_prec, ave_recall, ave_f1, macro_prec, micro_prec, pair_prec, macro_recall, micro_recall, \
            pair_recall, macro_f1, micro_f1, pair_f1, model_clusters, model_Singletons, gold_clusters, gold_Singletons \
                = cluster_test(args, context_cluster_predict_list)
            print('Ave-prec=', ave_prec, 'macro_prec=', macro_prec, 'micro_prec=', micro_prec,
                  'pair_prec=', pair_prec)
            print('Ave-recall=', ave_recall, 'macro_recall=', macro_recall, 'micro_recall=', micro_recall,
                  'pair_recall=', pair_recall)
            print('Ave-F1=', ave_f1, 'macro_f1=', macro_f1, 'micro_f1=', micro_f1, 'pair_f1=', pair_f1)
            print('Model: #Clusters: %d, #Singletons %d' % (model_clusters, model_Singletons))
            print('Gold: #Clusters: %d, #Singletons %d' % (gold_clusters, gold_Singletons))
            print()

            ent_embedding = outputs
            ent_embedding = ent_embedding.cpu().detach().numpy()
            embeddings = []
            for id in issub:
                embeddings.append(ent_embedding[id])
            labels, clusters_center = HAC_getClusters(args, embeddings, cluster_threshold_real, args.embed_size)
            cluster_predict_list = list(labels)

            ave_prec, ave_recall, ave_f1, macro_prec, micro_prec, pair_prec, macro_recall, micro_recall, \
            pair_recall, macro_f1, micro_f1, pair_f1, model_clusters, model_Singletons, gold_clusters, gold_Singletons \
                = cluster_test(args, cluster_predict_list)
            print('Ave-prec=', ave_prec, 'macro_prec=', macro_prec, 'micro_prec=', micro_prec,
                  'pair_prec=', pair_prec)
            print('Ave-recall=', ave_recall, 'macro_recall=', macro_recall, 'micro_recall=', micro_recall,
                  'pair_recall=', pair_recall)
            print('Ave-F1=', ave_f1, 'macro_f1=', macro_f1, 'micro_f1=', micro_f1, 'pair_f1=', pair_f1)
            print('Model: #Clusters: %d, #Singletons %d' % (model_clusters, model_Singletons))
            print('Gold: #Clusters: %d, #Singletons %d' % (gold_clusters, gold_Singletons))
            print()

            ent_embedding = torch.cat((structure_outputs, context_outputs, outputs), dim=-1)
            ent_embedding = ent_embedding.cpu().detach().numpy()
            embeddings = []
            for id in issub:
                embeddings.append(ent_embedding[id])
            labels, clusters_center = HAC_getClusters(args, embeddings, cluster_threshold_real, 900)
            cluster_predict_list = list(labels)

            ave_prec, ave_recall, ave_f1, macro_prec, micro_prec, pair_prec, macro_recall, micro_recall, \
            pair_recall, macro_f1, micro_f1, pair_f1, model_clusters, model_Singletons, gold_clusters, gold_Singletons \
                = cluster_test(args, cluster_predict_list)
            print('Ave-prec=', ave_prec, 'macro_prec=', macro_prec, 'micro_prec=', micro_prec,
                  'pair_prec=', pair_prec)
            print('Ave-recall=', ave_recall, 'macro_recall=', macro_recall, 'micro_recall=', micro_recall,
                  'pair_recall=', pair_recall)
            print('Ave-F1=', ave_f1, 'macro_f1=', macro_f1, 'micro_f1=', micro_f1, 'pair_f1=', pair_f1)
            print('Model: #Clusters: %d, #Singletons %d' % (model_clusters, model_Singletons))
            print('Gold: #Clusters: %d, #Singletons %d' % (gold_clusters, gold_Singletons))
            print()

            if step >= margin_step:
                new_entity_cluster_list = []
                for i in range(nentity):
                    new_entity_cluster_list.append([i])
                clust2ent = {}
                for sub_id, cluster_id in enumerate(cluster_predict_list):
                    if cluster_id in clust2ent.keys():
                        clust2ent[cluster_id].append(sub_id)
                    else:
                        clust2ent[cluster_id] = [sub_id]
                for i, pred_label in enumerate(cluster_predict_list):
                    new_entity_cluster_list[i] = clust2ent[pred_label]
                    for ent in clust2ent[pred_label]:
                        for exist_id in origin_entity_cluster_list[ent]:
                            if exist_id not in new_entity_cluster_list[i]:
                                new_entity_cluster_list[i].append(exist_id)
                for i in range(len(issub), nentity):
                    new_entity_cluster_list[i] = origin_entity_cluster_list[i]
                for i in range(len(issub)):
                    for ent in new_entity_cluster_list[i]:
                        for exist_id in new_entity_cluster_list[ent]:
                            if exist_id not in new_entity_cluster_list[i]:
                                new_entity_cluster_list[i].append(exist_id)
                entity_cluster_list = new_entity_cluster_list

        sys.stdout.flush()
