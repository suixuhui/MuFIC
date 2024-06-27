import numpy
import torch.nn as nn
from pytorch_transformers.modeling_bert import (
    BertPreTrainedModel,
    BertConfig,
    BertModel,
)
import torch.nn.functional as F
import torch
from options import args

class MMS_loss(torch.nn.Module):
    def __init__(self):
        super(MMS_loss, self).__init__()

    def forward(self, S, margin=0.001):
        deltas = margin * torch.eye(S.size(0)).to(S.device)
        S = S - deltas

        target = torch.LongTensor(list(range(S.size(0)))).to(S.device)
        I2C_loss = F.nll_loss(F.log_softmax(S, dim=1), target)
        C2I_loss = F.nll_loss(F.log_softmax(S.t(), dim=1), target)
        loss = I2C_loss + C2I_loss
        return loss


class MVC(nn.Module):

    def __init__(self, args, E_init, R_init, entity_context_token_ids, entity_context_embedding):
        super(MVC, self).__init__()
        self.name = args.model
        self.gpu = args.gpu
        self.E_init = torch.from_numpy(E_init).cuda(self.gpu)
        self.R_init = torch.from_numpy(R_init).cuda(self.gpu)
        self.entity_embedding = nn.Parameter(self.E_init)
        self.relation_embedding = nn.Parameter(self.R_init)

        self.all_entity_context_embedding_init = entity_context_embedding.cuda(self.gpu)
        self.all_entity_context_embedding = nn.Parameter(self.all_entity_context_embedding_init)
        self.entity_context_token_ids = entity_context_token_ids
        self.bert = BertModel.from_pretrained(args.bert_dir)
        self.dropout = nn.Dropout(0.5)
        self.leaner = nn.Linear(768, 300)

        self.embedding1_weights = nn.Linear(300, 300)
        self.embedding2_weights = nn.Linear(300, 300)
        self.sigmod = nn.Sigmoid()
        self.structure_guided_multihead_attn = nn.MultiheadAttention(300, 4)
        self.context_guided_multihead_attn = nn.MultiheadAttention(300, 4)

        self.recon_context = nn.Sequential(
            nn.Linear(300, 768),
            nn.ReLU(inplace=True),
            nn.Linear(768, 768),
            nn.ReLU(inplace=True)
        )
        self.mse = nn.MSELoss(reduction='none')

        self.mms = MMS_loss()



    def forward(self, positive_sample, negative_sample, mode, entity_cluster_list, relation_cluster_list, flag):

        negative_score = self.get_structure_score((positive_sample, negative_sample), mode=mode)
        positive_score = self.get_structure_score(positive_sample)
        positive_score = positive_score.repeat(1, args.negative_number)
        gamma = torch.full((1, args.negative_number), float(args.single_gamma)).cuda(self.gpu)
        loss = self.hinge_loss(positive_score, negative_score, gamma)
        loss = loss.mean()

        sub_context_embedding = None
        obj_context_embedding = None
        for sample in positive_sample:
            sub_input_id = self.entity_context_token_ids[sample[0]]
            obj_input_id = self.entity_context_token_ids[sample[2]]
            sub_input_id, obj_input_id = torch.LongTensor(sub_input_id).cuda(self.gpu), torch.LongTensor(obj_input_id).cuda(self.gpu)
            sub_input_id, obj_input_id = torch.unsqueeze(sub_input_id, dim=0), torch.unsqueeze(obj_input_id, dim=0)
            sub_output_bert, _ = self.bert(sub_input_id)
            obj_output_bert, _ = self.bert(obj_input_id)

            sub_embedding = None
            sub_input_id = torch.squeeze(sub_input_id)
            for i in range(len(sub_input_id)):
                if sub_input_id[i] == 1:
                    sub_embedding = sub_output_bert[0][i]
                    break
            if sub_embedding == None:
                sub_embedding = sub_output_bert[0][0]
            obj_embedding = None
            obj_input_id = torch.squeeze(obj_input_id)
            for i in range(len(obj_input_id)):
                if obj_input_id[i] == 1:
                    obj_embedding = obj_output_bert[0][i]
                    break
            if obj_embedding == None:
                obj_embedding = obj_output_bert[0][0]

            if sub_context_embedding == None:
                sub_context_embedding = sub_embedding
            else:
                sub_context_embedding = torch.cat((sub_context_embedding, sub_embedding))

            if obj_context_embedding == None:
                obj_context_embedding = obj_embedding
            else:
                obj_context_embedding = torch.cat((obj_context_embedding, obj_embedding))
        sub_context_embedding = sub_context_embedding.view(-1, 768)
        obj_context_embedding = obj_context_embedding.view(-1, 768)

        sub_context_embedding = self.dropout(sub_context_embedding)
        obj_context_embedding = self.dropout(obj_context_embedding)

        sub_context_embedding = self.leaner(sub_context_embedding)
        obj_context_embedding = self.leaner(obj_context_embedding)


        head_seed_index, relation_seed_index, tail_seed_index = [], [], []
        for sample in positive_sample:
            head_seed_index.append(entity_cluster_list[sample[0]])
            relation_seed_index.append(relation_cluster_list[sample[1]])
            tail_seed_index.append(entity_cluster_list[sample[2]])

        head_structure = torch.index_select(
            self.entity_embedding,
            dim=0,
            index=positive_sample[:, 0]
        )

        relation = torch.index_select(
            self.relation_embedding,
            dim=0,
            index=positive_sample[:, 1]
        )

        tail_structure = torch.index_select(
            self.entity_embedding,
            dim=0,
            index=positive_sample[:, 2]
        )

        head = self.aggregate(head_structure, sub_context_embedding)
        tail = self.aggregate(tail_structure, obj_context_embedding)

        bs = positive_score.size(0)

        head_seed_stucture = None
        for indexs in head_seed_index:
            head_seed_embedding = self.entity_embedding[indexs]
            if head_seed_stucture == None:
                head_seed_stucture = torch.mean(head_seed_embedding, dim=0)
            else:
                head_seed_stucture = torch.cat((head_seed_stucture, torch.mean(head_seed_embedding, dim=0)))
        relation_seed = None
        for indexs in relation_seed_index:
            if relation_seed == None:
                relation_seed = torch.mean(self.relation_embedding[indexs], dim=0)
            else:
                relation_seed = torch.cat((relation_seed, torch.mean(self.relation_embedding[indexs], dim=0)))
        tail_seed_stucture = None
        for indexs in tail_seed_index:
            tail_seed_embedding = self.entity_embedding[indexs]
            if tail_seed_stucture == None:
                tail_seed_stucture = torch.mean(tail_seed_embedding, dim=0)
            else:
                tail_seed_stucture = torch.cat((tail_seed_stucture, torch.mean(tail_seed_embedding, dim=0)))

        head_seed_stucture, relation_seed, tail_seed_stucture = head_seed_stucture.view(bs, -1), relation_seed.view(bs, -1), tail_seed_stucture.view(bs, -1)
        head_stucture_prototype, relation_prototype, tail_stucture_prototype = head_structure.mm(head_seed_stucture.t()), relation.mm(relation_seed.t()), tail_structure.mm(tail_seed_stucture.t())
        target = torch.LongTensor(torch.arange(bs)).cuda(self.gpu)
        sturcture_prototype_loss = (F.cross_entropy(head_stucture_prototype, target, reduction="mean") + F.cross_entropy(relation_prototype, target, reduction="mean") + F.cross_entropy(tail_stucture_prototype, target, reduction="mean")) / 3
        loss += sturcture_prototype_loss

        head_seed_context = None
        for indexs in head_seed_index:
            head_seed_context_embedding = self.all_entity_context_embedding[indexs]
            head_seed_context_embedding = self.dropout(head_seed_context_embedding)
            head_seed_embedding = self.leaner(head_seed_context_embedding)
            if head_seed_context == None:
                head_seed_context = torch.mean(head_seed_embedding, dim=0)
            else:
                head_seed_context = torch.cat((head_seed_context, torch.mean(head_seed_embedding, dim=0)))
        tail_seed_context = None
        for indexs in tail_seed_index:
            tail_seed_context_embedding = self.all_entity_context_embedding[indexs]
            tail_seed_context_embedding = self.dropout(tail_seed_context_embedding)
            tail_seed_embedding  = self.leaner(tail_seed_context_embedding)
            if tail_seed_context == None:
                tail_seed_context = torch.mean(tail_seed_embedding, dim=0)
            else:
                tail_seed_context = torch.cat((tail_seed_context, torch.mean(tail_seed_embedding, dim=0)))

        head_seed_context, tail_seed_context = head_seed_context.view(bs, -1), tail_seed_context.view(bs, -1)
        head_context_prototype, tail_context_prototype = sub_context_embedding.mm(head_seed_context.t()), obj_context_embedding.mm(tail_seed_context.t())
        target = torch.LongTensor(torch.arange(bs)).cuda(self.gpu)
        context_prototype_loss = (F.cross_entropy(head_context_prototype, target, reduction="mean") + F.cross_entropy(tail_context_prototype, target, reduction="mean")) / 2
        loss += context_prototype_loss

        head_seed = None
        for indexs in head_seed_index:
            head_seed_context_embedding = self.all_entity_context_embedding[indexs]
            head_seed_context_embedding = self.dropout(head_seed_context_embedding)
            head_seed_context_embedding = self.leaner(head_seed_context_embedding)
            head_seed_embedding = self.aggregate(self.entity_embedding[indexs], head_seed_context_embedding)
            if head_seed == None:
                head_seed = torch.mean(head_seed_embedding, dim=0)
            else:
                head_seed = torch.cat((head_seed, torch.mean(head_seed_embedding, dim=0)))
        tail_seed = None
        for indexs in tail_seed_index:
            tail_seed_context_embedding = self.all_entity_context_embedding[indexs]
            tail_seed_context_embedding = self.dropout(tail_seed_context_embedding)
            tail_seed_context_embedding = self.leaner(tail_seed_context_embedding)
            tail_seed_embedding = self.aggregate(self.entity_embedding[indexs], tail_seed_context_embedding)
            if tail_seed == None:
                tail_seed = torch.mean(tail_seed_embedding, dim=0)
            else:
                tail_seed = torch.cat((tail_seed, torch.mean(tail_seed_embedding, dim=0)))

        head_seed, tail_seed = head_seed.view(bs, -1), tail_seed.view(bs, -1)
        head_prototype, tail_prototype = head.mm(head_seed.t()), tail.mm(tail_seed.t())
        target = torch.LongTensor(torch.arange(bs)).cuda(self.gpu)
        prototype_loss = (F.cross_entropy(head_prototype, target, reduction="mean") + F.cross_entropy(tail_prototype, target, reduction="mean")) / 2
        loss += prototype_loss

        sub_context_recon = self.recon_context(sub_context_embedding)
        obj_context_recon = self.recon_context(obj_context_embedding)
        head_context_init = torch.index_select(
            self.all_entity_context_embedding_init,
            dim=0,
            index=positive_sample[:, 0]
        )

        tail_context_init = torch.index_select(
            self.all_entity_context_embedding_init,
            dim=0,
            index=positive_sample[:, 2]
        )
        mse_s = torch.mean(self.mse(sub_context_recon, head_context_init), dim=-1)
        mse_o = torch.mean(self.mse(obj_context_recon, tail_context_init), dim=-1)
        mse_loss = torch.mean((mse_s + mse_o) / 2)
        loss += mse_loss

        head_batch_sim = torch.matmul(head_structure, sub_context_embedding.t())
        head_contrast_loss = self.mms(head_batch_sim)
        tail_batch_sim = torch.matmul(tail_structure, obj_context_embedding.t())
        tail_contrast_loss = self.mms(tail_batch_sim)
        contrast_loss = (head_contrast_loss + tail_contrast_loss) / 2
        loss += contrast_loss

        if flag == "test":
            outputs = None
            for input_id in self.entity_context_token_ids:
                input_id = torch.LongTensor(input_id).cuda(self.gpu)
                input_id = torch.unsqueeze(input_id, dim=0)
                output_bert, _ = self.bert(input_id)

                embedding = None
                input_id = torch.squeeze(input_id)
                for i in range(len(input_id)):
                    if input_id[i] == 1:
                        embedding = output_bert[0][i]
                        break
                if embedding == None:
                    embedding = output_bert[0][0]

                if outputs == None:
                    outputs = embedding
                else:
                    outputs = torch.cat((outputs, embedding))
            context_outputs = outputs.view(-1, 768)
            context_outputs = self.leaner(context_outputs)
            outputs = self.aggregate(self.entity_embedding, context_outputs)
            return loss, self.entity_embedding, context_outputs, outputs
        else:
            outputs = sub_context_embedding

        return loss, outputs


    def get_structure_score(self, sample, mode='single'):

        if mode == 'single':

            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=sample[:, 0]
            ).unsqueeze(1)

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=sample[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=sample[:, 2]
            ).unsqueeze(1)

        elif mode == 'head-batch':
            tail_part, head_part = sample
            batch_size, negative_sample_size = head_part.size(0), head_part.size(1)

            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=head_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=tail_part[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=tail_part[:, 2]
            ).unsqueeze(1)

        elif mode == 'tail-batch':
            head_part, tail_part = sample
            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)

            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=head_part[:, 0]
            ).unsqueeze(1)

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=head_part[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=tail_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)

        else:
            raise ValueError('mode %s not supported' % mode)

        if mode == 'head-batch':
            score = head + (relation - tail)
        else:
            score = (head + relation) - tail

        score = torch.norm(score, p=1, dim=2)

        return score

    def hinge_loss(self, positive_score, negative_score, gamma):
        err = positive_score - negative_score + gamma
        max_err = err.clamp(0)
        return max_err

    def aggregate(self, embedding1, embedding2):

        key = value = torch.unsqueeze(embedding2, dim=0)
        query = torch.unsqueeze(embedding1, dim=0)
        v1, _ = self.structure_guided_multihead_attn(query, key, value)
        v1 = torch.squeeze(v1)
        if v1.size(0) == v1.size(-1):
            v1 = torch.unsqueeze(v1, dim=0)
        key = value = torch.unsqueeze(embedding1, dim=0)
        query = torch.unsqueeze(v1, dim=0)
        v2, _ = self.structure_guided_multihead_attn(query, key, value)
        v2 = torch.squeeze(v2)
        if v2.size(0) == v2.size(-1):
            v2 = torch.unsqueeze(v2, dim=0)
        g = self.sigmod(torch.matmul(v1, self.embedding1_weights.weight) +
                        torch.matmul(v2, self.embedding2_weights.weight))
        v = torch.mul((1 - g), v1) + torch.mul(g, v2)
        return v


def pick_model(args, E_init, R_init, entity_context_tokens_id, entity_context_embedding):
    if args.model == "mvc":
        model = MVC(args, E_init, R_init, entity_context_tokens_id, entity_context_embedding)
    else:
        raise RuntimeError("wrong model name")
    if args.gpu >= 0:
        model.cuda(args.gpu)
    return model