import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument('-MODEL_DIR', type=str, default='./models')
parser.add_argument('-DATA_DIR', type=str, default='./data')
parser.add_argument('-FILE_DIR', type=str, default='./file')


parser.add_argument("-data_path", type=str, default='./data/OPIEC59K/OPIEC59K_test')
# parser.add_argument("-data_path", type=str, default='./data/reverb45k_change/reverb45k_change_test_read')
parser.add_argument("-dataset", type=str, default='OPIEC59K', choices=['reverb45k_change', 'OPIEC59K'])
parser.add_argument("-embedding_path", type=str, default='./embedding/OPIEC59K')

# model
parser.add_argument("-model", type=str, default='mvc')
parser.add_argument("-test_model", type=str, default=None)
parser.add_argument("-embed_size", type=int, default=300)
parser.add_argument("-metric", type=str, default='cosine')
parser.add_argument("-linkage", type=str, default='complete', choices=['complete', 'single', 'average'])


# training
parser.add_argument("-max_steps", type=int, default=250000)
parser.add_argument("-batch_size", type=int, default=200)
parser.add_argument("-lr", type=float, default=0.0001)
parser.add_argument("-bert_lr", type=float, default=0.00001)
parser.add_argument("-negative_number", type=int, default=32)
parser.add_argument("-single_gamma", type=float, default=12.0)
parser.add_argument("-alpha", type=float, default=0.5)
parser.add_argument("-beta", type=float, default=0.5)
parser.add_argument("-dropout", type=float, default=0.)
parser.add_argument("-gpu", type=int, default=3, help='-1 if not use gpu, >=0 if use gpu')
parser.add_argument('-random_seed', type=int, default=1, help='0 if randomly initialize the model, other if fix the seed')
parser.add_argument('-bert_dir', type=str, default="bert-base-uncased")


args = parser.parse_args()
command = ' '.join(['python'] + sys.argv)
args.command = command
