import argparse
import tensorflow as tf
import sys
import os.path

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.item_ranking.cdae import ICDAE
from models.item_ranking.bprmf import BPRMF
from models.item_ranking.cml import CML
from models.item_ranking.neumf import NeuMF
from models.item_ranking.gmf import GMF
from models.item_ranking.jrl import JRL
from models.item_ranking.mlp import MLP
from models.item_ranking.lrml import LRML
from models.item_ranking.neumf_my import NeuMF_my
from models.item_ranking.neumf_my_tail import NeuMF_my_tail
from models.item_ranking.NeuMF_cikm import NeuMF_my_cikm
from models.item_ranking.NeuMF_cikm_que import NeuMF_my_cikm_que
from models.item_ranking.NeuMF_cikm_p import NeuMF_my_cikm_p
# from utils.load_data.load_data_ranking import *
from utils.load_data.load_data_my import *

def parse_args():
    parser = argparse.ArgumentParser(description='DeepRec')
    parser.add_argument('--model', choices=['CDAE', 'CML', 'NeuMF', 'GMF', 'MLP', 'BPRMF', 'JRL', 'LRML'],
                        default='NeuMF_my_cikm_p')
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--num_factors', type=int, default=10)
    parser.add_argument('--display_step', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=1024)  # 128 for unlimpair
    parser.add_argument('--learning_rate', type=float, default=1e-3)  # 1e-4 for unlimpair
    parser.add_argument('--reg_rate', type=float, default=0.1)  # 0.01 for unlimpair
    parser.add_argument('--A2C_weight', type=float, default=100)  # 0.01 for unlimpair
    parser.add_argument('--center_weight', type=float, default=0.001)  # 0.01 for unlimpair
    parser.add_argument('--pseudo_weight', type=float, default=0.001)  # 0.01 for unlimpair
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    epochs = args.epochs
    learning_rate = args.learning_rate
    reg_rate = args.reg_rate
    num_factors = args.num_factors
    display_step = args.display_step
    batch_size = args.batch_size

    # train_data, test_data, n_user, n_item = load_data_neg(test_size=0.2, sep="\t")
    # train_data, test_data, n_user, n_item = load_data_myneg(test_size=0.2, sep=";;")
    train_data, test_data , n_qids, test_data_hot, test_data_long, hot_item, long_item, hot_dic, long_dic = load_data_myneg_cikm()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        model = None
        # Model selection
        if args.model == "CDAE":
            train_data, test_data, n_user, n_item = load_data_all(test_size=0.2, sep="\t")
            model = ICDAE(sess, n_user, n_item)
        if args.model == "CML":
            model = CML(sess, n_user, n_item)
        if args.model == "LRML":
            model = LRML(sess, n_user, n_item)
        if args.model == "BPRMF":
            model = BPRMF(sess, n_user, n_item)
        if args.model == "NeuMF":
            model = NeuMF(sess, n_user, n_item)
        if args.model == "GMF":
            model = GMF(sess, n_user, n_item)
        if args.model == "MLP":
            model = MLP(sess, n_user, n_item)
        if args.model == "JRL":
            model = JRL(sess, n_user, n_item)
        if args.model == "NeuMF_my":
            model = NeuMF_my(sess, n_user, n_item)
        if args.model == "NeuMF_my_tail":
            model = NeuMF_my_tail(sess, n_user, n_item)

        if args.model == "NeuMF_my_cikm":
            model = NeuMF_my_cikm(sess, 1, 1)

        if args.model == "NeuMF_my_cikm_query":
            model = NeuMF_my_cikm_que(sess, 1, 1)
        if args.model == "NeuMF_my_cikm_p":
            model = NeuMF_my_cikm_p(sess, 1, 1,epoch=epochs,A2C_weight=args.A2C_weight,center_weight=args.center_weight,pseudo_weight=args.pseudo_weight)
        # build and execute the model
        if model is not None:
            model.build_network_my()
            # model.execute(train_data, test_data)
            model.execute_my(train_data, test_data, n_qids, test_data_hot, test_data_long, hot_item,long_item, hot_dic, long_dic)
