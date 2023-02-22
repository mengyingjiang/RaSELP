import sqlite3  # 用于打开数据库
import time
from sklearn.model_selection import StratifiedKFold
from RaSELP import *
from model import *
from loss import *
from accuracy import *
from utils import *
import warnings
warnings.filterwarnings("ignore")

file_path="./"
feature_list = ["smile","target","enzyme"]


def main():

    conn = sqlite3.connect("./event.db")
    df_drug = pd.read_sql('select * from drug;', conn)
    extraction = pd.read_sql('select * from extraction;', conn)
    X_drug = pd.read_csv("./df_drug.csv")
    drug_smile = X_drug['smile']
    # for large dataset

    # df_drug = pd.read_csv("./drug_information_del_noDDIxiaoyu50.csv")
    # extraction = pd.read_csv("./df_extraction_cleanxiaoyu50.csv")
    # drug_smile = df_drug['smile']

    mechanism = extraction['mechanism']  # 37264 所有交互边的 mechanism #
    action = extraction['action']
    drugA = extraction['drugA']
    drugB = extraction['drugB']
    drug_coding = drug_smile_coding(drug_smile, drug_encoding)

    new_label, drugA, drugB, event_num, X_vector, DDI_edge, multi_graphs = prepare(df_drug,feature_list,mechanism,action,drugA,drugB)

    print("dataset len", len(new_label))
    result_all = np.zeros((6, 0), dtype=float)  # vector=[]
    start=time.time()
    result_all_2, result_eve_2, result_all_3, result_eve_3 = cross_val(new_label, drugA, drugB, event_num, X_vector, DDI_edge, multi_graphs, drug_coding)
    # result_all = np.hstack((result_all, result))
    # print(result)

    print("time used:", (time.time() - start) / 3600)
    # save_result(file_path,"all",result_all,task)
    # save_result(file_path,"each",result_eve,task)

def cross_val(label, drugA, drugB, event_num, X_vector, DDI_edge, multi_graph, drug_coding):

    y_true_2 = np.array([])
    y_score_2 = np.zeros((0, event_num), dtype=float)
    y_pred_2= np.array([])

    y_true_3 = np.array([])
    y_score_3 = np.zeros((0, event_num), dtype=float)
    y_pred_3 = np.array([])

    skf = StratifiedKFold(n_splits=cross_ver_tim)
    # ddi_newfea = ddi_newfea_gene(DDI_edge, label, event_num, X_vector)
    if task == "task2" or "task3":
        temp_drug1 = [[] for i in range(event_num)]
        temp_drug2 = [[] for i in range(event_num)]
        for i in range(len(label)):
            temp_drug1[label[i]].append(drugA[i])  # list，有65行，每一行表示 交互类型为i的一半drug
            temp_drug2[label[i]].append(drugB[i])  # list，有65行，每一行表示 交互类型为i的另一半drug
        drug_cro_dict = {}
        for i in range(event_num):
            for j in range(len(temp_drug1[i])):  # 遍历temp_drug中的每一行和每行的药物
                drug_cro_dict[temp_drug1[i][
                    j]] = j % cross_ver_tim  # 计算将该药物参与的交互类型与5的余数，从而得到一个dict，key为的药物，value为这些药物参与的交互类型与5的余数（0,1,2,3,4）
                drug_cro_dict[temp_drug2[i][j]] = j % cross_ver_tim
        train_drug = [[] for i in range(cross_ver_tim)]
        test_drug = [[] for i in range(cross_ver_tim)]
        for i in range(cross_ver_tim):
            for dr_key in drug_cro_dict.keys():
                if drug_cro_dict[dr_key] == i:
                    test_drug[i].append(dr_key)
                else:
                    train_drug[i].append(dr_key)

    cross_ver = 0
    for train_index, test_index in skf.split(DDI_edge, label):
        if task == "task1":
            y_train, y_test = label[train_index], label[test_index]  # 标签 [29811],[7453]
            ddi_edge_train, ddi_edge_test = DDI_edge[train_index], DDI_edge[test_index]

        if task == "task2" or "task3":
            y_train = [];y_test_2 = [];y_test_3 = [];
            ddi_edge_train = [];ddi_edge_test_2 = [];ddi_edge_test_3 = []
            for i in range(len(drugA)):
                if (drugA[i] in np.array(train_drug[cross_ver])) and (
                        drugB[i] in np.array(train_drug[cross_ver])):  # 第i个ddi的两个drug都在训练集中，就将该ddi添加进训练集
                    y_train.append(label[i])
                    ddi_edge_train.append(DDI_edge[i])

                if (drugA[i] not in np.array(train_drug[cross_ver])) and (
                        drugB[i] in np.array(train_drug[cross_ver])):  # 第i个ddi的第一个drug不在训练集中，另一个在训练集中，就将其添加进测试集
                    y_test_2.append(label[i])
                    ddi_edge_test_2.append(DDI_edge[i])

                if (drugA[i] in np.array(train_drug[cross_ver])) and (
                        drugB[i] not in np.array(train_drug[cross_ver])):  # 第i个ddi的第一个drug不在训练集中，另一个在训练集中，就将其添加进测试集
                    y_test_2.append(label[i])  # 将第i个ddi的feature和label添加进测试集
                    ddi_edge_test_2.append(DDI_edge[i])

                if (drugA[i] not in np.array(train_drug[cross_ver])) and (
                        drugB[i] not in np.array(train_drug[cross_ver])):
                    y_test_3.append(label[i])
                    ddi_edge_test_3.append(DDI_edge[i])

        # if task == "task3":
        #     y_train = [];y_test = []
        #     ddi_edge_train = [];ddi_edge_test = []
        #     for i in range(len(drugA)):
        #         if (drugA[i] in np.array(train_drug[cross_ver])) and (drugB[i] in np.array(train_drug[cross_ver])):
        #             y_train.append(label[i])
        #             ddi_edge_train.append(DDI_edge[i])
        #             # 如果该ddi涉及的drug都在train_drug中，则该ddi属于训练集
        #
        #         if (drugA[i] not in np.array(train_drug[cross_ver])) and (
        #                 drugB[i] not in np.array(train_drug[cross_ver])):
        #             y_test.append(label[i])
        #             ddi_edge_test.append(DDI_edge[i])



        y_train = np.array(y_train)
        y_test_2 = np.array(y_test_2)
        y_test_3 = np.array(y_test_3)

        ddi_edge_train = np.array(ddi_edge_train, dtype = int)
        ddi_edge_test_2 = np.array(ddi_edge_test_2, dtype=int)
        ddi_edge_test_3 = np.array(ddi_edge_test_3, dtype=int)

        adj = adj_Heter_gene(ddi_edge_train, X_vector, event_num, y_train)

        model =  RaSELP(len(X_vector[0])*2, event_num, n_hop)

        print("train len", len(y_train))
        print("task 2 test len", len(y_test_2))
        print("task 3 test len", len(y_test_3))

        pred_score_2,pred_score_3 = RaSELP_train(model, y_train, y_test_2, y_test_3,event_num,X_vector ,adj,
                                    ddi_edge_train,ddi_edge_test_2,ddi_edge_test_3, multi_graph, drug_coding)  # [29811,3432],[29811],[7453,3432],[7453],65
        cross_ver = cross_ver + 1
        pred_type_2 = np.argmax(pred_score_2, axis=1)
        y_pred_2 = np.hstack((y_pred_2, pred_type_2))
        y_score_2 = np.row_stack((y_score_2, pred_score_2))
        y_true_2 = np.hstack((y_true_2, y_test_2))
        result_all_now_2,_ = evaluate(y_pred_2, y_score_2, y_true_2, event_num)
        print(result_all_now_2)

        pred_type_3 = np.argmax(pred_score_3, axis=1)
        y_pred_3= np.hstack((y_pred_3, pred_type_3))
        y_score_3 = np.row_stack((y_score_3, pred_score_3))
        y_true_3 = np.hstack((y_true_3, y_test_3))
        result_all_now_3,_ = evaluate(y_pred_3, y_score_3, y_true_3, event_num)
        print(result_all_now_3)

        del model
        del ddi_edge_train
        gc.collect()

        # break
    result_all_2, result_eve_2 = evaluate(y_pred_2, y_score_2, y_true_2, event_num)
    result_all_3, result_eve_3 = evaluate(y_pred_3, y_score_3, y_true_3, event_num)
    print(result_all_2)
    print(result_all_3)

    return result_all_2, result_eve_2, result_all_3, result_eve_3

main()





