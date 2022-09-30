import opt
import torch
from torch.optim import Adam
import torch.nn.functional as F
from sklearn.cluster import KMeans
from utils import adjust_learning_rate
from utils import eva, target_distribution
import numpy as np
import time
import scipy.stats
from info_nce import InfoNCE
from scipy.special import entr

acc_reuslt = []
nmi_result = []
ari_result = []
f1_result = []
use_adjust_lr = ['usps', 'hhar', 'reut', 'acm', 'dblp', 'cite']

def JS_divergence(p, q):
    M = (p + q) / 2
    return 0.5 * scipy.stats.entropy(p, M) + 0.5 * scipy.stats.entropy(q, M)

def mine_nearest_neighbors2(features, str_sim, t, T, JS,device):
    # mine the topk nearest neighbors for every sample

    fea_sim = torch.matmul(features, features.T)
    fea_sim_T = fea_sim.T
    Zmax, Zmin = fea_sim_T.max(axis=0), fea_sim_T.min(axis=0)
    Z2 = (fea_sim_T - Zmin.values) / (Zmax.values - Zmin.values)
    fea_sim = Z2.T
    # fea_sim = maxminnorm(fea_sim)
    # beta = 1/(1+math.exp(-(t/T-0.5)))
    beta = 1 - JS
    total_sim = beta * fea_sim + (1 - beta) * str_sim.to(device)
    indices = torch.argsort(-total_sim)
    return indices


def dy_calculate_INfo_loss(feature, top_k, y_pred_last, str_sim, t, T, JS,device):
    # distances, indices = mine_nearest_neighbors(z,y.shape[0]-1)
    # indices = mine_nearest_neighbors(feature,top_k)
    indices = mine_nearest_neighbors2(feature, str_sim, t, T, JS,device)
    pos = indices[:, 1:2]
    neg = indices[:, -top_k:]

    pos_feature = feature[pos]
    pos_feature = torch.squeeze(pos_feature, 1)
    neg_feature = feature[neg]
    info_loss = InfoNCE(negative_mode='paired')
    contra_loss = info_loss(feature, pos_feature, neg_feature)

    # cluster_class = {}
    # for i in range(args.n_clusters):
    #     cluster_class[i] = np.where(y_pred_last == i)[0]
    #
    # neg = []
    # for i in range(y_pred_last.shape[0]):
    #     neg_value = []
    #     i_cluster = y_pred_last[i]
    #     for j in range(args.n_clusters):
    #         if j != i_cluster:
    #             neg_cluster = cluster_class[j]
    #             nums = neg_cluster.shape[0]
    #             neg_index = np.random.randint(0, nums, size=top_k)
    #             neg_value = np.concatenate((neg_value,neg_cluster[neg_index].tolist()))
    #     neg.append(neg_value.tolist())
    # neg = np.array(neg)
    # pos_feature = feature[pos]
    # pos_feature = torch.squeeze(pos_feature,1)
    # neg_feature = feature[neg]
    # info_loss = InfoNCE(negative_mode='paired')
    # contra_loss = info_loss(feature, pos_feature, neg_feature)
    return contra_loss

def EuclideanDistances(A, B):
    BT = B.transpose()
    # vecProd = A * BT
    vecProd = np.dot(A, BT)
    # print(vecProd)
    SqA = A ** 2
    # print(SqA)
    sumSqA = np.matrix(np.sum(SqA, axis=1))
    sumSqAEx = np.tile(sumSqA.transpose(), (1, vecProd.shape[1]))
    # print(sumSqAEx)

    SqB = B ** 2
    sumSqB = np.sum(SqB, axis=1)
    sumSqBEx = np.tile(sumSqB, (vecProd.shape[0], 1))
    SqED = sumSqBEx + sumSqAEx - 2 * vecProd
    SqED[SqED < 0] = 0.0
    ED = np.sqrt(SqED)
    return ED

def calculate_Info_loss2(old_center, new_center):
    n = old_center.shape[0]
    neg = []
    for i in range(n):
        indice1 = []
        for j in range(n):
            if j != i:
                indice1 = np.concatenate((indice1, [j]))
                # indice.append(j)
        neg.append(indice1)
    neg = np.array(neg)
    distance = EuclideanDistances(np.array(new_center.cpu()), np.array(old_center.cpu()))
    pos = distance.argmin(1).tolist()
    pos = np.array(pos)
    pos_feature = old_center[pos.T]
    pos_feature = torch.squeeze(pos_feature, 1)
    start_tensor = new_center[neg[0]]
    for i in range(1, n):
        start_tensor = torch.cat((start_tensor, new_center[neg[i]]), axis=-2)
    start_tensor = start_tensor.reshape((n, n - 1, 20))
    neg_feature = start_tensor
    info_loss = InfoNCE(negative_mode='paired')
    contra_loss = info_loss(new_center, pos_feature, neg_feature)
    return contra_loss

def Train(epoch, model1, data, adj, label, lr, pre_model_save_path, final_model_save_path, n_clusters,
          original_acc, gamma_value, lambda_value, device,model2,str_sim):

    for item3 in [0.001,0.01,0.1,1,10,100,1000]:
        for item4 in [0.001,0.01,0.1,1,10,100,1000]:
            top_k = 5
            model_total_result = []
            time1 = time.time()
            epoch_dc = 120
            print('epoch_dc: ', epoch_dc)
            for k_epoch in range(5):

                optimizer1 = Adam(model1.parameters(), lr=lr)
                model1.load_state_dict(torch.load(pre_model_save_path, map_location='cpu'))
                optimizer2 = Adam(model2.parameters(), lr=lr)
                model2.load_state_dict(torch.load(pre_model_save_path, map_location='cpu'))
                pre_center = model1.cluster_layer.data

                with torch.no_grad():
                    x_hat, z_hat, adj_hat, z_ae, z_igae, _, _, _, z_tilde,pred = model1(data, adj)
                kmeans = KMeans(n_clusters=n_clusters, n_init=20)
                cluster_id = kmeans.fit_predict(z_tilde.data.cpu().numpy())
                model1.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)
                model2.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)
                eva(label, cluster_id, 'Initialization')

                model1_total_list = []
                model2_total_list = []
                best_acc = 0
                best_nmi = 0
                best_ari = 0
                best_f1 = 0
                original_acc = 0

                for epoch in range(epoch):
                    # if opt.args.name in use_adjust_lr:
                    #     adjust_learning_rate(optimizer, epoch)
                    x_hat1, z_hat1, adj_hat1, z_ae1, z_igae1, q1, q11, q21, z_tilde1,pred1 = model1(data, adj)
                    x_hat2, z_hat2, adj_hat2, z_ae2, z_igae2, q2, q12, q22, z_tilde2,pred2 = model2(data, adj)


                    tmp_q1 = q1.data
                    p1 = target_distribution(tmp_q1)
                    tmp_q2 = q2.data
                    p2 = target_distribution(tmp_q2)

                    kemans = KMeans(n_clusters=n_clusters, n_init=20)
                    ae_z_y_pred1 = kmeans.fit_predict(z_tilde1.data.cpu().numpy())
                    new_cluster_center = torch.tensor(kmeans.cluster_centers_).to(device)
                    ae_z_y_pred2 = kmeans.fit_predict(z_tilde2.data.cpu().numpy())

                    res11 = tmp_q1.cpu().numpy().argmax(1)  # Q
                    res12 = q21.data.cpu().numpy().argmax(1)  # Z
                    res13 = p1.data.cpu().numpy().argmax(1)  # P

                    res21 = tmp_q2.cpu().numpy().argmax(1)  # Q
                    res22 = q22.data.cpu().numpy().argmax(1)  # Z
                    res23 = p2.data.cpu().numpy().argmax(1)  # P

                    kmeans1 = KMeans(n_clusters=n_clusters, n_init=20).fit(z_tilde1.data.cpu().numpy())
                    kmeans2 = KMeans(n_clusters=n_clusters, n_init=20).fit(z_tilde2.data.cpu().numpy())


                    acc1, nmi1, ari1, f11 = eva(label, kmeans1.labels_, epoch)
                    acc_reuslt.append(acc1)
                    nmi_result.append(nmi1)
                    ari_result.append(ari1)
                    f1_result.append(f11)

                    acc2, nmi2, ari2, f12 = eva(label, kmeans2.labels_, epoch)
                    acc_reuslt.append(acc2)
                    nmi_result.append(nmi2)
                    ari_result.append(ari2)
                    f1_result.append(f12)

                    if acc1 > original_acc:
                        original_acc = acc1
                        best_acc = acc1
                        best_nmi =nmi1
                        best_f1 = f11
                        best_ari = ari1
                        torch.save(model1.state_dict(), final_model_save_path)
                    if acc2 > original_acc:
                        original_acc = acc2
                        best_acc = acc2
                        best_nmi =nmi2
                        best_f1 = f12
                        best_ari = ari2
                        torch.save(model2.state_dict(), final_model_save_path)


                    if epoch < epoch_dc:

                        JS2 = JS_divergence(res22, res23)
                        contr_dy_loss = dy_calculate_INfo_loss(z_tilde2, top_k, ae_z_y_pred2, str_sim, epoch,
                                                               200, JS2,device)

                        contr_center_loss = calculate_Info_loss2(pre_center, new_cluster_center)

                        loss_ae1 = F.mse_loss(x_hat1, data)
                        loss_w1 = F.mse_loss(z_hat1, torch.spmm(adj, data))
                        loss_a1 = F.mse_loss(adj_hat1, adj.to_dense())
                        loss_igae1 = loss_w1 + gamma_value * loss_a1
                        loss_kl1 = F.kl_div((q1.log() + q11.log() + q21.log()) / 3, p1, reduction='batchmean')
                        loss1 = loss_ae1 + loss_igae1 + lambda_value * loss_kl1
                        loss1_fin = loss1 + item3 * contr_center_loss


                        loss_ae2 = F.mse_loss(x_hat2, data)
                        loss_w2 = F.mse_loss(z_hat2, torch.spmm(adj, data))
                        loss_a2 = F.mse_loss(adj_hat2, adj.to_dense())
                        loss_igae2 = loss_w2 + gamma_value * loss_a2
                        loss_kl2 = F.kl_div((q2.log() + q12.log() + q22.log()) / 3, p2, reduction='batchmean')
                        loss2 = loss_ae2 + loss_igae2 + lambda_value * loss_kl2
                        loss2_fin = loss2 + item4* contr_dy_loss

                        optimizer1.zero_grad()
                        loss1_fin.backward()
                        optimizer1.step()

                        optimizer2.zero_grad()
                        loss2_fin.backward()
                        optimizer2.step()

                        model1_entropy = entr(q21.cpu().detach().numpy()).sum(axis=1)
                        model1_total_list.append(model1_entropy.tolist())
                        model2_entropy = entr(q22.cpu().detach().numpy()).sum(axis=1)
                        model2_total_list.append(model2_entropy.tolist())
                    else:
                        model1_entropy = entr(q21.cpu().detach().numpy()).sum(axis=1)
                        model1_total_list.append(model1_entropy.tolist())
                        model2_entropy = entr(q22.cpu().detach().numpy()).sum(axis=1)
                        model2_total_list.append(model2_entropy.tolist())

                        model1_clustering_score = np.array(model1_total_list).mean(0)
                        model1_clustering_score_std = np.array(model1_total_list).std(0)
                        model2_clustering_score = np.array(model2_total_list).mean(0)
                        model2_clustering_score_std = np.array(model2_total_list).std(0)

                        if len(model1_total_list) > 1:
                            model1_train_mask = np.where(model1_clustering_score > 0.8)[0]
                            model2_train_mask = np.where(model2_clustering_score > 0.8)[0]
                        else:
                            model1_train_mask = np.ones(len(model1_total_list[0])).astype(int)
                            model2_train_mask = np.ones(len(model2_total_list[0])).astype(int)

                        JS2 = JS_divergence(res22, res23)
                        contr_dy_loss = dy_calculate_INfo_loss(z_tilde2[model1_train_mask], top_k,
                                                               ae_z_y_pred2[model1_train_mask],
                                                               str_sim[model1_train_mask][:, model1_train_mask], epoch,
                                                               200, JS2,device)

                        contr_center_loss = calculate_Info_loss2(pre_center, new_cluster_center)

                        loss_ae1 = F.mse_loss(x_hat1[model2_train_mask], data[model2_train_mask])
                        loss_w1 = F.mse_loss(z_hat1[model2_train_mask], torch.spmm(adj, data)[model2_train_mask])
                        loss_a1 = F.mse_loss(adj_hat1[model2_train_mask], adj.to_dense()[model2_train_mask])
                        loss_igae1 = loss_w1 + gamma_value * loss_a1
                        loss_kl1 = F.kl_div((q1.log()[model2_train_mask] + q11.log() [model2_train_mask]+ q21.log()[model2_train_mask]) / 3, p1[model2_train_mask], reduction='batchmean')
                        loss1 = loss_ae1 + loss_igae1 + lambda_value * loss_kl1
                        loss1_fin = loss1 + item3 * contr_center_loss


                        loss_ae2 = F.mse_loss(x_hat2[model1_train_mask], data[model1_train_mask])
                        loss_w2 = F.mse_loss(z_hat2[model1_train_mask], torch.spmm(adj, data)[model1_train_mask])
                        loss_a2 = F.mse_loss(adj_hat2[model1_train_mask], adj.to_dense()[model1_train_mask])
                        loss_igae2 = loss_w2 + gamma_value * loss_a2
                        loss_kl2 = F.kl_div((q2.log()[model1_train_mask] + q12.log()[model1_train_mask] + q22.log()[model1_train_mask]) / 3, p2[model1_train_mask], reduction='batchmean')
                        loss2 = loss_ae2 + loss_igae2 + lambda_value * loss_kl2
                        loss2_fin = loss2 + item4* contr_dy_loss

                        optimizer1.zero_grad()
                        loss1_fin.backward()
                        optimizer1.step()

                        optimizer2.zero_grad()
                        loss2_fin.backward()
                        optimizer2.step()
                    pre_center = new_cluster_center
                model_total_result.append([best_acc, best_nmi, best_ari, best_f1])
            time2 = time.time()
            print('top_k：', top_k,  ' item3 :', item3, ' item4 :',
                    item4, ' time :',
                    (time2 - time1))
            model_total_result_mean = np.array(model_total_result).mean(0)

            model_total_result_var = np.array(model_total_result).std(0)

            print('total_result: acc {:.4f}'.format(model_total_result_mean[0]),
                    ' nmi {:.4f}'.format(model_total_result_mean[1]),
                    ' ari {:.4f}'.format(model_total_result_mean[2]),
                    ' f1 {:.4f}'.format(model_total_result_mean[3]))
            print('total_result_var: acc {:.4f}'.format(model_total_result_var[0]),
                    ' nmi {:.4f}'.format(model_total_result_var[1]), ' ari {:.4f}'.format(model_total_result_var[2]),
                    ' f1 {:.4f}'.format(model_total_result_var[3]))

                    # loss_ae = F.mse_loss(x_hat, data)
                    # loss_w = F.mse_loss(z_hat, torch.spmm(adj, data))
                    # loss_a = F.mse_loss(adj_hat, adj.to_dense())
                    # loss_igae = loss_w + gamma_value * loss_a
                    # loss_kl = F.kl_div((q.log() + q1.log() + q2.log()) / 3, p, reduction='batchmean')
                    # loss = loss_ae + loss_igae + lambda_value * loss_kl
                    # print('{} loss: {}'.format(epoch, loss))
                    #
                    # optimizer1.zero_grad()
                    # loss.backward()
                    # optimizer1.step()

                    # kmeans = KMeans(n_clusters=n_clusters, n_init=20).fit(z_tilde.data.cpu().numpy())
                    #
                    # acc, nmi, ari, f1 = eva(label, kmeans.labels_, epoch)
                    # acc_reuslt.append(acc)
                    # nmi_result.append(nmi)
                    # ari_result.append(ari)
                    # f1_result.append(f1)
                    #
                    # if acc > original_acc:
                    #     original_acc = acc
                    #     torch.save(model1.state_dict(), final_model_save_path)
                # print('Acc: ',acc)