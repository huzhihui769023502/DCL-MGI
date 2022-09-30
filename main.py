import opt
import torch
import numpy as np
from DFCN import DFCN
from utils import setup_seed
from sklearn.decomposition import PCA
from load_data import LoadDataset, load_graph, construct_graph
from train import Train, acc_reuslt, nmi_result, f1_result, ari_result
import time
import scipy.stats
from info_nce import InfoNCE
from scipy.special import entr
from torch.optim import Adam
import torch.nn.functional as F
from sklearn.cluster import KMeans
from utils import adjust_learning_rate
from utils import eva, target_distribution

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
    indices = mine_nearest_neighbors2(feature, str_sim, t, T, JS,device)
    pos = indices[:, 1:2]
    neg = indices[:, -top_k:]

    pos_feature = feature[pos]
    pos_feature = torch.squeeze(pos_feature, 1)
    neg_feature = feature[neg]
    info_loss = InfoNCE(negative_mode='paired')
    contra_loss = info_loss(feature, pos_feature, neg_feature)

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

setup_seed(opt.args.seed)

print("network setting…")

if opt.args.name == 'usps':
    opt.args.k = 5
    opt.args.n_clusters = 10
    opt.args.n_input = 30
elif opt.args.name == 'hhar':
    opt.args.k = 5
    opt.args.n_clusters = 6
    opt.args.n_input = 50
elif opt.args.name == 'reut':
    opt.args.k = 5
    opt.args.n_clusters = 4
    opt.args.n_input = 100
elif opt.args.name == 'acm':
    opt.args.k = None
    opt.args.n_clusters = 3
    opt.args.n_input = 100
elif opt.args.name == 'dblp':
    opt.args.k = None
    opt.args.n_clusters = 4
    opt.args.n_input = 50
elif opt.args.name == 'cite':
    opt.args.k = None
    opt.args.n_clusters = 6
    opt.args.n_input = 100
else:
    print("error!")

### cuda
print("use cuda: {}".format(opt.args.cuda))
device = torch.device("cuda" if opt.args.cuda else "cpu")

### root
opt.args.data_path = 'data/{}.txt'.format(opt.args.name)
opt.args.label_path = 'data/{}_label.txt'.format(opt.args.name)
opt.args.graph_k_save_path = 'graph/{}{}_graph.txt'.format(opt.args.name, opt.args.k)
opt.args.graph_save_path = 'graph/{}_graph.txt'.format(opt.args.name)
opt.args.pre_model_save_path = 'model/model_pretrain/{}_pretrain.pkl'.format(opt.args.name)
opt.args.final_model_save_path = 'model/model_final/{}_final.pkl'.format(opt.args.name)

### data pre-processing
print("Data: {}".format(opt.args.data_path))
print("Label: {}".format(opt.args.label_path))

graph = ['acm', 'dblp', 'cite']
non_graph = ['usps', 'hhar', 'reut']

x = np.loadtxt(opt.args.data_path, dtype=float)
y = np.loadtxt(opt.args.label_path, dtype=int)

pca = PCA(n_components=opt.args.n_input)
X_pca = pca.fit_transform(x)
# plot_pca_scatter(args.name, args.n_clusters, X_pca, y)

dataset = LoadDataset(X_pca)

if opt.args.name in non_graph:
    construct_graph(opt.args.graph_k_save_path, X_pca, y, 'heat', topk=opt.args.k)

adj = load_graph(opt.args.k, opt.args.graph_k_save_path, opt.args.graph_save_path, opt.args.data_path).to(device)
data = torch.Tensor(dataset.x).to(device)
label = y

###  model definition
model1 = DFCN(ae_n_enc_1=opt.args.ae_n_enc_1, ae_n_enc_2=opt.args.ae_n_enc_2, ae_n_enc_3=opt.args.ae_n_enc_3,
             ae_n_dec_1=opt.args.ae_n_dec_1, ae_n_dec_2=opt.args.ae_n_dec_2, ae_n_dec_3=opt.args.ae_n_dec_3,
             gae_n_enc_1=opt.args.gae_n_enc_1, gae_n_enc_2=opt.args.gae_n_enc_2, gae_n_enc_3=opt.args.gae_n_enc_3,
             gae_n_dec_1=opt.args.gae_n_dec_1, gae_n_dec_2=opt.args.gae_n_dec_2, gae_n_dec_3=opt.args.gae_n_dec_3,
             n_input=opt.args.n_input,
             n_z=opt.args.n_z,
             n_clusters=opt.args.n_clusters,
             v=opt.args.freedom_degree,
             n_node=data.size()[0],
             device=device).to(device)

model2 = DFCN(ae_n_enc_1=opt.args.ae_n_enc_1, ae_n_enc_2=opt.args.ae_n_enc_2, ae_n_enc_3=opt.args.ae_n_enc_3,
             ae_n_dec_1=opt.args.ae_n_dec_1, ae_n_dec_2=opt.args.ae_n_dec_2, ae_n_dec_3=opt.args.ae_n_dec_3,
             gae_n_enc_1=opt.args.gae_n_enc_1, gae_n_enc_2=opt.args.gae_n_enc_2, gae_n_enc_3=opt.args.gae_n_enc_3,
             gae_n_dec_1=opt.args.gae_n_dec_1, gae_n_dec_2=opt.args.gae_n_dec_2, gae_n_dec_3=opt.args.gae_n_dec_3,
             n_input=opt.args.n_input,
             n_z=opt.args.n_z,
             n_clusters=opt.args.n_clusters,
             v=opt.args.freedom_degree,
             n_node=data.size()[0],
             device=device).to(device)

### training
print("Training on {}…".format(opt.args.name))
if opt.args.name == "usps":
    lr = opt.args.lr_usps
elif opt.args.name == "hhar":
    lr = opt.args.lr_hhar
elif opt.args.name == "reut":
    lr = opt.args.lr_reut
elif opt.args.name == "acm":
    lr = opt.args.lr_acm
elif opt.args.name == "dblp":
    lr = opt.args.lr_dblp
elif opt.args.name == "cite":
    lr = opt.args.lr_cite
else:
    print("missing lr!")


# this code is used to construct Mss
n = adj.size()[0]
dense_adj = adj.to_dense()
con_list = []
for i in range(n):
    con_index = list(np.array(torch.where(dense_adj[i] > 0)[0].cpu()))
    con_list.append(con_index)
    torch.where(dense_adj[i] > 0)[0]
str_sim = torch.zeros((n, n))
for i in range(n):
    for j in range(i + 1, n):
        union_indice = list(set(con_list[i]).union(set(con_list[j])))
        inter_indice = list(set(con_list[i]).intersection(set(con_list[j])))
        str_sim[i][j] = len(inter_indice) / len(union_indice)
        str_sim[j][i] = str_sim[i][j]
torch.save(str_sim, opt.args.name+'_str_sim.pth')

# Mss can be loaded directly
# str_sim = torch.load(opt.args.name+'_str_sim.pth')
# print('train_dy_start')


pre_model_save_path = opt.args.pre_model_save_path
final_model_save_path = opt.args.final_model_save_path
n_clusters = opt.args.n_clusters
original_acc = opt.args.n_clusters
gamma_value = opt.args.gamma_value
lambda_value = opt.args.lambda_value


for ld2 in [0.1]:
    for ld1 in [0.1]:
        top_k = 9
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
                x_hat, z_hat, adj_hat, z_ae, z_igae, _, _, _, z_tilde, pred = model1(data, adj)
            kmeans = KMeans(n_clusters=n_clusters, n_init=20)
            cluster_id = kmeans.fit_predict(z_tilde.data.cpu().numpy())
            model1.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)
            model2.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)
            eva(label, cluster_id, 'Initialization model','Initialization')

            model1_total_list = []
            model2_total_list = []
            best_acc = 0
            best_nmi = 0
            best_ari = 0
            best_f1 = 0
            original_acc = 0
 
            for epoch in range(opt.args.epoch+100):
                # if opt.args.name in use_adjust_lr:
                #     adjust_learning_rate(optimizer, epoch)
                x_hat1, z_hat1, adj_hat1, z_ae1, z_igae1, q1, q11, q21, z_tilde1, pred1 = model1(data, adj)
                x_hat2, z_hat2, adj_hat2, z_ae2, z_igae2, q2, q12, q22, z_tilde2, pred2 = model2(data, adj)

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

                acc1, nmi1, ari1, f11 = eva(label, kmeans1.labels_,'graph_model', epoch)
                acc_reuslt.append(acc1)
                nmi_result.append(nmi1)
                ari_result.append(ari1)
                f1_result.append(f11)

                acc2, nmi2, ari2, f12 = eva(label, kmeans2.labels_,'node_model', epoch)
                acc_reuslt.append(acc2)
                nmi_result.append(nmi2)
                ari_result.append(ari2)
                f1_result.append(f12)
                
                if acc1 > original_acc:
                    original_acc = acc1
                    best_acc = acc1
                    best_nmi = nmi1
                    best_f1 = f11
                    best_ari = ari1
                    torch.save(model1.state_dict(), final_model_save_path)
                if acc2 > original_acc:
                    original_acc = acc2
                    best_acc = acc2
                    best_nmi = nmi2
                    best_f1 = f12
                    best_ari = ari2
                    torch.save(model2.state_dict(), final_model_save_path)

                if epoch < epoch_dc:

                    JS2 = JS_divergence(res22, res23)
                    contr_dy_loss = dy_calculate_INfo_loss(z_tilde2, top_k, ae_z_y_pred2, str_sim, epoch,
                                                           200, JS2, device)

                    contr_center_loss = calculate_Info_loss2(pre_center, new_cluster_center)

                    loss_ae1 = F.mse_loss(x_hat1, data)
                    loss_w1 = F.mse_loss(z_hat1, torch.spmm(adj, data))
                    loss_a1 = F.mse_loss(adj_hat1, adj.to_dense())
                    loss_igae1 = loss_w1 + gamma_value * loss_a1
                    loss_kl1 = F.kl_div((q1.log() + q11.log() + q21.log()) / 3, p1, reduction='batchmean')
                    loss1 = loss_ae1 + loss_igae1 + lambda_value * loss_kl1
                    loss1_fin = loss1 + ld2 * contr_center_loss

                    loss_ae2 = F.mse_loss(x_hat2, data)
                    loss_w2 = F.mse_loss(z_hat2, torch.spmm(adj, data))
                    loss_a2 = F.mse_loss(adj_hat2, adj.to_dense())
                    loss_igae2 = loss_w2 + gamma_value * loss_a2
                    loss_kl2 = F.kl_div((q2.log() + q12.log() + q22.log()) / 3, p2, reduction='batchmean')
                    loss2 = loss_ae2 + loss_igae2 + lambda_value * loss_kl2
                    loss2_fin = loss2 + ld1 * contr_dy_loss

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
                        model1_train_mask = np.where(model1_clustering_score > 0.2)[0]
                        model2_train_mask = np.where(model2_clustering_score > 0.2)[0]
                    else:
                        model1_train_mask = np.ones(len(model1_total_list[0])).astype(int)
                        model2_train_mask = np.ones(len(model2_total_list[0])).astype(int)

                    JS2 = JS_divergence(res22, res23)
                    contr_dy_loss = dy_calculate_INfo_loss(z_tilde2[model1_train_mask], top_k,
                                                           ae_z_y_pred2[model1_train_mask],
                                                           str_sim[model1_train_mask][:, model1_train_mask], epoch,
                                                           200, JS2, device)

                    contr_center_loss = calculate_Info_loss2(pre_center, new_cluster_center)

                    loss_ae1 = F.mse_loss(x_hat1[model2_train_mask], data[model2_train_mask])
                    loss_w1 = F.mse_loss(z_hat1[model2_train_mask], torch.spmm(adj, data)[model2_train_mask])
                    loss_a1 = F.mse_loss(adj_hat1[model2_train_mask], adj.to_dense()[model2_train_mask])
                    loss_igae1 = loss_w1 + gamma_value * loss_a1
                    loss_kl1 = F.kl_div(
                        (q1.log()[model2_train_mask] + q11.log()[model2_train_mask] + q21.log()[model2_train_mask]) / 3,
                        p1[model2_train_mask], reduction='batchmean')
                    loss1 = loss_ae1 + loss_igae1 + lambda_value * loss_kl1
                    loss1_fin = loss1 + ld2 * contr_center_loss

                    loss_ae2 = F.mse_loss(x_hat2[model1_train_mask], data[model1_train_mask])
                    loss_w2 = F.mse_loss(z_hat2[model1_train_mask], torch.spmm(adj, data)[model1_train_mask])
                    loss_a2 = F.mse_loss(adj_hat2[model1_train_mask], adj.to_dense()[model1_train_mask])
                    loss_igae2 = loss_w2 + gamma_value * loss_a2
                    loss_kl2 = F.kl_div(
                        (q2.log()[model1_train_mask] + q12.log()[model1_train_mask] + q22.log()[model1_train_mask]) / 3,
                        p2[model1_train_mask], reduction='batchmean')
                    loss2 = loss_ae2 + loss_igae2 + lambda_value * loss_kl2
                    loss2_fin = loss2 + ld1 * contr_dy_loss

                    optimizer1.zero_grad()
                    loss1_fin.backward()
                    optimizer1.step()

                    optimizer2.zero_grad()
                    loss2_fin.backward()
                    optimizer2.step()

                pre_center = new_cluster_center
            model_total_result.append([best_acc, best_nmi, best_ari, best_f1])
        time2 = time.time()
        print('top_k：', top_k, ' ld2 :', ld2, ' ld1 :',
              ld1, ' time :',
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


