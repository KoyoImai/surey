import os
import torch

import matplotlib.pyplot as plt


"""
クラス・タスクなどの粒度を変えながらSVDしたい
"""



# タスク毎に特徴量，ラベルを分割
def split_by_task(features: torch.Tensor,
                  labels: torch.Tensor,
                  cls_per_task: int):
    """
    features : [N, D]  – 抽出済み特徴
    labels   : [N]     – 0,1,2,... のグローバルクラス ID
    cls_per_task : int – 1タスクに含まれるクラス数
    -------------------------------------------------
    戻り値:
        task_features : List[Tensor]  # タスク t の全特徴行列
        task_labels   : List[Tensor]  # 〃 に対応するラベル
        label_list4task : List[List[int]]  # 各タスクのクラス ID
    """
    # ---------- タスク ID を計算 ---------------
    # “クラス ID // cls_per_task” がそのサンプルのタスク番号になる
    task_id = labels // cls_per_task          # [N]
    num_tasks = int(task_id.max().item()) + 1

    task_features, task_labels, label_list4task = [], [], []

    for t in range(num_tasks):
        # ① そのタスクに属するクラス集合
        cls_start = t * cls_per_task
        cls_end   = (t + 1) * cls_per_task
        class_ids = list(range(cls_start, cls_end))
        label_list4task.append(class_ids)

        # ② マスクして特徴を抽出
        mask = (task_id == t)
        if mask.sum() == 0:
            continue
        task_features.append(features[mask])   # [n_t, D]
        task_labels.append(labels[mask])       # [n_t]

    return task_features, task_labels, label_list4task


def svd(opt, features, labels, plot=True, max_k=20, cls_per_task=1, name="practice", threshold=0.9):
    
    """
        features:[num_data, embed_dim]のtensor
        labels  :[num_data]のtensor
    """
    
    task_feats, task_labs, cls_sets = split_by_task(features, labels, cls_per_task)
    
    if cls_per_task == 1:
        cls_or_task = "cls"
    else:
        cls_or_task = "task"

    plt.figure()

    label_list = []
    k_list = []
    for t, (z, y) in enumerate(zip(task_feats, task_labs)):
        
        # --- ここで SVD / intrinsic_dim など ----
        print(f"Task {t} → classes {cls_sets[t]}")
        
        label_list.append(t)

        # 中心化
        z_centered = z - z.mean(dim=0, keepdim=True)

        # 共分散行列
        cov = z_centered.T @ z_centered / (z.size(0) - 1)  # [d, d]

        # 共分散行列に対してSVD
        U, S, V = torch.svd(cov)  # Sは固有値と一致

        # 累積寄与率 α_k の計算
        total_var = S.sum()
        alpha_k = torch.cumsum(S, dim=0) / total_var

        k_dim = int((alpha_k > threshold).nonzero(as_tuple=True)[0][0].item()) + 1
        print(f"  - α_k > {threshold:.2f} となる最小k: {k_dim}")
        k_list.append(k_dim)

        # プロット
        if plot:
            k = min(max_k, len(alpha_k))
            plt.plot(range(1, k+1), alpha_k[:k].cpu().numpy(), label=f'{cls_or_task} {t} (k={k_dim})')
    
    if plot:

        plt.xlabel('number of k')
        plt.ylabel('alpha_k')
        plt.title(f'{opt.method}')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        if opt.save_path:
            save_path = f"{opt.save_path}/{name}/"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            file_path = f"{save_path}/model{opt.target_task}_{cls_or_task}{label_list}.pdf"
            plt.savefig(file_path)
        
        if plot:
            plt.xlim([0, 1.0])
            plt.ylim([0, max_k+0.5])
            plt.show()
        else:
            plt.clf()

    return None











# # クラス毎に特異値分解を実行
# def svd(opt, features, labels, plot=True, max_k=20, cls_per_task=1):
    
#     """
#         features:[num_data, embed_dim]のtensor
#         labels  :[num_data]のtensor
#     """
    
#     # クラスリストの作成
#     label_list = sorted(torch.unique(labels).tolist())

#     # タスク毎のクラスリスト(opt.cls_per_taskを使いたい)
#     task_id = labels // opt.cls_per_task
#     label_list4task = []

#     plt.figure()

#     for label in label_list:

#         index = (labels == label).nonzero(as_tuple=True)[0]
#         # print("index: ", index)
        
#         z = features[index]  # [n, d]

#         if z.size(0) < 2:
#             continue  # サンプル数が1以下のクラスはスキップ

#         # 中心化
#         z_centered = z - z.mean(dim=0, keepdim=True)
#         # print("z_centered.shape: ", z_centered.shape)

#         cov = z_centered.T @ z_centered / (z.size(0) - 1)  # [d, d]
#         # print("cov.shape: ", cov.shape)

#         # 共分散行列に対してSVD
#         U, S, V = torch.svd(cov)  # Sは固有値と一致

#         # 累積寄与率 α_k の計算
#         total_var = S.sum()
#         alpha_k = torch.cumsum(S, dim=0) / total_var

#         # プロット
#         if plot:
#             k = min(max_k, len(alpha_k))
#             plt.plot(range(1, k+1), alpha_k[:k].cpu().numpy(), label=f'class {label}')
    
#     if plot:

#         plt.xlabel('k')
#         plt.ylabel('alpha_k')
#         plt.title('cov SVD alpha_k')
#         plt.grid(True)
#         plt.legend()
#         plt.tight_layout()

#         if opt.save_path:
#             os.makedirs(os.path.dirname(opt.save_path), exist_ok=True)
#             file_path = f"{opt.save_path}/model{opt.target_task}_label{label_list}.pdf"
#             plt.savefig(file_path)
        
#         if plot:
#             plt.show()
#         else:
#             plt.clf()

#     return None