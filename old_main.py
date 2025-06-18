import torch
from AttackPack import bridge, visualize_entry
import matplotlib.pyplot as plt
import numpy as np


torch.manual_seed(11)

# device = torch.device('mps' if torch.mps.is_available() else 'cpu')
device = torch.device('cpu')

# Load ECG dataset
# test_set = bridge.create_dataset('vpunfold/data/ecg_test.mat', device=device, dtype=torch.float)
test_set = bridge.create_dataset('vpunfold/data/mitdb_filt35_w300adapt_ds2_float.mat', device=device, dtype=torch.float)

reduced_set = [test_set[i] for i in range(5)]

# hyperparameters (best of balanced run)
batch_size = 4096
lr = 1e-2
epoch = 50
params_init = [1.0, 0.0]
n_vp = 8
n_hiddens = [8]
vp_penalty = 0.0

name, model, criterion = bridge.create_model("vpnet_hermite2_rr", [batch_size, lr, epoch, params_init, n_vp, n_hiddens, vp_penalty], test_set[0], torch.float, device)
name, model_good, criterion = bridge.create_model("vpnet_hermite2_rr", [batch_size, lr, epoch, params_init, n_vp, n_hiddens, vp_penalty], test_set[0], torch.float, device)
model.eval()

sample, rr, label = test_set[0]

visualize_entry(test_set[0])

print(name)
print(model)
print(criterion)

# print(sample)

# best_name = "vpnet-hermite2_b4096_lr0.01_e50_p1.0-0.0_n4_h16-16_r0.1"
best_name = "vpnet-hermite2-rr_b4096_lr0.01_e50_p1.0-0.0_n8_h8_r0.0.pt"
best_name = "vpnet/models_vpnet/" + best_name
best_state_dict = torch.load(best_name, weights_only=True, map_location=device)

model_good.load_state_dict(best_state_dict)


print(model(sample, rr))
print(model_good(sample, rr))
print(label)


# visualize an entry


# attack with fgsm
t_loader = torch.utils.data.DataLoader(test_set, num_workers=0)

eps = [0, .05, .1, .15, .2, .25, .3]
accs = []
exes = []
for e in eps:
    acc, ex = attack_model_grad(model_good, criterion, device, t_loader, e)
    accs.append(acc)
    exes.append(ex)

plt.figure(figsize=(5,5))
plt.plot(eps, accs, "*-")
plt.yticks(np.arange(0, 1.1, step=0.1))
plt.xticks(np.arange(0, .35, step=0.05))
plt.title("Accuracy vs Epsilon")
plt.xlabel("Epsilon")
plt.ylabel("Accuracy")
plt.show()

# Plot several examples of adversarial samples at each epsilon
cnt = 0
plt.figure(figsize=(8,10))
for i in range(len(eps)):
    for j in range(len(exes[i])):
        cnt += 1
        plt.subplot(len(eps),len(exes[0]),cnt)
        plt.xticks([], [])
        plt.yticks([], [])
        if j == 0:
            plt.ylabel(f"Eps: {eps[i]}", fontsize=14)
        orig,adv,ex = exes[i][j]
        plt.title(f"{orig} -> {adv}")
        plt.imshow(ex, cmap="gray")
plt.tight_layout()
plt.show()

# get baseline
# default_bl = []
# good_bl = []
# for sample, rr, label in reduced_set:
#     print(model(sample))
#     print(model_good(sample))
#     print(model_good(sample, rr))
#     print(label)
#     def_res = model(sample, rr)
#     good_res = model_good(sample, rr)
#     default_bl.append(def_res)
#     good_bl.append(good_res)

# print(default_bl)
# print(good_bl)



