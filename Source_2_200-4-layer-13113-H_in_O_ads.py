import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD,Adam
from torch.utils.data import DataLoader, TensorDataset
import os
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from datetime import datetime
import time
import random
import matplotlib as mpl

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial']
mpl.rcParams['mathtext.fontset'] = 'custom'
mpl.rcParams['mathtext.it'] = 'Arial:italic'
mpl.rcParams['mathtext.rm'] = 'Arial'
mpl.rcParams['mathtext.bf'] = 'Arial:bold'

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class MLPmodel(nn.Module):
    def __init__(self, a, b, c, d):
        super(MLPmodel, self).__init__()

        self.hidden1 = nn.Sequential(
            nn.Linear(48, a), nn.BatchNorm1d(a), nn.ReLU())
        self.hidden2 = nn.Sequential(
            nn.Linear(a, b), nn.BatchNorm1d(b), nn.ReLU())
        self.hidden3 = nn.Sequential(
            nn.Linear(b, c), nn.BatchNorm1d(c), nn.ReLU())
        self.hidden4 = nn.Sequential(
            nn.Linear(c, d), nn.BatchNorm1d(d), nn.ReLU())

        self.regression = nn.Sequential(nn.Linear(d, 1), nn.ReLU())

    def forward(self, x):
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.hidden3(x)
        x = self.hidden4(x)
        return self.regression(x)

def data_load(filename, sheet_name):
    data = pd.read_excel(filename, sheet_name=sheet_name)
    result = data.values.tolist()
    nrows, ncols = data.shape
    
    structure_name = [x[0] for x in result]
    Layer1 = [x[1] for x in result]
    Layer2 = [x[2] for x in result]
    Layer3 = [x[3] for x in result]
    label = np.array([x[4] for x in result])
    features = np.array([x[5:ncols] for x in result])
    features_name = np.array(data.columns[5:ncols])
    
    return structure_name, Layer1, Layer2, Layer3, features, label, features_name

def batch_loader(dataX, dataY, batch_size=128, seed=42):
    minvalue = min(dataY)
    dataY = dataY - minvalue

    trainX, testX, trainY, testY = train_test_split(
        dataX, dataY, train_size=0.9, test_size=0.1, random_state=seed)
    
    train_dataset = TensorDataset(
        torch.from_numpy(trainX.astype(np.float32)),
        torch.from_numpy(trainY.astype(np.float32)))
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(seed))
    
    return train_loader, trainX, testX, trainY, testY

#Initialize MLP model
def initialize_mlp(neuron_number_layers,train_loader,learning_rate
                  ,folder_name,SEED):
    mlp = MLPmodel(neuron_number_layers[0]
                   ,neuron_number_layers[1]
                   ,neuron_number_layers[2]
                   ,neuron_number_layers[3]
                  )
    # print(mlp)
    state = train_mlp_original(mlp,train_loader,200,learning_rate,0)

    for i in range(100):
        if state == 1:
            break
        if state == 0:
            mlp = MLPmodel(neuron_number_layers[0]
                           ,neuron_number_layers[1]
                           ,neuron_number_layers[2]
                           ,neuron_number_layers[3]
                           )
            state = train_mlp_original(mlp,train_loader, 200,learning_rate,0)
    if state == 1:
        print('initialize succeed, continue')
    if state == 0:
        print('Initialize failed')

    model_save_path = f"{folder_name}/original_model_initialized_fn" + \
    f"{neuron_number_layers[0]}_{neuron_number_layers[1]}_" + \
    f"{neuron_number_layers[2]}_{neuron_number_layers[3]}_" + \
    f"{SEED}.pt"
    torch.save(mlp, model_save_path)
    return(model_save_path)

def train_mlp_original(mlp,train_loader,iteration,learning_rate,viewloss):
    optimizer=Adam(mlp.parameters(),
                    lr=learning_rate,
                    betas=(0.9, 0.999),
                    eps=1e-08,
                    weight_decay=0,
                    amsgrad=False)
    loss_func=nn.MSELoss()
    train_loss_all=[]
    for epoch in range(iteration):
        for step,(b_x,b_y) in enumerate(train_loader):
            output=mlp(b_x).flatten()
            train_loss=loss_func(output,b_y)
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
        train_loss_all.append(train_loss.item())
        # print(epoch,train_loss.item())
        # if epoch < 5 or epoch >= iteration - 5:
        #     print(epoch, train_loss.item())
        if train_loss.item() < 0.0001:
            break
    if viewloss == 1:
        plt.figure(figsize = (4,3),dpi = 300)
        plt.plot(train_loss_all,"b-")
        plt.title("train loss per Ite")
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss Value', fontsize=12)
        plt.tight_layout()
        plt.close()
        
    if train_loss_all[0] / train_loss_all[-1]+0.000001 < 1.3:
        state = 0
    else:
        state = 1
    return(state)

def train_mlp_classical_TL(model, train_loader, iteration
              , testX=None, testY=None 
              , trainX=None, trainY=None
              , minvalue=0, plot_freq=10):
    optimizer=Adam(model.parameters(),
                    lr=0.00005,
                    betas=(0.9, 0.999),
                    eps=1e-08,
                    weight_decay=0,
                    amsgrad=False)
    loss_func = nn.MSELoss()
    
    train_loss_all = []

    test_recorded_preds = []  
    train_recorded_preds = []  
    for epoch in range(iteration):
        for step,(b_x,b_y) in enumerate(train_loader):
            output=mlp(b_x).flatten()
            train_loss=loss_func(output,b_y)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
        train_loss_all.append(train_loss.item())

        if (epoch + 1) % plot_freq == 0 or epoch == iteration - 1:
            with torch.no_grad():
                test_tensor = torch.from_numpy(np.array(testX).astype(np.float32))
                test_pred = mlp(test_tensor).detach().numpy().flatten() + minvalue
                test_recorded_preds.append(test_pred)
                
                test_r2_epoch = r2_score(testY + minvalue, test_pred)
                test_mae_epoch = mean_absolute_error(testY + minvalue, test_pred)

                train_tensor = torch.from_numpy(np.array(trainX).astype(np.float32))
                train_pred = mlp(train_tensor).detach().numpy().flatten() + minvalue
                train_recorded_preds.append(train_pred)
                
                train_r2_epoch = r2_score(trainY + minvalue, train_pred)
                train_mae_epoch = mean_absolute_error(trainY + minvalue, train_pred)
        
        if train_loss.item() < 0.0001:
            break

    if test_recorded_preds:
        test_true = np.array(testY) + minvalue
        test_mean_pred = np.mean(test_recorded_preds, axis=0)


        train_true = np.array(trainY) + minvalue
        train_mean_pred = np.mean(train_recorded_preds, axis=0)

        train_r2 = r2_score(train_true, train_mean_pred)
        train_mae = mean_absolute_error(train_true, train_mean_pred)
        test_r2 = r2_score(test_true, test_mean_pred)
        test_mae = mean_absolute_error(test_true, test_mean_pred)

        print("\n=== Training Set Metrics ===")
        print(f"R2 Score: {train_r2:.4f}")
        print(f"MAE: {train_mae:.4f}")
        print("\n=== Test Set Metrics ===")
        print(f"R2 Score: {test_r2:.4f}")
        print(f"MAE: {test_mae:.4f}")

        plt.figure(figsize = (12,5),dpi = 300)

        plt.subplot(1, 2, 1)
        plt.scatter(train_true, train_mean_pred, alpha=0.6, color='green',s=100)
        plt.plot([min(train_true), max(train_true)], [min(train_true), max(train_true)], 
                'r--', lw=1)
        plt.xlabel("True Values (Train)",fontsize = 20)
        plt.ylabel("Predicted Values (Train)",fontsize = 20)
        plt.tick_params(axis='both', which='major', labelsize=20)  
        plt.title(f"Train Set ($R^2$={train_r2:.2f}, $MAE$={train_mae:.2f})"
                  , fontsize=20)

        plt.subplot(1, 2, 2)
        plt.scatter(test_true, test_mean_pred, alpha=0.6, color='blue',s=100)
        plt.plot([min(test_true), max(test_true)], [min(test_true), max(test_true)], 
                'r--', lw=1)
        plt.xlabel("True Values (Test)",fontsize = 20)
        plt.ylabel("Predicted Values (Test)",fontsize = 20)
        plt.tick_params(axis='both', which='major', labelsize=20)  
        plt.title(f"Test Set ($R^2$={test_r2:.2f}, $MAE$={test_mae:.2f})"
                  ,fontsize = 20)
        plt.tight_layout()
        plt.close()
    
    return train_r2, train_mae, test_r2, test_mae, train_true,\
           train_mean_pred, test_true, test_mean_pred, (train_loss_all[-1])

def plot_results(train_true_list, train_pred_list, test_true_list, test_pred_list, folder_name):
    def check_data_consistency(data_list, data_name):
        if len(data_list) == 0:
            return True, None
        
        reference = data_list[0]
        for i, data in enumerate(data_list[1:], 1):
            if not np.array_equal(data, reference):
                print(f"ERROR: The {i+1}-th data in {data_name} is not equal to the first data")
                print(f"First data: {reference}")
                print(f"{i+1}-th data: {data}")
                return False, None
        
        return True, reference
    
    train_true_consistent, train_true_ref = check_data_consistency(train_true_list, "train_true_list")
    if not train_true_consistent:
        print("Cannot compute average: data in train_true_list is inconsistent")
        return
    
    test_true_consistent, test_true_ref = check_data_consistency(test_true_list, "test_true_list")
    if not test_true_consistent:
        print("Cannot compute average: data in test_true_list is inconsistent")
        return
    
    print("All ground truth data groups are consistent, proceeding to compute average")

    avg_train_true = np.mean(np.array(train_true_list), axis=0)
    avg_train_pred = np.mean(np.array(train_pred_list), axis=0)
    avg_test_true = np.mean(np.array(test_true_list), axis=0)
    avg_test_pred = np.mean(np.array(test_pred_list), axis=0)
    
    train_pred_std = np.std(np.array(train_pred_list), axis=0)
    test_pred_std = np.std(np.array(test_pred_list), axis=0)
    
    try:
        df_train = pd.DataFrame({
            'avg_train_true': avg_train_true,
            'avg_train_pred': avg_train_pred,
            'train_pred_std': train_pred_std
        })
        
        train_csv_filename = f'{folder_name}/train_results_with_std.csv'
        df_train.to_csv(train_csv_filename, index=False)
        print(f"Train set save in: {train_csv_filename}")
        
        df_test = pd.DataFrame({
            'avg_test_true': avg_test_true,
            'avg_test_pred': avg_test_pred,
            'test_pred_std': test_pred_std
        })
        
        test_csv_filename = f'{folder_name}/test_results_with_std.csv'
        df_test.to_csv(test_csv_filename, index=False)
        print(f"Test set save in: {test_csv_filename}")

        df_combined = pd.concat([df_train.reset_index(drop=True)
                                 , df_test.reset_index(drop=True)], axis = 1)
        
        combined_csv_filename = f'{folder_name}/combined_results_with_std.csv'
        df_combined.to_csv(combined_csv_filename, index=False)
        print(f"Combined results save in: {combined_csv_filename}")
        
    except Exception as e:
        print(f"Saving ERROR: {e}")
    
    train_r2 = r2_score(avg_train_true, avg_train_pred)
    train_mae = mean_absolute_error(avg_train_true, avg_train_pred)
    test_r2 = r2_score(avg_test_true, avg_test_pred)
    test_mae = mean_absolute_error(avg_test_true, avg_test_pred)
    
    print(f'Train R2: {train_r2:.3f}, Train MAE: {train_mae:.3f}')
    print(f'Test R2: {test_r2:.3f}, Test MAE: {test_mae:.3f}')
    
    plt.figure(figsize=(4, 3), dpi=300)
    plt.scatter(avg_train_true, avg_train_pred, c='blue', alpha=0.6,
               label=f'Train ($R^2$={train_r2:.2f}, $MAE$={train_mae:.2f} eV)')
    plt.scatter(avg_test_true, avg_test_pred, c='red', alpha=0.6,
               label=f'Test ($R^2$={test_r2:.2f}, $MAE$={test_mae:.2f} eV)')
    
    min_val = min(np.min(avg_train_true), np.min(avg_test_true))
    max_val = max(np.max(avg_train_true), np.max(avg_test_true))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=1)
    
    plt.xlabel('DFT Calculated $E_{\mathrm{ads}}$(H@O)', fontsize=12)
    plt.ylabel('ML Predicted $E_{\mathrm{ads}}$(H@O)', fontsize=12)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f'{folder_name}/prediction_scatter.jpg', dpi=300)
    plt.close()

def generate_layer_neurons(input_dim, output_dim, num_layers, min_neurons=30):
    min_neurons = min(min_neurons, output_dim)
    layers = np.linspace(input_dim, min_neurons, num_layers + 1)[1:]
    
    neuron_counts = []
    prev = input_dim
    for n in layers:
        current = max(min_neurons, int(round(n)))
        current = min(current, prev)  
        neuron_counts.append(current)
        prev = current

    if output_dim < min_neurons:
        neuron_counts[-1] = output_dim
    else:
        neuron_counts[-1] = min_neurons
    
    return neuron_counts

# Source data
workbook = '13113-H_in_O_ads.xlsx'  # Input data excel
sheet = str("13113-H_in_O_ads")  # Input data excel sheet
structure_name_s, Layer1_s, Layer2_s, Layer3_s, features_s, label_s, features_name_s = data_load(workbook, sheet)
print('Source data size:', len(label_s))  
print('min: ', min(label_s),'max: ', max(label_s))

# Target data
workbook = '270_DFT_3090_50_prediction-Ea.xlsx'  # Input data excel
sheet = str("270_DFT_3090_50_prediction-Ea")  # Input data excel sheet
structure_name_t, Layer1_t, Layer2_t, Layer3_t, features_t, label_t, features_name_t = data_load(workbook, sheet)
# print('Feature list:', list(features_name_t))  
print('Target data size:', len(label_t))  
print('min: ', min(label_t),'max: ', max(label_t))

label_t_non_null_count = np.sum(~np.isnan(label_t))
label_t_null_count = np.sum(np.isnan(label_t))

print(f"Target value number: {label_t_non_null_count}")
print(f"Target nan number: {label_t_null_count}")
print(f"Target total number: {len(label_t)}")

SEED = 1
Round = 10
input_dim = 48
output_dim = 14  
num_layers = 4
set_seed(SEED)

folder_name = f"Source_2_{len(label_s)}_H_in_O_ads_{label_t_non_null_count}_{len(label_t)}" + \
               f"_Ea_{Round}_200_{num_layers}l_{output_dim}d"
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

print("Preparing source data...")
dataX_source = np.array(features_s)
dataY_source = np.array(label_s)
print('Source minvalue：', min(dataY_source)
      , 'Source maxvalue：', max(dataY_source))
train_loader_s, trainX_s, testX_s, trainY_s, testY_s = batch_loader(
    dataX_source, dataY_source, 128, SEED)

dataX_target = np.array(features_t)
dataY_target = np.array(label_t)
train_loader_t, trainX_t, testX_t, trainY_t, testY_t = batch_loader(
    dataX_target, dataY_target, 128, SEED)

total_start_time = time.time()
start_time = datetime.now()
print('Start time: ',start_time)

neuron_number_layers = generate_layer_neurons(input_dim, output_dim, num_layers)
print(neuron_number_layers)
print('Neuron Layers: ',len(neuron_number_layers))
print('Output Dimension: ',output_dim)

print("\nInitializing model...")
initialized_mlp = initialize_mlp(neuron_number_layers,train_loader_s,0.0001
                                ,folder_name,SEED)
mlp = torch.load(initialized_mlp)

print("\n=== Save initial feature ===")
with torch.no_grad():
    h_s_train_initial = mlp.hidden4(mlp.hidden3(
               mlp.hidden2(mlp.hidden1(torch.from_numpy(
               np.array(trainX_s).astype(np.float32)))))).numpy()
    h_s_test_initial = mlp.hidden4(mlp.hidden3(
              mlp.hidden2(mlp.hidden1(torch.from_numpy(
              np.array(testX_s).astype(np.float32)))))).numpy()

    h_t_train_initial = mlp.hidden4(mlp.hidden3(
               mlp.hidden2(mlp.hidden1(torch.from_numpy(
               np.array(trainX_t).astype(np.float32)))))).numpy()
    h_t_test_initial = mlp.hidden4(mlp.hidden3(
              mlp.hidden2(mlp.hidden1(torch.from_numpy(
              np.array(testX_t).astype(np.float32)))))).numpy()

h_s_initial = np.concatenate([h_s_train_initial, h_s_test_initial], axis=0)
h_t_initial = np.concatenate([h_t_train_initial, h_t_test_initial], axis=0)

y_s_initial = np.concatenate([trainY_s, testY_s], axis=0)
y_t_initial = np.concatenate([trainY_t, testY_t], axis=0)
y_s_initial += min(label_s)  
y_t_initial += min(label_t)  

print("\n=== Source property range ===")
print('min: ', min(y_s_initial),'max: ', max(y_s_initial))
print("\n=== Target property range ===")
print('min: ', np.nanmin(y_t_initial),'max: ', np.nanmax(y_t_initial))

np.save(f'{folder_name}/initial_hidden_features_source.npy', h_s_initial)
np.save(f'{folder_name}/initial_hidden_features_target.npy', h_t_initial)
np.save(f'{folder_name}/initial_source_labels.npy', y_s_initial)
np.save(f'{folder_name}/initial_target_labels.npy', y_t_initial)

print("\nSave success:")
print(f"- Source feature shape: {h_s_initial.shape}, Source property shape: {y_s_initial.shape}")
print(f"- Target feature shape: {h_t_initial.shape}, Target property shape: {y_t_initial.shape}")

y_t_initial_non_null_count = np.sum(~np.isnan(y_t_initial))
y_t_initial_null_count = np.sum(np.isnan(y_t_initial))

print(f"- Target initial value number: {y_t_initial_non_null_count}")
print(f"- Target initial nan number: {y_t_initial_null_count}")
print(f"- Target initial total number: {len(y_t_initial)}")

print("\nStarting training...")
train_true_list = []
train_pred_list = []
test_true_list = []
test_pred_list = []
train_r2_list = []
train_mae_list = []
test_r2_list = []
test_mae_list = []

for i in range(Round):
    print(f"\n=== Training Round {i+1}/{Round} ===")

    train_r2, train_mae, test_r2, test_mae, \
    train_true, train_pred, test_true, test_pred, train_losses = train_mlp_classical_TL(
        model = mlp, train_loader = train_loader_s, iteration = 200
        , testX=testX_s, testY=testY_s
        , trainX=trainX_s, trainY=trainY_s
        , minvalue=min(label_s)
        , plot_freq=2)
    
    train_true_list.append(train_true)
    train_pred_list.append(train_pred)
    test_true_list.append(test_true)
    test_pred_list.append(test_pred)
    train_r2_list.append(train_r2)
    train_mae_list.append(train_mae)
    test_r2_list.append(test_r2)
    test_mae_list.append(test_mae)

    torch.save(mlp, f"{folder_name}/model_round_{i+1}.pt")

print("\nPlotting results...")
plot_results(train_true_list, train_pred_list, test_true_list, test_pred_list, folder_name)

print("\nSaving the features of output layers")
with torch.no_grad():
    h_s = mlp.hidden4(mlp.hidden3(
        mlp.hidden2(mlp.hidden1(torch.from_numpy(
        dataX_source.astype(np.float32)))))).numpy()

    h_t = mlp.hidden4(mlp.hidden3(
        mlp.hidden2(mlp.hidden1(torch.from_numpy(
        dataX_target.astype(np.float32)))))).numpy()

np.save(f'{folder_name}/source_features.npy', h_s)
np.save(f'{folder_name}/target_features.npy', h_t)
np.save(f'{folder_name}/source_labels.npy', dataY_source)
np.save(f'{folder_name}/target_labels.npy', dataY_target)

print("\nSave success:")
print(f"- Source feature shape: {h_s.shape}, Source property shape: {dataY_source.shape}")
print(f"- Target feature shape: {h_t.shape}, Target property shape: {dataY_target.shape}")

dataY_target_non_null_count = np.sum(~np.isnan(dataY_target))
dataY_target_null_count = np.sum(np.isnan(dataY_target))

print(f"- Target final value number: {dataY_target_non_null_count}")
print(f"- Target final nan number: {dataY_target_null_count}")
print(f"- Target final total number: {len(dataY_target)}")

Y = np.load(f'{folder_name}/target_labels.npy')
print(f"- Target property min: {np.nanmin(Y)}, Target property max: {np.nanmax(Y)}")

print("\nAll done! Results saved to:", folder_name)
