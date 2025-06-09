import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from torch import nn
import numpy as np
import torch
import os
import random
from tqdm import tqdm as tqdm
from IPython import display

from models.vgg import VGG_A
from models.vgg import VGG_A_BatchNorm 
from data.loaders import get_cifar_loader

# ## Constants (parameters) initialization
device_id = [0,1]
num_workers = 4
batch_size = 128

# add our package dir to path 
module_path = os.path.dirname(os.getcwd())
home_path = module_path
figures_path = os.path.join(home_path, 'reports', 'figures')
models_path = os.path.join(home_path, 'reports', 'models')

# Make sure you are using the right device.
device_id = device_id
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
device = torch.device("cuda:{}".format(1) if torch.cuda.is_available() else "cpu")
print(device)
print(torch.cuda.get_device_name(1))



# Initialize your data loader and
# make sure that dataloader works
# as expected by observing one
# sample from it.
train_loader = get_cifar_loader(train=True)
val_loader = get_cifar_loader(train=False)
for X,y in train_loader:
    ## --------------------
    # Add code as needed
    print(X.shape)
    
    #
    #
    #
    ## --------------------
    break




# Set a random seed to ensure reproducible results
def set_random_seeds(seed_value=0, device='cpu'):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    if device != 'cpu': 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# We use this function to complete the entire
# training process. In order to plot the loss landscape,
# you need to record the loss value of each step.
# Of course, as before, you can test your model
# after drawing a training round and save the curve
# to observe the training
def train(model, optimizer, criterion, train_loader, val_loader, scheduler=None, epochs_n=100,root=''):
    os.makedirs(root, exist_ok=True)
    model.to(device)
    learning_curve = [np.nan] * epochs_n
    train_accuracy_curve = [np.nan] * epochs_n
    val_accuracy_curve = [np.nan] * epochs_n
    max_val_accuracy = 0
    max_val_accuracy_epoch = 0

    batches_n = len(train_loader)
    losses_list = []
    #grads = []
    lr_list = []
    for epoch in tqdm(range(epochs_n), unit='epoch'):
        correct_train = 0
        total_train = 0
        if scheduler is not None:
            scheduler.step()
        model.train()

        loss_list = []  # use this to record the loss value of each step
        #grad = []  # use this to record the loss gradient of each step
        learning_curve[epoch] = 0  # maintain this to plot the training curve

        for data in train_loader:
            x, y = data
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            prediction = model(x)
            loss = criterion(prediction, y)
            # You may need to record some variable values here
            # if you want to get loss gradient, use
            
            ## --------------------
            # Add your code
            loss.backward()
            #sgrad = model.classifier[4].weight.grad.clone()
            #sgrad = model.classifier[4].weight.grad.detach().cpu().clone().numpy()
            
            #grad.append(sgrad)
            loss_list.append(loss.item())
            optimizer.step()
            _, predicted = torch.max(prediction.data, 1)
            correct_train += (predicted == y).sum().item()
            total_train += y.size(0)

            learning_curve[epoch] += loss.item()
            
            

        losses_list.append(loss_list)
        #grads.append(grad)
        #display.clear_output(wait=True)
       # f, axes = plt.subplots(1, 2, figsize=(15, 3))

        train_accuracy_curve[epoch] = correct_train / total_train

        learning_curve[epoch] /= batches_n
        #axes[0].plot(learning_curve)

        # Test your model and save figure here (not required)
        # remember to use model.eval()
        ## --------------------
        # Add code as needed
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in val_loader:
                
                images, labels = data
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        epoch_acc=correct/total
        val_accuracy_curve[epoch] = correct / total
        if epoch_acc>=max_val_accuracy:
            max_val_accuracy=epoch_acc
            max_val_accuracy_epoch=epoch  
            save_path = os.path.join(root, 'best_model.pth')
            torch.save(model.state_dict(), save_path)

    avg_loss_per_epoch = [np.mean(epoch_losses) for epoch_losses in losses_list]
    
    plt.figure(figsize=(8, 6))
    plt.plot(avg_loss_per_epoch, label='Training Loss', color='blue')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.grid(True)
    os.makedirs(root, exist_ok=True)
    plt.savefig(os.path.join(root, 'loss_curve.png'))
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(range(epochs_n), val_accuracy_curve, label='Val Accuracy', color='orange')
    plt.scatter(max_val_accuracy_epoch, max_val_accuracy, color='red', label=f'Max Acc = {max_val_accuracy:.4f} @ Epoch {max_val_accuracy_epoch}')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy Curve")
    plt.legend()
    plt.grid(True) 
    os.makedirs(root, exist_ok=True)
    save_name = os.path.join(root, f"val_accuracy_curve_epoch.png")
    plt.savefig(save_name)
    plt.close()    
        #
        #
        #
        ## --------------------
        
    
    return losses_list#grads


# Train your model
# feel free to modify
os.makedirs("VGG_A", exist_ok=True)
epo = 20
loss_save_path = 'VGG_A'
#grad_save_path = 'VGG_A'



learning_rates = [1e-3, 2e-3, 1e-4, 5e-4]

all_model_losses = []  # shape: (num_models, num_steps)

for lr in learning_rates:
    print(f"Training with learning rate: {lr}")
    set_random_seeds(seed_value=2020, device=device)
    model = VGG_A()
    model._init_weights()
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    criterion = nn.CrossEntropyLoss()
    losses_list= train(model, optimizer, criterion, train_loader, val_loader, None,epo,'VGG_A')
    
    step_losses = [l for epoch_losses in losses_list for l in epoch_losses]
    print(len(step_losses))
    all_model_losses.append(step_losses)
all_model_losses = np.array([np.array(losses) for losses in all_model_losses])
max_len = min(map(len, all_model_losses))  
all_model_losses = np.array([losses[:max_len] for losses in all_model_losses])
all_model_losses = np.array(all_model_losses)#(2,7820)
max_curve = np.max(all_model_losses, axis=0)
min_curve = np.min(all_model_losses, axis=0)

all_model_losses = []
for lr in learning_rates:
    print(f"Training with learning rate: {lr}")
    set_random_seeds(seed_value=2020, device=device)
    model = VGG_A_BatchNorm()
    model._init_weights()
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    criterion = nn.CrossEntropyLoss()
    losses_list= train(model, optimizer, criterion, train_loader, val_loader, None,epo,'VGG_A_B')
    
    step_losses = [l for epoch_losses in losses_list for l in epoch_losses]
    print(len(step_losses))
    all_model_losses.append(step_losses)
all_model_losses = np.array([np.array(losses) for losses in all_model_losses])
max_len = min(map(len, all_model_losses))  
all_model_losses = np.array([losses[:max_len] for losses in all_model_losses])
all_model_losses = np.array(all_model_losses)#(2,7820)
max_curve_B = np.max(all_model_losses, axis=0)
min_curve_B = np.min(all_model_losses, axis=0)


#np.savetxt(os.path.join(loss_save_path, 'loss.txt'), loss, fmt='%s', delimiter=' ')
#np.savetxt(os.path.join(grad_save_path, 'grads.txt'), grads, fmt='%s', delimiter=' ')
#np.save(os.path.join(grad_save_path, 'grads.npy'), np.array(grads, dtype=object))
# Maintain two lists: max_curve and min_curve,
# select the maximum value of loss in all models
# on the same step, add it to max_curve, and
# the minimum value to min_curve

# Use this function to plot the final loss landscape,
# fill the area between the two curves can use plt.fill_between()
def plot_loss_landscape():
    ## --------------------
    # Add your code
    stride = 40  # 每隔 40 个 step 取一次

    min_downsampled = min_curve[::stride]
    max_downsampled = max_curve[::stride]


    min_downsampled_B = min_curve_B[::stride]
    max_downsampled_B = max_curve_B[::stride]


    steps = np.arange(len(min_curve))[::stride]

    plt.plot(steps, min_downsampled, label='VGG without BN min', color='blue')
    plt.plot(steps, max_downsampled, label='VGG without BN max', color='blue')
    plt.fill_between(steps, min_downsampled, max_downsampled, alpha=0.3, color='gray')


    plt.plot(steps, min_downsampled_B, label='VGG with BN min', color='orange')
    plt.plot(steps, max_downsampled_B, label='VGG with BN max', color='orange')
    plt.fill_between(steps, min_downsampled_B, max_downsampled_B, alpha=0.3, color='orange')


    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.title("Min-Max Loss Curves for Two Settings")
    plt.legend()
    plt.grid(True)
    
    plt.savefig(os.path.join(figures_path, "loss_landscape.png"))
    plt.show()
    plt.close()


plot_loss_landscape()
 
