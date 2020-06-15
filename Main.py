from ImagePreprocessing import *
import time

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode


### Path settings
data_address = "..\data" # Change this to load files
train_label_filename = "training_data_with_labels.csv"  # New Training 
val_label_filename = "val_data_with_labels.csv"
val_label_small_filename = "validate_labels_small.csv"

### Clear running log
f = open('running.txt','w')
f.truncate(0)
f.close()

### Load Training Set and Validation Set
training_dataset = SnakeDataSet(filename=os.path.join(data_address,train_label_filename),image_path=os.path.join(data_address,'train_images')) 
val_dataset = SnakeDataSet(filename=os.path.join(data_address,val_label_filename),image_path=os.path.join(data_address,'validate_images')) 

# There are 783 classes (0-782)

### Load Set to Loader
train_loader = DataLoader(training_dataset, batch_size=32, sampler = RandomSampler(training_dataset))
val_loader = DataLoader(val_dataset, batch_size=10000, sampler = RandomSampler(val_dataset))

# Random Seed
torch.manual_seed(2020)
np.random.seed(2020)

# Device
device = torch.device("cpu")
model = BasicNet().to(device)
model = model.float()

# Optimizer
optimizer = optim.Adadelta(model.parameters(), lr=1)

# Set your learning rate scheduler
scheduler = StepLR(optimizer, step_size=1, gamma=0.7)

# Epochs
epochs = 15

# Training loop
train_loss = []
val_loss = []
for epoch in range(1, epochs+1):
    train(model, device, train_loader, optimizer, epoch, 5)
    train_loss.append(test(model, device, train_loader))
    val_loss.append(test(model, device, val_loader))
    scheduler.step()    # learning rate scheduler

torch.save(model.state_dict(), "cnn_model_epoch15_lr1_batchsize32.pt")
f = open("cnn_model_epoch15_lr1_batchsize32_accuracy.txt","w")
f.write('Training Loss: ' + ' '.join(str(_) for _ in train_loss) + '\n')
f.write('Validation Loss: ' + ' '.join(str(_) for _ in val_loss) + '\n')
f.close()

