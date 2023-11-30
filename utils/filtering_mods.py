import geopandas as gpd
import numpy as np
from PIL import Image
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
torch.manual_seed(0)
np.random.seed(0)

#- defining and creating our filtering models 
class binary_search_models:
    def __init__(self, type, tree,  dataObj,  input_dim=8, hidden_dim=4, batch_size=128,  output_dim=1, model_save_path='./utils/models/', learning_rate=0.01, epochs=20):
        self.epochs = epochs
        self.lr = learning_rate
        self.tree = tree
        self.data = dataObj
        self.training_indx = self.tree._find_train_test_indxs(0.1)
        self.testing_indx = self.tree._find_train_test_indxs(0.05)
        self.model_save_path = model_save_path
        self.batch_size = batch_size
        if type=='log':
            self.model = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(),nn.Linear(hidden_dim, output_dim), nn.Sigmoid())
            self.criterion = nn.BCELoss()
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
            self._train_log_filter(1000, False)
        if type=='cnn':
            self.model = cnn_filter()
            self.criterion = nn.CrossEntropyLoss()
            self.optimizer = torch.optim.Adam(self.model.parameters(),lr=0.0001)
            self.train_dataset = filter_dataset(self.training_indx)
            self.test_dataset = filter_dataset(self.testing_indx)
            self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
            self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=True)
            self._train_cnn_filter(35, False)
    def _train_cnn_filter(self, epochs, save):
        max_acc = 0
        for epoch in range(epochs):
            self.model.train()
            ls = []
            for j,(x_train,y_train) in enumerate(self.train_loader):
                output = self.model(x_train)
                y_train = torch.flatten(y_train)
                loss = self.criterion(output,y_train)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                ls.append(loss.item())
            mean_loss = np.mean(ls)
            if epoch % 5 == 0:
                accs = []
                for x_test,y_test in self.test_loader:
                    self.model.eval()
                    output = self.model(x_test)
                    y_test = torch.flatten(y_test)
                    loss = self.criterion(output,y_test)
                    pred_labels = torch.argmax(output, 1).detach().numpy()
                    total_test = y_test.size(0)
                    acc = (np.sum(pred_labels == y_test.detach().numpy())*100)/total_test
                    accs.append(acc)
                mean_acc = np.mean(accs)
                if mean_acc>max_acc:
                    max_acc= mean_acc
                    if save:
                        torch.save(self.model.state_dict(), 'cnn_filter.pt')
        return 'Done'
        
    def _train_log_filter(self, epochs, save):
        X_train, y_train, x_test, y_test = self.data._get_log_data(True, self.training_indx , self.testing_indx)
        max_acc = 0
        for epoch in range(epochs):
            self.model.train()
            x = X_train
            labels = y_train
            self.optimizer.zero_grad() 
            outputs = self.model(x)
            loss = self.criterion(torch.squeeze(outputs), labels) 
            loss.backward()
            self.optimizer.step()
            if epoch % 100 == 0:
                self.model.eval()
                outputs_test = torch.squeeze(self.model(x_test))
                loss_test = self.criterion(outputs_test, y_test)
                predicted_test = outputs_test.round().detach().numpy()
                total_test = y_test.size(0)
                correct_test = np.sum(predicted_test == y_test.detach().numpy())
                acc = (100 * correct_test/total_test)
                if max_acc < acc:
                    max_acc = acc
                    torch.save(self.model.state_dict(), 'log_filter.pt')
            self.model.eval()
            # self.model.load_state_dict(torch.load('log_filter.pt'))
        return 'Done'

#- dataset for cnn filtering for training and filtering 
class filter_dataset(Dataset):
    def __init__(self, indices, metadata_path='./buildings/samples_bld.geojson', path_images= '/Users/ridhaalkhabaz/Documents/mlds/images/', path_labels='/Users/ridhaalkhabaz/Documents/mlds/labels/'):
        self.indices = indices
        self.data = []
        self.imgs_path = path_images
        self.labs_path = path_labels
        self.df_meta = gpd.read_file(metadata_path)
        for indx in indices:
            # path_to_label = self.labs_path+'label_'+str(indx)+'.png'
            path_to_image = self.imgs_path+'imag_'+str(indx)+'.png'
            # mask = np.array(Image.open(path_to_label))
            # label = numIslands(mask)
            label_ind = self.df_meta.iloc[indx]['FID']
            label = 1 if label_ind > 0 else 0
            self.data.append([path_to_image, label])
        self.img_dim = (224, 224)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        img_path, class_name = self.data[idx]
        img = np.array(Image.open(img_path))
        # img = np.expand_dims(img, axis=0)
        class_id = class_name
        img_tensor = torch.tensor(img).to(torch.float32)
        img_tensor = torch.unsqueeze(img_tensor, 0)
        class_id = torch.tensor([class_id])
        return img_tensor, class_id
#- our cnn-based filter 
class cnn_filter(nn.Module):
    ''' Models a simple Convolutional Neural Network'''
    def __init__(self, input_channels=1, cnn_channels=2, cnn_channels_second=8, output_classes=1, kernel_size=2):
        super(cnn_filter, self).__init__()
        self.cnn = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.Conv2d(1, 2, 2),
            nn.ReLU(),
            nn.BatchNorm2d(2),  
            nn.MaxPool2d(2, 2), 
            nn.Conv2d(2, 8, 2),
            nn.ReLU(),
            nn.BatchNorm2d(8),  
            nn.MaxPool2d(2, 2),
            nn.Conv2d(8, 16, 2),
            nn.ReLU(),
            nn.BatchNorm2d(16),  
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 2),
            nn.ReLU(),
            nn.BatchNorm2d(32),  
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 2),
            nn.ReLU(),
            nn.BatchNorm2d(64),  
            nn.MaxPool2d(2, 2),
            # nn.Conv2d(64, 128, 2),
            # nn.ReLU(),
            # nn.BatchNorm2d(128),  
            # nn.MaxPool2d(2, 2),
            nn.Flatten(start_dim=1),
        )
        self.fc1 = nn.Linear(2304, 128)
        self.dropout = nn.Dropout(p=0.4)
        self.fc2 = nn.Linear(128, 2)
        self.act = nn.Softmax()
    def forward(self, x):
        x = self.cnn(x)
        x = F.relu(self.dropout(self.fc1(x)))
        x = self.act(self.fc2(x))
        return x

#- returns a subset of the 
# def find_interestings_subtrees(treeObj, model, model_type, dataObj, ratio, threshold):
#     keys = []
#     for key in list(treeObj.tree.keys()):
#         indices = treeObj._get_subtree_sample_indx(key, ratio)
#         if model_type=='log':
#             X = dataObj._get_log_data(False, indices, None)
#             output = torch.squeeze(model(X))
#             pred_ext = output.round().detach().numpy()
#             tot_exm = X.size(0)
#             ext = pred_ext.sum()/tot_exm
#             if ext >=threshold:
#                 keys.append(key)
#         if model_type=='cnn':
#             preds = []
#             for i in indices:
#                 if i > 65998:
#                     continue;
#                 img = dataObj._img_to_tensor(i)
#                 output = model(img)
#                 pred_ext = torch.argmax(output, 1).detach().numpy()
#                 preds.append(pred_ext[0])
#             ext = np.mean(preds)
#             if ext >=threshold:
#                 keys.append(key)
#     list_indx  = []
#     for k in keys:
#         list_indx.extend(treeObj._get_subtree_indxs(k))
#     return list_indx
# def find_interestings_subtrees(treeObj, model, model_type, dataObj, ratio, threshold, FULL=False):
#     keys = []
#     if FULL:
#         indices = [i for i in range(dataObj.num_recs-1)] # some data corruption issues 
#         if model_type=='log':
#             X = dataObj._get_log_data(False, indices, None)
#             output = torch.squeeze(model(X))
#             pred_ext = output.round().detach().numpy()
#             out.round().detach().numpy()
#             return np.where(res==1)[0]
#         if model_type=='cnn':
#             preds = []
#             for i in indices:
#                 img = dataObj._img_to_tensor(i)
#                 output = model(img)
#                 pred_ext = torch.argmax(output, 1).detach().numpy()
#                 if pred_ext == 1:
#                     preds.append(i)
#             return preds 
            
#     for key in list(treeObj.tree.keys()):
#         indices = treeObj._get_subtree_sample_indx(key, ratio)
#         if model_type=='log':
#             X = dataObj._get_log_data(False, indices, None)
#             output = torch.squeeze(model(X))
#             pred_ext = output.round().detach().numpy()
#             print(pred_ext)
#             tot_exm = X.size(0)
#             ext = pred_ext.sum()/tot_exm
#             if ext >=threshold:
#                 keys.append(key)
#         if model_type=='cnn':
#             preds = []
#             for i in indices:
#                 if i > 65998:
#                     continue;
#                 img = dataObj._img_to_tensor(i)
#                 output = model(img)
#                 pred_ext = torch.argmax(output, 1).detach().numpy()
#                 preds.append(pred_ext[0])
#             ext = np.mean(preds)
#             if ext >=threshold:
#                 keys.append(key)
#     list_indx  = []
#     for k in keys:
#         list_indx.extend(treeObj._get_subtree_indxs(k))
#     return list_indx 
def find_interestings_subtrees(treeObj, model, model_type, dataObj, ratio, threshold, FULL=False):
    keys = []
    if FULL:
        indices = [i for i in range(dataObj.num_recs-1)] # some data corruption issues
        # print(len(indices))
        if model_type=='log':
            X = dataObj._get_log_data(False, indices, None)
            output = torch.squeeze(model(X))
            pred_class= output.round().detach().numpy()
            return np.where(pred_class==1)[0]
        if model_type=='cnn':
            preds = []
            for i in indices:
                img = dataObj._img_to_tensor(i)
                output = model(img)
                pred_ext = torch.argmax(output, 1).detach().numpy()
                if pred_ext == 1:
                    preds.append(i)
            return preds 
            
    for key in list(treeObj.tree.keys()):
        indices = treeObj._get_subtree_sample_indx(key, ratio)
        if model_type=='log':
            X = dataObj._get_log_data(False, indices, None)
            output = torch.squeeze(model(X))
            pred_ext = output.round().detach().numpy()
            # print(pred_ext)
            tot_exm = X.size(0)
            ext = pred_ext.sum()/tot_exm
            if ext >=threshold:
                keys.append(key)
        if model_type=='cnn':
            preds = []
            for i in indices:
                if i > 65998:
                    continue;
                img = dataObj._img_to_tensor(i)
                output = model(img)
                pred_ext = torch.argmax(output, 1).detach().numpy()
                preds.append(pred_ext[0])
            ext = np.mean(preds)
            if ext >=threshold:
                keys.append(key)
    list_indx  = []
    for k in keys:
        list_indx.extend(treeObj._get_subtree_indxs(k))
    return list_indx 