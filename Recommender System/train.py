import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split

from model import ModelClass
from utils import RecommendationDataset

import matplotlib.pyplot as plt

def calculate_valid(valid_loader, model, n_ratings_pred):

    model.eval()

    cost_valid = 0
    with torch.no_grad():
        for users, items, ratings in valid_loader:
            ratings_pred = model(users, items)
            loss = criterion(ratings_pred,ratings)
            cost_valid += loss.item() * len(ratings)
        cost_valid /= n_ratings_pred
        print('valid cost: {:6f}'.format(cost_valid))
    return cost_valid

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='2021 AI Final Project')
    parser.add_argument('--save-model', default='model.pt', help="Model's state_dict")
    parser.add_argument('--dataset', default='./data', help='dataset directory')
    parser.add_argument('--batch-size', default=64, help='train loader batch size')

    args = parser.parse_args()

    # load dataset in train folder
    train_data = RecommendationDataset(f"{args.dataset}/ratings.csv", train=True)
    num_train = int(len(train_data) * 0.9)
    split_train, split_valid = random_split(train_data,[num_train, len(train_data)-num_train])
    train_loader = DataLoader(split_train, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(split_valid, batch_size=args.batch_size)
    
    n_ratings_pred = len(split_valid)
    n_users, n_items, n_ratings = train_data.get_datasize()
    # print(n_users)
    # print(n_items)
    # print(n_ratings)
    
    # instantiate model
    model = ModelClass(n_users, n_items, rank=28)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay = 1e-5)
    criterion = nn.MSELoss()
    
    X = []
    Y = []
    Y_test = []
    for epoch in range(20):
        cost = 0
        for users, items, ratings in train_loader:
            optimizer.zero_grad()
            ratings_pred = model(users,items)
            loss = criterion(ratings_pred, ratings)
            loss.backward()
            optimizer.step()
            cost += loss.item() * len(ratings)
            
        cost /= n_ratings

        X.append(epoch+1)
        Y.append(cost) 
        
        print(f"Epoch: {epoch}")
        print("train cost: {:.6f}" .format(cost))

        Y_test.append(calculate_valid(valid_loader,model,n_ratings_pred))

    plt.ylabel("MSE")
    plt.xlabel("Epoch")
    plt.plot(X,Y, c="blue", label="Training MSE")
    plt.plot(X,Y_test, c="red", label="Test MSE")
    plt.legend()
    plt.show()
    
    
    """
    Implement code for training the recommendation model
    """

    torch.save(model.state_dict(), args.save_model)



