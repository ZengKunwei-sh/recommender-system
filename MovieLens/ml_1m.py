import numpy as np
import pandas as pd
import tqdm

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.nn import BCELoss

from fm import FactorizationMachineModel

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


class MovieLens1M_dateset(Dataset):
	"""_summary_
		file name:
			ratings.csv: records UserId, ItemId, and Rating
		columns:
			userId: range from 1 to 610
			movieId: range from 1 to 9724
			rating: range from 0 to 5
			timestamp: not used
	"""
	def __init__(self, data):
		"""_summary_
		Args:
			data (pd.DataFrame): includes fields and target
		"""
		data['userId'] = data['userId'] - 1
		data['movieId'] = data['movieId'] - 1
		self.y = data['rating'].map(self.__rate_to_like)
		self.X = data.drop(['rating', 'timestamp'], axis=1)
		self.field_dims = [np.max(self.X[col]+1) for col in ['userId', 'movieId']]

	def __len__(self):
		return self.y.shape[0]

	def __getitem__(self, index):
		X = np.array(self.X.loc[index])
		y = np.array(self.y[index])
		return torch.from_numpy(X), torch.from_numpy(y) 

	def __rate_to_like(self, rate):
		if rate <= 3:
			return 0
		else:
			return 1

def train(data_loader, model, criterion, optimizer, device=None):
	if device is not None:
		model = model.to(device)
	
	model.train()
	
	for i, (fields, target) in tqdm.tqdm(enumerate(data_loader)):
		if device is not None:
			fields, target = fields.to(device), target.to(device)
		
		output = model(fields)
		loss = criterion(output, target.float())

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

def eval(model, data_loader, device=None):
	model.eval()
	targets, predicts = list(), list()
	with torch.no_grad():
		for i, (fields, target) in tqdm.tqdm(enumerate(data_loader)):
			if device is not None:
				fields, target = fields.to(device), target.to(device)
			output = model(fields)
			targets.extend(target.detach().tolist())
			predicts.extend(output.detach().tolist())
	
	score = roc_auc_score(targets, predicts)
	#print(f'AUC score: {score}')
	return score

def train_and_eval(model, train_loader, test_loader, epochs, loss, optimizer):
	for epoch in range(epochs):
		print('='*5 + f'epoch {epoch+1}' + '='*5)
		train(train_loader, fm, loss, optimizer, device='cuda')
		train_auc = eval(fm, train_loader, device='cuda')
		test_auc = eval(fm, test_loader, device='cuda')
		print(f'train AUC: {train_auc}, test AUC: {test_auc}')

if __name__ == '__main__':
	path = 'G:/movielens/ml-latest-small/ratings.csv'
	data = pd.read_csv(path)
	y = data['rating']
	data = data.drop('rating', axis=1)
	X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.2, random_state=1)


	train_dataset = MovieLens1M_dateset(pd.concat([X_train, y_train], axis=1).reset_index(drop=True))
	train_loader = DataLoader(dataset=train_dataset, batch_size=256, shuffle=True)

	test_dataset = MovieLens1M_dateset(pd.concat([X_test, y_test], axis=1).reset_index(drop=True))
	test_loader = DataLoader(dataset=test_dataset, batch_size=256, shuffle=True)

	field_dims = [np.max(data[col])+1 for col in ['userId', 'movieId']]
	embed_dim = 8
	fm = FactorizationMachineModel(field_dims, embed_dim)

	loss = BCELoss(reduction='sum').to('cuda')
	optimizer = torch.optim.SGD(fm.parameters(), lr=0.1)

	epochs = 5
	train_and_eval(fm, train_loader, test_loader, epochs, loss, optimizer)