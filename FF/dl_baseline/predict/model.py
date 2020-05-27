#!/bin/python3

import torch
from torch.utils.tensorboard import SummaryWriter

class GRUPredictor(torch.nn.Module):
  def __init__(self, embedding_dim, hidden_dim):
    super(GRUPredictor, self).__init__()
    self.hidden_dim = hidden_dim
    self.gru = torch.nn.GRU(embedding_dim, hidden_dim)
    self.fc = torch.nn.Linear(hidden_dim, 1)

  def forward(self, features, lengths):
    packed_features = torch.nn.utils.rnn.pack_padded_sequence(features, lengths, enforce_sorted=False)
    _, hidden = self.gru(packed_features)
    return self.fc(hidden)


def train_model(model, optimizer, criterion, train_dataloader, EPOCHS, device, log_path="", save_path="", save=False, log=False):
  if log:
    writer = SummaryWriter(log_path)

  for epoch in range(EPOCHS):
    running_loss = 0.0
    items = 0

    for index, (features, lengths, points) in enumerate(train_dataloader):
      features = features.squeeze(0).to(device)
      lengths = lengths.view(-1).to(device)
      points = points.view(-1).to(device)

      model.zero_grad()
      pred = model(features, lengths).view(-1)
      loss = criterion(pred, points)
      loss.backward()
      optimizer.step()

      running_loss += loss.item()
      items = items + 1
    
    avg_loss = running_loss / items
    if log:
      print('Epoch {0}: average loss {1}'.format(epoch + 1, avg_loss))
      writer.add_scalar("Loss", avg_loss, epoch + 1)

  if save:
    torch.save(model.state_dict(), save_path)
