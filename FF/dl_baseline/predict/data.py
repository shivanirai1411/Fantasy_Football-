#!/bin/python3

import torch
import pandas as pd
import random
import os
from sklearn import preprocessing



FIELDS = ['assists', 'bonus', 'bps', 'clean_sheets', 'creativity', 'goals_conceded', 'goals_scored', 'ict_index', 'influence', 'minutes', 'opponent_team', 'own_goals', 'penalties_missed', 'penalties_saved', 'player', 'red_cards', 'round', 'saves', 'selected', 'team_a_score', 'team_h_score', 'threat',  'total_points', 'transfers_balance', 'transfers_in', 'transfers_out', 'value', 'was_home', 'yellow_cards']


"""
Some players have different names at different points of the data
e.g. Isaac Success and Isaac Success Ajayi. 

This function evaluates to a dictionary mapping different representations
to a single representation using name_conversions.csv, which is manually generated.
"""
def name_conversions():
  # Initialize the mapping of names in players_raw.csv that need to be translated
  name_mapping = {}
  player_mapping = pd.read_csv('./fpl_prediction/name_conversions.csv', encoding = "UTF-8")
  for row in player_mapping.itertuples():
      name_mapping[row.bad_name.lower()] = row.good_name.lower()
  
  return name_mapping


"""
Evaluates to a dictionary whose keys are player names and whose values
are the entire player's history up till season SEASON and round ROUND.
name_mapping is expected to be the result of name_conversions.
"""
def get_players_data(SEASON, ROUND, name_mapping):
  SEASON = { 2016 : 0, 2017 : 1, 2018 : 2, 2019 : 3 }[SEASON]
  directory_string = './fpl_prediction/Fantasy-Premier-League/data/20{0}-{1}/players/'
  players_data = {}
  players = {}
  index_count = 0

  for season in range(SEASON + 1):
      formatted_string = directory_string.format(season + 16, season + 16 + 1)
      directory = os.fsencode(formatted_string)

      for file in os.listdir(directory):
          filename = os.fsdecode(file)
          name = " ".join(filename.split('_')[:2]).lower()
          name = name_mapping[name] if name in name_mapping else name

          if name not in players:
              players[name] = index_count
              index_count = index_count + 1

          csv = pd.read_csv(formatted_string + filename + '/gw.csv', encoding = "UTF-8")
          csv = csv[csv['round'] <= ROUND] if season == SEASON else csv
          csv['round'] = 38 * season + csv['round']
          csv['player'] = pd.Series([players[name]] * len(csv))
          csv = csv[FIELDS]
          csv = csv.astype('float')

          if name not in players_data:
              players_data[name] = csv
          else:
              players_data[name] = pd.concat([players_data[name], csv])

  players_data = { name : df.drop_duplicates(subset=['round'], keep='last') for (name,df)  in players_data.items() if len(df) > 0}

  return players_data


"""
Evaluates to a dictionary whose keys are player names and whose values are
pairs containing the player's position and team for the season SEASON. Ideally we
would have these computed week-by-week but I don't think we have that data.
name_mapping is expected to be the output of name_conversions.
"""
def positions_and_teams(SEASON, name_mapping):
  SEASON = { 2016 : 0, 2017 : 1, 2018 : 2, 2019 : 3 }[SEASON]  
  directory_string = './fpl_prediction/Fantasy-Premier-League/data/20{0}-{1}/'
  formatted_string = directory_string.format(SEASON + 16, SEASON + 16 + 1)

  result={}
  csv = pd.read_csv(formatted_string + 'players_raw.csv', encoding = "UTF-8")
  for row in csv.itertuples():
      name = (row.first_name + ' ' + row.second_name).lower()
      name = name_mapping[name] if name in name_mapping else name
      position = row.element_type
      team_id = row.team_code
      result[name] = (position, team_id)
    
  return result


"""
Evaluates to a dictionary where the keys are the player names and the values
are dictionaries mapping attributes used for optimization and computing
team scores in gameweek ROUND of season SEASON. 
These attributes are ['round', 'value', 'total_points', 'minutes', 'yellow_cards', 'red_cards'].
positions_and_teams is expected to be the output of positions_and_teams(SEASON, name_mapping)
name_mapping is expected to be the output of name_conversions
"""
def get_gameweek_data(SEASON, ROUND, positions_and_teams, name_mapping):
  SEASON = { 2016 : 0, 2017 : 1, 2018 : 2, 2019 : 3 }[SEASON]  
  directory_string = './fpl_prediction/Fantasy-Premier-League/data/20{0}-{1}/players/'
  players_data = {}
  fields = ['round', 'value', 'total_points', 'minutes', 'yellow_cards', 'red_cards']

  # Fetch each player's performance for round ROUND and season SEASON
  formatted_string = directory_string.format(SEASON + 16, SEASON + 16 + 1)
  directory = os.fsencode(formatted_string)
  for file in os.listdir(directory):
      filename = os.fsdecode(file)
      name = " ".join(filename.split('_')[:2]).lower()
      name = name_mapping[name] if name in name_mapping else name
      csv = pd.read_csv(formatted_string + filename + '/gw.csv', encoding = "UTF-8")
      csv = csv[csv['round'] == ROUND]
      csv = csv[fields]
      csv = csv.astype('float')
      players_data[name] = csv
  
  players_data = { name : df.drop_duplicates(subset=['round'], keep='last') for (name,df) in players_data.items() if len(df) > 0}

  gameweek_data = {}
  for name in players_data:
    position = int(positions_and_teams[name][0])
    team = int(positions_and_teams[name][1])
    value = float(players_data[name]['value'])
    minutes = float(players_data[name]['minutes'])
    red_cards = float(players_data[name]['red_cards'])
    yellow_cards = float(players_data[name]['yellow_cards'])
    total_points = float(players_data[name]['total_points'])
    data = {'team' : team, 'position' : position, 'value': value, 'total_points' : total_points,
            'minutes' : minutes, 'yellow_cards' : yellow_cards, 'red_cards' : red_cards}
    gameweek_data[name] = data

  return gameweek_data


class PlayerDataset(torch.utils.data.Dataset):
  def __init__(self, players_data, batch_size, embedding_dim):
    self.batch_size = batch_size
    self.embedding_dim = embedding_dim

    all_data = pd.concat([players_data[name] for name in players_data])
    all_features = all_data.drop(['total_points'], axis=1).to_numpy()
    all_points = all_data.drop(all_data.columns.difference(['total_points']), axis=1).to_numpy()

    """Apparently no need to scale points since MSE is robust to scaling?"""
    feature_scaler = preprocessing.StandardScaler()
    scaled_features = feature_scaler.fit_transform(all_features)

    self.data = {}
    training_data = []
    end = 0

    # Save each player's history and points, and add it to training set. 
    # Also save entire player's history for prediction later on
    for name in players_data:
      history_length = len(players_data[name])

      for length in range(history_length - 1): # -1 because we have no prediction for the last point
        history = torch.Tensor(scaled_features[end : end + 1 + length, :])
        points = all_points[end + length + 1]
        training_data.append((history, points))

      new_end = end + history_length
      self.data[name] = torch.Tensor(scaled_features[end : new_end, :])
      end = new_end

    # Create the training batches
    random.shuffle(training_data)
    num_batches = len(training_data) // self.batch_size
    batches = [(k * self.batch_size, (k + 1) * self.batch_size) for k in range(num_batches)]
    batches.append((num_batches * self.batch_size, len(training_data)))

    self.batched_data = []
    for (start, end) in batches:
      if start != end:
        lengths = [len(features) for (features, _) in training_data[start : end]]
        three_d = torch.zeros((max(lengths), end - start, self.embedding_dim))

        total_points = []
        for index in range(start, end):
          features, points = training_data[index]
          three_d[: features.shape[0], index - start, : features.shape[1]] = features
          total_points.append(points)

        self.batched_data.append((three_d, torch.FloatTensor(lengths), torch.FloatTensor(total_points)))
    
  def __len__(self):
    return len(self.batched_data)

  def __getitem__(self, index):
    return self.batched_data[index]
    
  def player_data(self, name):
    return self.data[name]
