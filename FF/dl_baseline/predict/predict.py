#!/bin/python3

import torch
from functools import cmp_to_key
from .model import *
from .optimize import *
from .data import *

"""
Evaluates to (squad, candidates, captain, vice_captain, priorites) picked by model + optimization to play in season SEASON and round ROUND, assuming that the squad from last week is previous_squad, and saved_transfer is True if you have a saved transfer.
penalty is the cost of substitutions used by the linear programming solver. If model is None then a new one is trained, otherwise it is loaded from save_path. You can also choose how many EPOCHS to train the model, see the training error relative to the epoch if log is True, and finally log the training curve using SummaryWriter if log_path is specified.
"""
def predict(SEASON, ROUND, model=None, EPOCHS=100, previous_squad=[], saved_transfer=False, penalty=7, log_path="", save_path="", log=False, save=False):
  HIDDEN_DIM, BATCH_SIZE, LR, EMBEDDING_DIM = 512, 512, 1e-3, len(FIELDS) - 1
  
  name_mapping = name_conversions()
  season, previous_week = (SEASON, ROUND - 1) if ROUND > 1 else (max([2016, SEASON - 1]), 38)
  players_data = get_players_data(season, previous_week, name_mapping)
  train_dataset = PlayerDataset(players_data, batch_size=BATCH_SIZE, embedding_dim=EMBEDDING_DIM)
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  
  if model is None:
    model = GRUPredictor(EMBEDDING_DIM, HIDDEN_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), LR)
    criterion = torch.nn.SmoothL1Loss()
    train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=True)
    train_model(model, optimizer, criterion, train_dataloader, EPOCHS, device, log=log, save=save, log_path=log_path, save_path=save_path)

  ps_and_ts = positions_and_teams(SEASON, name_mapping)
  values = get_gameweek_data(SEASON, ROUND, ps_and_ts, name_mapping) # Are player values posted here before the deadline?

  rankings = {}
  for name in ps_and_ts: # All players registered this season
    position, team = ps_and_ts[name]
    value = 1001 if name not in values else values[name]['value'] # Optimizer excludes this player

    prediction = 0 # I prefer for this to be -infinity but pulp refuses to solve such
    if name in players_data:
      history = train_dataset.player_data(name).to(device)
      length = history.shape[0]
      prediction = model(history.view(length, 1, EMBEDDING_DIM), [length]).view(-1)
      
    data = { 'team' : team, 'position' : position, 'value': value, 'total_points' : prediction }
    rankings[name] = data

  (name_mapping, fifteen) = optimize(rankings, previous_squad=previous_squad, saved_transfer=saved_transfer, penalty=penalty)
  (squad, candidates, captain, vice_captain) = pick_team(rankings, name_mapping, fifteen, ps_and_ts)

  priorities = {}
  for player in set(squad):
    priorities[player] = float(rankings[player]['total_points'])

  return (squad, candidates, captain, vice_captain, priorities)


"""
Evaluates to (squad, candidates, captain, vice_captain) given predictions, which
is a dictionary whose keys are player_names and whose values contain the expected
points of the player, name_mapping which maps from the names of the linear programming variables in fifteen
to player_names - these two are expected to be the output of optimize - and 
positions_and_teams is expected to be the output of positions_and_teams.
"""
def pick_team(predictions, name_mapping, fifteen, positions_and_teams):
  squad = set([name_mapping[v.name] for v in fifteen if v.varValue != 0])
  required = [1, 3, 0, 1]
  squad_positions = { 1 : [], 2 : [], 3 : [], 4 : [] }

  # Rank players in each position
  for player in squad:
      squad_positions[positions_and_teams[player][0]].append((player, predictions[player]['total_points']))
  for position in squad_positions:
      squad_positions[position] = sorted(squad_positions[position], key=cmp_to_key(lambda x, y: x[1] - y[1]), reverse=True)

  # Fill positions that need filling
  candidates = []
  for position in squad_positions:
    needed = required[position - 1]
    candidates += squad_positions[position][:needed]
    squad_positions[position] = squad_positions[position][needed:]

  # Fill remaining spots
  remaining = []
  for position in squad_positions:
    if not position == 1: # We've already picked a goalkeeper
      remaining += squad_positions[position]
  remaining = sorted(remaining, key=cmp_to_key(lambda x, y: x[1] - y[1]), reverse=True)
  playing_size = len(set(candidates))

  while playing_size < 11 :
    candidates.append(remaining.pop(0))
    playing_size += 1
  
  # Pick captain and vice captain
  candidates = sorted(candidates, key=cmp_to_key(lambda x, y: x[1] - y[1]), reverse=True)
  captain = candidates[0][0]
  vice_captain = candidates[1][0]
  candidates = set([ player for (player, _) in candidates ])

  return (squad, set(candidates), captain, vice_captain)


"""
Evaluates to a number > 1 if x played and y didn't, or if they both
played or didn't play but x is ranked higher than y. Otherwise evaluates
to -1.
"""
def play_priority_sort(x, y):
  (_, played1, points1), (_, played2, points2) = x, y
  if played1 and not played2: return 1
  if played2 and not played1: return -1
  return points1 - points2

"""
Evaluates to True if player played in the gameweek whose results are reflected in
gameweek_data. Otherwise evaluates to True.
"""
def played(player, gameweek_data):
  return gameweek_data[player]['minutes'] > 0 or \
        gameweek_data[player]['yellow_cards'] > 0 or \
        gameweek_data[player]['red_cards'] > 0

"""
Evaluates to a number, the score for the team scored, captain's scored is doubled if they played,
otherwise vice_captain's score is doubled. If the transfers from previous_squad to squad
number to more than 1, subtract 4 * that number -1, since everyone gets one free transfer
per week. Also add 4 back if saved transfer is True in this case.
"""
def compute_score(squad, candidates, captain, vice_captain, priorities, 
                  SEASON, ROUND, previous_squad=[], saved_transfer=False):
  name_mapping = name_conversions()
  ps_and_ts = positions_and_teams(SEASON, name_mapping)
  gameweek_data = get_gameweek_data(SEASON, ROUND, ps_and_ts, name_mapping)
  scored = team_scored(squad, candidates, priorities, gameweek_data)
  
  points_earned = 0
  for player in scored:
      points_earned += gameweek_data[player]['total_points']

  if played(captain, gameweek_data):
    points_earned += gameweek_data[captain]['total_points']
  elif played(vice_captain, gameweek_data):
    points_earned += gameweek_data[vice_captain]['total_points']

  num_transfers = 0 if len(previous_squad) == 0 else len(set(squad) - set(previous_squad))
  if num_transfers <= 1:
    return points_earned
  return points_earned - (4 * (num_transfers - (1 if saved_transfer else 0) - 1)) # -1 for free transfer


"""
Evaluates to an iterable of the team that actually played given squad of 15 and 11 candidates.
Some players might have to be substituted if they didn't play according to gameweek_data.
In this case use rankings[player]['total_points'] as substitution priorities.
"""
def team_scored(squad, candidates, priorities, gameweek_data):
  playing = [ player for player in candidates if played(player, gameweek_data) ]

  playing_size = len(set(playing))
  if playing_size < 11:
    # We'll play remaining in ascending order of their priorities
    squad = set(squad)
    dropped = set(candidates) - set(playing)
    substitutes = set(squad) - set(candidates)

    # Rank players according to their position
    remaining = { 1 : [], 2 : [], 3 : [], 4 : []}
    for player in (dropped.union(substitutes)):
      remaining[gameweek_data[player]['position']].append((player, played(player, gameweek_data), priorities[player]))
    for position in remaining:
      remaining[position] = sorted(remaining[position], key=cmp_to_key(play_priority_sort), reverse=True)

    # Compute the positions that need to be filled
    positions = [0, 0, 0, 0] # Goalkeeper, Defenders, Midfielders, Strikers
    for name in playing:
      positions[int(gameweek_data[name]['position']) - 1] += 1
    to_fill = [max(0, 1 - positions[0]), max(0, 3 - positions[1]), 0, max(0, 1 - positions[3])]

    # Fill the positions that need to be filled
    for position in range(len(positions)):
      still_need = to_fill[position]
      playing += [ player for (player, _, _) in remaining[position + 1][:still_need] ]
      remaining[position + 1] = remaining[position + 1][still_need:] if position != 0 else [] # Keeper already picked
    
    playing_size = len(set(playing))
    if playing_size < 11 : # Add remaining in ascending order till done
      last_batch = []
      for position in remaining:
        last_batch += remaining[position]
      last_batch = sorted(last_batch, key=cmp_to_key(lambda x, y: x[1] - y[1]), reverse=True)

      while playing_size < 11 :
        playing.append(last_batch.pop(0)[0])
        playing_size += 1

      """This is sufficient since if no more played the highest priority ones will
        be chosen, and these were candidates to begin with."""

  return set(playing)


"""
Evaluates to (squad, candidates, captain, vice_captain, priorities) for the best team
in season SEASON and round ROUND.
"""
def best_gameweek_team(SEASON, ROUND):
  name_mapping = name_conversions()
  ps_and_ts = positions_and_teams(SEASON, name_mapping)
  rankings = get_gameweek_data(SEASON, ROUND, ps_and_ts, name_mapping)
  (name_mapping, fifteen) = optimize(rankings)
  (squad, candidates, captain, vice_captain) = pick_team(rankings, name_mapping, fifteen, ps_and_ts)

  priorities = {}
  for player in set(squad):
    priorities[player] = float(rankings[player]['total_points'])

  return (squad, candidates, captain, vice_captain, priorities)
