#!/bin/python3

import pulp

"""
Evaluates to (name_mapping, fifteen) where fifteen is the the set of all linear programming variables optimized using predictions, assuming that last week's squad was previous_squad and saved_transfer is True if we have a saved transfer at this point penalty is the cost of a substitution. name_mapping maps the name of each linear programming variable to the player associated to that variable.

NOTE: We do not optimize for the eleven as this causes a maximum recursion depth error.
I believe there are just too many variables if we solve for that. But the problems are reasonably similar.
"""
def optimize(predictions, previous_squad=[], saved_transfer=False, penalty=0):
  fifteen = { name : (pulp.LpVariable(name, lowBound=0, upBound=1, cat="Integer"), name) for name in predictions }
  goal_keepers = { name : fifteen[name][0] for name in predictions if predictions[name]['position'] == 1}
  defenders = { name : fifteen[name][0] for name in predictions if predictions[name]['position'] == 2}
  mid_fielders = { name: fifteen[name][0] for name in predictions if predictions[name]['position'] == 3}
  strikers = { name : fifteen[name][0] for name in predictions if predictions[name]['position'] == 4}

  model = pulp.LpProblem("Fantasy Premier League", pulp.LpMaximize)

  # 2 Goalkeepers, 5 defenders, 5 mid_fields, 5 strikers in whole squad
  model += pulp.lpSum( [goal_keepers[name] for name in goal_keepers] ) == 2
  model += pulp.lpSum( [defenders[name] for name in defenders] ) == 5
  model += pulp.lpSum( [mid_fielders[name] for name in mid_fielders] ) == 5
  model += pulp.lpSum( [strikers[name] for name in strikers] ) == 3

  # Cost Cap
  model += pulp.lpSum( [fifteen[name][0] * predictions[name]['value'] for name in predictions] ) <= 1000

  # Only three players in the squad per team
  for team in set([predictions[name]['team'] for name in predictions]):
      team_members = { name : fifteen[name][0] for name in predictions if int(predictions[name]['team']) == int(team)}
      model += pulp.lpSum( [team_members[name] for name in team_members] ) <= 3
    
  # Maximize the squad score minus the transfer penalty. Can we do anything about the free transfer?
  model += pulp.lpSum( [fifteen[name][0] * predictions[name]['total_points'] for name in fifteen] ) \
          - (penalty * pulp.lpSum( [fifteen[name][0] * (1 if name not in previous_squad else 0) for name in predictions] )) \
          + ((2 if saved_transfer else 1) * penalty)

  model.solve()

  name_mapping = { variable.name : name for (variable, name) in fifteen.values() }
  fifteen = [ variable for (variable, _) in fifteen.values() ]

  return (name_mapping, fifteen)
