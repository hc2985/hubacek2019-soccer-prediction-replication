import os
import pandas as pd
import numpy as np
from collections import defaultdict, deque
import math



class PageRank:
    def __init__(self, team_cnt):
        #matrix will hold (3wins+draws, games)
        self.points = np.zeros((team_cnt, team_cnt))
        self.games = np.zeros((team_cnt, team_cnt))
        self.team_indices = {}

    def get_teams(self):
        return self.team_indices


    def add_match(self, home_team, away_team, result_home):
        if home_team not in self.team_indices:
            self.team_indices[home_team] = len(self.team_indices)
        if away_team not in self.team_indices:
            self.team_indices[away_team] = len(self.team_indices)

        idx1 = self.team_indices[home_team]
        idx2 = self.team_indices[away_team]

        if result_home == 2:  # Home win
            self.points[idx1, idx2] += 3
            self.games[idx1, idx2] += 1
            self.games[idx2, idx1] += 1
        elif result_home == 1:  # Draw
            self.points[idx1, idx2] += 1
            self.games[idx1, idx2] += 1
            self.points[idx2, idx1] += 1
            self.games[idx2, idx1] += 1
        elif result_home == 0:  # Away win
            self.points[idx2, idx1] += 3
            self.games[idx2, idx1] += 1
            self.games[idx1, idx2] += 1

    def get_matrix(self, home_team, away_team):
        home_point = 0
        home_game = 0
        away_point = 0
        away_game = 0
        if home_team in self.team_indices and away_team in self.team_indices:
            idx1 = self.team_indices[home_team]
            idx2 = self.team_indices[away_team]
            home_point = self.points[idx1, idx2]
            home_game = self.games[idx1, idx2]
            away_point = self.points[idx2, idx1]
            away_game = self.games[idx2, idx1]

        return home_point, home_game, away_point, away_game