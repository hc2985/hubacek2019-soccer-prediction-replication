import os
import pandas as pd
import numpy as np
from collections import defaultdict, deque
import math

class LeagueSeason:
    def __init__(self, league = "e1"):
        self.league = league
        self.games = 0
        self.h_goals_list = []
        self.h_goals_total = 0
        self.h_goals_avg = 0
        self.h_goals_std = 0

        self.a_goals_list = []
        self.a_goals_total = 0
        self.a_goals_avg = 0
        self.a_goals_std = 0

        self.home_wins = 0
        self.draws = 0
        self.h_win_pct = 0
        self.draw_pct = 0
        self.team_cnt = 20
        self.goal_diff_std = 0
        self.rnd_cnt = 38

    def add_game(self, home_goals, away_goals):
        self.games += 1

        self.h_goals_list.append(home_goals)
        self.h_goals_total += home_goals
        self.h_goals_avg = self.h_goals_total/self.games
        self.h_goals_std = np.std(self.h_goals_list)


        self.a_goals_list.append(away_goals)
        self.a_goals_total += away_goals
        self.a_goals_avg = self.a_goals_total/self.games
        self.a_goals_std = np.std(self.a_goals_list)

        if home_goals > away_goals:
            self.home_wins += 1
        elif home_goals == away_goals:
            self.draws += 1
        self.h_win_pct = self.home_wins / self.games
        self.draw_pct = self.draws / self.games

        self.goal_diff_std = np.std(np.array(self.h_goals_list) - np.array(self.a_goals_list))

    def home_goals_avg(self):
        return self.h_goals_avg
    
    def home_goals_std(self):
        return self.h_goals_std
    
    def away_goals_avg(self):
        return self.a_goals_avg

    def away_goals_std(self):
        return self.a_goals_std

    def home_win_pct(self):
        return self.h_win_pct

    def draw_pct(self):
         return self.draw_pct

    def goal_diff_std(self):
        return self.goal_diff_std
    
    def team_cnt(self):
        return self.team_cnt
    
    def rnd_cnt(self):
        return self.rnd_cnt




class StandingsEntry:
    def __init__(self, team_name: str):
        self.team_name = team_name
        self.points    = 0
        self.standing  = 0

class SeasonStats:
    def __init__(self, team_name):
        self.h_games = 0
        self.h_wins = 0
        self.h_draws = 0
        self.h_goals = 0
        self.h_concede = 0
        self.h_goals_avg = 0
        self.h_concede_avg = 0
        self.h_goals_std = 0
        self.h_concede_std = 0
        self.h_goals_list = []
        self.h_concede_list = []

        self.a_games = 0
        self.a_wins = 0
        self.a_draws = 0
        self.a_goals = 0
        self.a_concede = 0
        self.a_goals_avg = 0
        self.a_concede_avg = 0
        self.a_goals_std = 0
        self.a_concede_std = 0
        self.a_goals_list = []
        self.a_concede_list = []


class Teamstats:
    def __init__(self, team_name):
        self.team_name = team_name
        self.games = 0
        #Historical Strength list
        self.historical_strength = deque(maxlen=3) 
        #Current Form (past 5)
        self.form_q = deque(maxlen=5)
        self.goals_q = deque(maxlen=5)
        self.concede_q = deque(maxlen=5)
        self.result_q = deque(maxlen=5)
        self.win_pct = 0
        self.draw_pct = 0
        self.goals_avg = 0
        self.concede_avg = 0
        self.goals_std = 0
        self.concede_std = 0
        self.last_played = 0
        #Pi-Ratings - initial = 0
        self.home_pi = 0
        self.away_pi = 0
        self.egd = 0
        #PageRank
        self.pagerank = 0
        
    def new_season(self):
        new_season = SeasonStats(self.team_name)
        self.historical_strength.append(new_season)
        self.games = 0

    def add_game(self, goals_for, goals_against, result, status, date):
        self.games += 1
        #result 2: win, 1: draw, 0:loss
        self.last_played = date
        #For historical strength
        if status == "H":
            self.historical_strength[-1].h_games += 1
            self.historical_strength[-1].h_goals_list.append(goals_for)
            self.historical_strength[-1].h_goals += goals_for
            self.historical_strength[-1].h_concede_list.append(goals_against)
            self.historical_strength[-1].h_concede += goals_against
            if result == 2:
                self.historical_strength[-1].h_wins += 1
            elif result == 1:
                self.historical_strength[-1].h_draws += 1
        else:
            self.historical_strength[-1].a_games += 1
            self.historical_strength[-1].a_goals_list.append(goals_for)
            self.historical_strength[-1].a_goals += goals_for
            self.historical_strength[-1].a_concede_list.append(goals_against)
            self.historical_strength[-1].a_concede += goals_against
            if result == 2:
                self.historical_strength[-1].a_wins += 1
            elif result == 1:
                self.historical_strength[-1].a_draws += 1

        #For form
        self.form_q.append(result)
        self.goals_q.append(goals_for)
        self.concede_q.append(goals_against)
        self.result_q.append(result)

        self.win_pct = self.form_q.count(2)/len(self.form_q) if len(self.form_q) > 0 else np.nan
        self.draw_pct = self.form_q.count(1)/len(self.form_q) if len(self.form_q) > 0 else np.nan
        self.goals_avg = sum(self.goals_q)/len(self.goals_q) if len(self.goals_q) > 0 else np.nan
        self.concede_avg = sum(self.concede_q)/len(self.concede_q) if len(self.concede_q) > 0 else np.nan
        self.goals_std = np.std(self.goals_q) if len(self.goals_q) > 0 else np.nan
        self.concede_std = np.std(self.concede_q) if len(self.concede_q) > 0 else np.nan

    def get_historical_strength(self):

        h_goals_list = []
        h_concede_list = []
        a_goals_list = []
        a_concede_list = []

        h_games = 0
        h_wins = 0
        h_draws = 0
        h_goals = 0
        h_conceded = 0

        a_games = 0
        a_wins = 0
        a_draws = 0
        a_goals = 0
        a_conceded = 0
        

        for season in self.historical_strength:
            h_goals_list += season.h_goals_list
            h_concede_list += season.h_concede_list
            h_games += season.h_games
            h_wins += season.h_wins
            h_draws += season.h_draws
            h_goals += season.h_goals
            h_conceded += season.h_concede

            a_goals_list += season.a_goals_list
            a_concede_list += season.a_concede_list
            a_games += season.a_games
            a_wins += season.a_wins
            a_draws += season.a_draws
            a_goals += season.a_goals
            a_conceded += season.a_concede

        h_win_pct = h_wins/h_games if h_games > 0 else 0
        a_win_pct = a_wins/a_games if a_games > 0 else 0
        h_draw_pct = h_draws/h_games if h_games > 0 else 0
        a_draw_pct = a_draws/a_games if a_games > 0 else 0
        h_goals_avg = h_goals/h_games if h_games > 0 else 0
        a_goals_avg = a_goals/a_games if a_games > 0 else 0
        h_concede_avg = h_conceded/h_games if h_games > 0 else 0
        a_concede_avg = a_conceded/a_games if a_games > 0 else 0
        h_goals_std = np.std(h_goals_list) if h_goals_list else 0
        a_goals_std = np.std(a_goals_list) if a_goals_list else 0
        h_concede_std = np.std(h_concede_list) if h_concede_list else 0
        a_concede_std = np.std(a_concede_list) if a_concede_list else 0

        output = {
            "h_win_pct": h_win_pct,
            "h_draw_pct": h_draw_pct,
            "h_goals_avg": h_goals_avg,
            "h_concede_avg": h_concede_avg,
            "h_goals_std": h_goals_std,
            "h_concede_std": h_concede_std,
            "a_win_pct": a_win_pct,
            "a_draw_pct": a_draw_pct,
            "a_goals_avg": a_goals_avg,
            "a_concede_avg": a_concede_avg,
            "a_goals_std": a_goals_std,
            "a_concede_std": a_concede_std
        }

        return output

    def get_form(self, date):
        output = {
            "win_pct": self.win_pct,
            "draw_pct": self.draw_pct,
            "goals_avg": self.goals_avg,
            "concede_avg": self.concede_avg,
            "goals_std": self.goals_std,
            "concede_std": self.concede_std,
            "rest": (date - self.last_played).days if self.last_played else 0
        }

        return output

    def calculate_pi_rating(self, goals_for, goals_against, result, status, opponent_rating, lambda_=0.06, gamma=0.5, b=10, c=3):

        def expected_from_rating(rating, b=b, c=c):
            mag = (b ** (abs(rating) / c)) - 1
            return math.copysign(mag, rating)
        
        #step 1,2,3 get expected goal difference and observed goal difference
        if status == "H":
            pred_home = expected_from_rating(self.home_pi, b=b, c=c)
            pred_away = expected_from_rating(opponent_rating, b=b, c=c)
            predicted_gd = pred_home - pred_away
            observed_gd = goals_for - goals_against
        else:
            pred_home = expected_from_rating(opponent_rating, b=b, c=c)
            pred_away = expected_from_rating(self.away_pi, b=b, c=c)
            predicted_gd = pred_home - pred_away
            observed_gd = goals_against - goals_for 
        
        def expected_goal_diff():
            return abs(predicted_gd)

        #step 4 compute error
        error = abs(observed_gd - predicted_gd)

        #step 5 diminishing equation
        psi = c * math.log10(1 + error)

        #signed psi: positive-home overperformed, negative-home underperformed
        signed_psi = math.copysign(psi, observed_gd - predicted_gd)

        # core_deltas depending on home/away
        core_delta_home = lambda_ * signed_psi
        core_delta_away = -core_delta_home

        # apply to primary ratings and propagate gamma to secondary ratings
        if status == "H":
            self.home_pi += core_delta_home
            self.away_pi += core_delta_home * gamma
        else:
            self.away_pi += core_delta_away
            self.home_pi += core_delta_away * gamma

    def home_pi(self):
        return self.home_pi

    def away_pi(self):
        return self.away_pi
    
    def edg(self):
        return self.egd