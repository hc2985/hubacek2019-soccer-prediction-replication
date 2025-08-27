import os
import pandas as pd
import numpy as np
from collections import defaultdict, deque
from feature_eng.team_tracking import Teamstats, StandingsEntry, LeagueSeason
from feature_eng.pagerank import PageRank


team_dict = {}
standings_dict = {}
season_q =  deque(maxlen=2)
pagerank_q = deque(maxlen=3)
combined_index = {}
combined_points = []
combined_games = []


def get_pagerank(combined_index, points, games, home_team, away_team,
                                  restart_prob=0.05, tol=1e-9, max_iter=1000):
    # fast fail
    if len(combined_index) == 0:
        return 0.0, 0.0

    # build A = points / games (per-cell ratio)
    n = len(combined_index)
    A = np.zeros((n, n), dtype=float)
    mask = games > 0.0
    A[mask] = points[mask] / games[mask]

    # column-normalize A -> M
    col_sums = A.sum(axis=0)
    M = np.zeros_like(A, dtype=float)
    nonzero_cols = col_sums > 0.0
    M[:, nonzero_cols] = A[:, nonzero_cols] / col_sums[nonzero_cols]

    # dangling columns mask
    dangling_mask = (~nonzero_cols).astype(float)

    # power iteration: pi <- alpha*(M @ pi + dangling_mass/n) + d * v
    alpha = 1.0 - restart_prob
    v = np.ones(n, dtype=float) / n
    pi = v.copy()

    for _ in range(max_iter):
        pi_old = pi
        dangling_mass = (pi_old * dangling_mask).sum()
        pi = alpha * (M @ pi_old + dangling_mass / n) + restart_prob * v
        if np.linalg.norm(pi - pi_old, 1) < tol:
            break

    if pi.sum() > 0:
        pi /= pi.sum()

    # lookup indices and return
    home_idx = combined_index.get(home_team)
    away_idx = combined_index.get(away_team)
    pagerank_home = float(pi[home_idx]) if home_idx is not None else 0.0
    pagerank_away = float(pi[away_idx]) if away_idx is not None else 0.0
    return pagerank_home, pagerank_away


def update_standings_dict(home, away, home_points, away_points):
    if home not in standings_dict:
        standings_dict[home] = StandingsEntry(home)
    standings_dict[home].points += home_points
    
    if away not in standings_dict:
        standings_dict[away] = StandingsEntry(away)
    standings_dict[away].points += away_points

def get_specific_ranking(name):
    sorted_standings = sorted(standings_dict.items(),key=lambda item: item[1].points,reverse=True)
    for rank, (team_name, sorted_stand) in enumerate(sorted_standings, start=1):
        if team_name == name:
            return rank

def get_ranking(points, current_round):
    # Sort the standings dictionary by points and update the standing attribute for each team
    # Also provide current team and points on each threshold
    output = [0]*10
    idx = 0
    sorted_standings = sorted(standings_dict.items(),key=lambda item: item[1].points,reverse=True) #sort by points, where item[1] is StandingsEntry
    for rank, (team_name, sorted_stand) in enumerate(sorted_standings, start=1): #rank - integer position , (team_name, sorted_stand)- corresponding dictionary key:value pair     
        if rank <= 5 or rank >= season_q[-1].team_cnt - 4:
            output[idx] = (sorted_stand.points-points)/current_round
            idx += 1
    return output

def new_season_check(date, prev_date):
    return (date-prev_date).days > 60

def new_season():
    #reset standings
    standings_dict.clear()
    #add new season/remove season older than 3 years
    season_q.append(LeagueSeason())
    #add new matrix/remove matrix older than 3 years
    pagerank_q.append(PageRank(season_q[-1].team_cnt))

def combine_matrix():
    combined_index = {}
    for matrix in pagerank_q:
        for team, idx in matrix.get_teams():
            if team not in combined_index:
                combined_index[team] = len(combined_index)
    points = np.zeros((len(combined_index), len(combined_index)))
    games = np.zeros((len(combined_index), len(combined_index)))

    for matrix in pagerank_q:
        for team, idx in matrix.get_teams():
            for other_team, other_idx in matrix.get_teams():
                points[combined_index[team], combined_index[other_team]] += matrix.points[idx, other_idx]
                games[combined_index[team], combined_index[other_team]] += matrix.games[idx, other_idx]

    return combined_index, points, games

combined_index, combined_points, combined_games = combine_matrix()

def get_league_stats():
    h_avgs = []
    h_stds = []
    a_avgs = []
    a_stds = []
    home_win_pct = []
    draw_pct = []
    goal_diff_std = []

    for season in season_q:
        h_avgs.append(season.h_goals_avg)
        h_stds.append(season.h_goals_std)
        a_avgs.append(season.a_goals_avg)
        a_stds.append(season.a_goals_std)
        home_win_pct.append(season.h_win_pct)
        draw_pct.append(season.draw_pct)
        goal_diff_std.append(season.goal_diff_std)

    return {
        "home_goals_avg": float(np.mean(h_avgs)),
        "home_goals_std": float(np.mean(h_stds)),
        "away_goals_avg": float(np.mean(a_avgs)),
        "away_goals_std": float(np.mean(a_stds)),
        "home_win_pct": float(np.mean(home_win_pct)),
        "draw_pct": float(np.mean(draw_pct)),
        "goal_diff_std": float(np.mean(goal_diff_std)),
        "team_cnt": season_q[-1].team_cnt,
        "round_cnt": season_q[-1].rnd_cnt
    }


def featureengineer(df, options = ""):
    #create new features that will be used for training the model

    standings_dict.clear()
    prevmonth = None
    prev_date = None
    features_list = []
    pagerank_home = 0
    pagerank_away = 0
    current_round = 1

    for idx, row in df.iterrows():
        date = pd.to_datetime(row['Date'], format='mixed', dayfirst=True, errors='raise')
        month = date.month

        if not prevmonth or new_season_check(date, prev_date):
            new_season()
            current_round = 1
            for team in team_dict.values():
                team.new_season()

        prevmonth = month
        prev_date = date

        home_team = row['HomeTeam']
        away_team = row['AwayTeam']

        for team_name in [home_team, away_team]:
            if team_name not in team_dict:     
                team_dict[team_name] = Teamstats(team_name)
                team_dict[team_name].new_season()
            if team_name not in standings_dict:
                standings_dict[team_name] = StandingsEntry(team_name)

        if len(standings_dict) > 20:
            print(home_team)
            print(away_team)
            print(date)

        home_goals = row['FTHG']
        away_goals = row['FTAG']

        result_home = 2 if home_goals > away_goals else 0 if home_goals < away_goals else 1
        result_away = 2-result_home

        home_points = 3 if result_home == 2 else 1 if result_home == 1 else 0
        away_points = 3 if result_away == 2 else 1 if result_away == 1 else 0

        home_historical = team_dict[home_team].get_historical_strength()
        away_historical = team_dict[away_team].get_historical_strength()

        home_form = team_dict[home_team].get_form(date)
        away_form = team_dict[away_team].get_form(date)

        egd = team_dict[home_team].egd

        home_diff = get_ranking(standings_dict[home_team].points, current_round)
        away_diff = get_ranking(standings_dict[away_team].points, current_round)

        league_stats = get_league_stats()   

        # Build features row before updating stats
        features_row = {
            #Historical Strength
            #home team
            "home_h_win_pct": home_historical.get("h_win_pct", 0),
            "home_a_win_pct": away_historical.get("a_win_pct", 0),
            "home_h_draw_pct": home_historical.get("h_draw_pct", 0),
            "home_a_draw_pct": away_historical.get("a_draw_pct", 0),
            "home_h_goals_avg": home_historical.get("h_goals_avg", 0),
            "home_a_goals_avg": away_historical.get("a_goals_avg", 0),
            "home_h_concede_avg": home_historical.get("h_concede_avg", 0),
            "home_a_concede_avg": away_historical.get("a_concede_avg", 0),
            "home_h_goals_std": home_historical.get("h_goals_std", 0),
            "home_a_goals_std": away_historical.get("a_goals_std", 0),
            "home_h_concede_std": home_historical.get("h_concede_std", 0),
            "home_a_concede_std": away_historical.get("a_concede_std", 0),
            #away team
            "away_h_win_pct": away_historical.get("h_win_pct", 0),
            "away_a_win_pct": away_historical.get("a_win_pct", 0),
            "away_h_draw_pct": away_historical.get("h_draw_pct", 0),
            "away_a_draw_pct": away_historical.get("a_draw_pct", 0),
            "away_h_goals_avg": away_historical.get("h_goals_avg", 0),
            "away_a_goals_avg": away_historical.get("a_goals_avg", 0),
            "away_h_concede_avg": away_historical.get("h_concede_avg", 0),
            "away_a_concede_avg": away_historical.get("a_concede_avg", 0),
            "away_h_goals_std": away_historical.get("h_goals_std", 0),
            "away_a_goals_std": away_historical.get("a_goals_std", 0),
            "away_h_concede_std": away_historical.get("h_concede_std", 0),
            "away_a_concede_std": away_historical.get("a_concede_std", 0),

            #Current Form
            #home team
            "home_win_pct": home_form.get("win_pct"),
            "home_draw_pct": home_form.get("draw_pct"),
            "home_goals_avg": home_form.get("goals_avg"),
            "home_concede_avg": home_form.get("concede_avg"),
            "home_goals_std": home_form.get("goals_std"),
            "home_concede_std": home_form.get("concede_std"),
            "home_rest": home_form.get("rest"),

            #away team
            "away_win_pct": away_form.get("win_pct"),
            "away_draw_pct": away_form.get("draw_pct"),
            "away_goals_avg": away_form.get("goals_avg"),
            "away_concede_avg": away_form.get("concede_avg"),
            "away_goals_std": away_form.get("goals_std"),
            "away_concede_std": away_form.get("concede_std"),
            "away_rest": away_form.get("rest"),

            #Pi Ratings
            "home_h_rtg": team_dict[home_team].home_pi,
            "home_a_rtg": team_dict[home_team].away_pi,
            "away_h_rtg": team_dict[away_team].home_pi,
            "away_a_rtg": team_dict[away_team].away_pi,
            "EGD": egd,

            #Page Rank
            "pagerank_home": pagerank_home,
            "pagerank_away": pagerank_away,

            #Match Importance
            #home
            "home_diff_1": home_diff[0],
            "home_diff_2": home_diff[1],
            "home_diff_3": home_diff[2],
            "home_diff_4": home_diff[3],
            "home_diff_5": home_diff[4],
            "home_diff_16": home_diff[5],
            "home_diff_17": home_diff[6],
            "home_diff_18": home_diff[7],
            "home_diff_19": home_diff[8],
            "home_diff_20": home_diff[9],
            #away
            "away_diff_1": away_diff[0],
            "away_diff_2": away_diff[1],
            "away_diff_3": away_diff[2],
            "away_diff_4": away_diff[3],
            "away_diff_5": away_diff[4],
            "away_diff_16": away_diff[5],
            "away_diff_17": away_diff[6],
            "away_diff_18": away_diff[7],
            "away_diff_19": away_diff[8],
            "away_diff_20": away_diff[9],

            #League Rank
            "h_gs_avg": league_stats.get("home_goals_avg"),
            "s_gs_avg": league_stats.get("away_goals_avg"),
            "h_gs_std": league_stats.get("home_goals_std"),
            "a_gs_std": league_stats.get("away_goals_std"),
            "h_win_pct": league_stats.get("home_win_pct"),
            "draw_pct": league_stats.get("draw_pct"),
            "team_cnt": league_stats.get("team_cnt"),
            "goal_diff_std": league_stats.get("goal_diff_std"),
            "round_cnt": league_stats.get("round_cnt"),

            #others
            'Date': date,
            'Home_Team': home_team,
            'Away_Team': away_team,
            'Match_Result': result_home
        }
        
        # Seperate list for training and testing
        features_list.append(features_row)

        # Update after appending
        team_dict[home_team].add_game(
            goals_for=home_goals, goals_against=away_goals, result=result_home, status="H", date=date
        )

        team_dict[away_team].add_game(
            goals_for=away_goals, goals_against=home_goals, result=result_away, status="A", date=date
        )

        current_round = max(team_dict[home_team].games, team_dict[away_team].games)

        season_q[-1].add_game(home_goals, away_goals)

        home_h = team_dict[home_team].home_pi
        away_a = team_dict[away_team].away_pi
        
        team_dict[home_team].calculate_pi_rating(home_goals, away_goals, result_home, "H", away_a)
        team_dict[away_team].calculate_pi_rating(away_goals, home_goals, result_away, "A", home_h)

        combined_index, points, games = combine_matrix()

        pagerank_home, pagerank_away = get_pagerank(combined_index, points, games, home_team, away_team) 

        update_standings_dict(home_team, away_team, home_points, away_points)
    
    final_features_df = pd.DataFrame(features_list)

    if options == "save":
        final_features_df.to_csv("datasets/2001-2025_processed.csv", index=False)

    return final_features_df