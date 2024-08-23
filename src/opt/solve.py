from typing import Optional, Sequence, Tuple

from mip import BINARY, CBC, INTEGER, Model, minimize, xsum

from src.opt.record import Record

N = 20  # 同一リーグ 1 カードの合計試合数
M = 18  # 交流戦の合計試合数
T = 6  # チーム数
D = 20  # 1 シーズンの可能な引き分け数
S = N * (T - 1) + M  # 1 シーズンの合計試合数

EPS = 0.00001
INF = 10000


def solve(
    records: Sequence[Sequence[Record]], rank_prefix: Sequence[int]
) -> Tuple[bool, Optional[Sequence[Sequence[Record]]]]:
    model = Model(solver_name=CBC)

    # チーム i がチーム j に勝つ回数
    num_wins = {
        (i, j): model.add_var(var_type=INTEGER, name=f"num_wins_{i}_{j}")
        for i in range(T)
        for j in range(T + 1)
    }

    # チーム i がチーム j に負ける回数
    num_loses = {
        (i, j): model.add_var(var_type=INTEGER, name=f"num_loses_{i}_{j}")
        for i in range(T)
        for j in range(T + 1)
    }

    # チーム i の合計勝ち数
    total_wins = {
        i: model.add_var(var_type=INTEGER, name=f"total_wins_{i}") for i in range(T)
    }

    # チーム i の合計負け数
    total_loses = {
        i: model.add_var(var_type=INTEGER, name=f"total_loses_{i}") for i in range(T)
    }

    # チーム i の引き分けを除いた試合数が j か（j > 0 を前提）
    num_valid_matches = {
        (i, j): model.add_var(var_type=BINARY, name=f"num_valid_matches_{i}_{j}")
        for i in range(T)
        for j in range(1, S + 1)
    }

    # チーム i の引き分けを除いた試合数が j の場合の勝率（j > 0 を前提）
    winning_rates = {
        (i, j): model.add_var(name=f"winning_rates_{i}_{j}")
        for i in range(T)
        for j in range(1, S + 1)
    }

    # チーム i の勝率がチーム j の勝率より高いか
    has_higher_winning_rates = {
        (i, j): model.add_var(var_type=BINARY, name=f"has_higher_winning_rates_{i}_{j}")
        for i in range(T)
        for j in range(T)
    }

    # チーム i の勝ち数がチーム j の勝ち数より多いか
    has_more_wins = {
        (i, j): model.add_var(var_type=BINARY, name=f"has_more_wins_{i}_{j}")
        for i in range(T)
        for j in range(T)
    }

    # チーム i の順位がチーム j より高いか
    has_higher_ranks = {
        (i, j): model.add_var(var_type=BINARY, name=f"has_higher_ranks_{i}_{j}")
        for i in range(T)
        for j in range(T)
    }

    # 自チームとの試合は存在しない
    for i in range(T):
        model += num_wins[i, i] == 0
        model += num_loses[i, i] == 0

    # 既存の試合結果に矛盾しないよう num_wins と num_loses を設定
    for i, record in enumerate(records):
        for j, r in enumerate(record):
            if i == j:
                continue
            model += r.win <= num_wins[i, j]
            model += r.lose <= num_loses[i, j]
            if j < T:
                # 同一リーグ内
                model += num_wins[i, j] + num_loses[i, j] <= N - r.draw
            else:
                # 交流戦
                model += num_wins[i, j] + num_loses[i, j] <= M - r.draw

    # num_wins, num_loses に対する対称性の制約
    for i in range(T):
        for j in range(T):
            model += num_wins[i, j] == num_loses[j, i]

    # total_wins に対する制約
    for i in range(T):
        model += total_wins[i] == xsum(num_wins[i, j] for j in range(T + 1))

    # total_loses に対する制約
    for i in range(T):
        model += total_loses[i] == xsum(num_loses[i, j] for j in range(T + 1))

    # num_valid_matches に対する制約
    for i in range(T):
        for j in range(1, S + 1):
            if j < S - D:
                model += num_valid_matches[i, j] == 0
                continue

            # num_valid_matches[i, j] == 1 => total_wins[i] + total_loses[i] == j
            model += total_wins[i] + total_loses[i] <= j * num_valid_matches[
                i, j
            ] + INF * (1 - num_valid_matches[i, j])
            model += total_wins[i] + total_loses[i] >= j * num_valid_matches[i, j]

        # これを入れないと全て 0 が実行可能になる
        model += xsum(num_valid_matches[i, j] for j in range(1, S + 1)) == 1

    # winning_rates に対する制約
    for i in range(T):
        for j in range(1, S + 1):
            model += winning_rates[i, j] >= 0
            # num_valid_matches[i, j] == 0 => winning_rates[i, j] == 0
            model += winning_rates[i, j] <= num_valid_matches[i, j]
            # num_valid_matches[i, j] == 1 => winning_rates[i, j] == total_wins[i] / j
            model += j * winning_rates[i, j] >= total_wins[i] - INF * (
                1 - num_valid_matches[i, j]
            )
            model += j * winning_rates[i, j] <= total_wins[i] + INF * (
                1 - num_valid_matches[i, j]
            )

    # has_higher_ranks に対する制約
    for i in range(T):
        for j in range(T):
            if i == j:
                continue

            # has_higher_winning_rates == 1 => チーム i の勝率がチーム j の勝率より高い
            model += xsum(winning_rates[i, k] for k in range(1, S + 1)) - xsum(
                winning_rates[j, k] for k in range(1, S + 1)
            ) >= EPS - INF * (1 - has_higher_winning_rates[i, j])
            # has_higher_winning_rates == 0 => チーム i の勝率がチーム j の勝率より高くない
            model += (
                xsum(winning_rates[j, k] for k in range(1, S + 1))
                - xsum(winning_rates[i, k] for k in range(1, S + 1))
                >= -INF * has_higher_winning_rates[i, j]
            )

            # has_more_wins == 1 => チーム i の勝ち数がチーム j の勝ち数より多い
            model += total_wins[i] - total_wins[j] >= 1 - INF * (
                1 - has_more_wins[i, j]
            )
            # has_more_wins == 0 => チーム i の勝ち数がチーム j の勝ち数より多くない
            model += total_wins[j] - total_wins[i] >= -INF * has_more_wins[i, j]

            # 勝率か勝ち数が上回っている場合、およびその場合に限り順位が上である
            model += (
                has_higher_ranks[i, j]
                <= has_higher_winning_rates[i, j] + has_more_wins[i, j]
            )
            model += has_higher_ranks[i, j] >= has_higher_winning_rates[i, j]
            model += has_higher_ranks[i, j] >= has_more_wins[i, j]

    # 所望の順位を満たすための制約
    low_teams = set(range(T)) - set(rank_prefix)
    for i, t in enumerate(rank_prefix):
        if i > 0:
            model += has_higher_ranks[rank_prefix[i - 1], t] == 1

        for l in low_teams:
            model += has_higher_ranks[t, l] == 1

    model.optimize(max_seconds=10, max_solutions=1)

    print(f"status: {model.status.value}")
    print(f"objective value: {model.objective.x}")

    for i in range(T):
        print(f"total_wins[{i}]: {total_wins[i].x}")
        print(f"total_loses[{i}]: {total_loses[i].x}")
        # for j in range(T + 1):
        #     print(f"num_wins[{i}, {j}]: {num_wins[i, j].x}")
        #     print(f"num_loses[{i}, {j}]: {num_loses[i, j].x}")
        #     if j < T:
        #         print(
        #             f"num_draws[{i}, {j}]: {N - num_wins[i, j].x - num_loses[i, j].x}"
        #         )
        #     else:
        #         print(
        #             f"num_draws[{i}, {j}]: {M - num_wins[i, j].x - num_loses[i, j].x}"
        #         )
        for j in range(1, S + 1):
            if num_valid_matches[i, j].x > 0:
                print(f"winning_rates[{i}, {j}]: {winning_rates[i, j].x}")

    return False, None
