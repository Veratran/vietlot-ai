import json
import random
import math
from collections import defaultdict, Counter

with open('D:/vietlot-ai/data/power655.jsonl', 'r', encoding='utf-8') as f:
    draws = [json.loads(line) for line in f if line.strip()]

# Reverse to chronological order (oldest to newest)
draws = draws[::-1]

def eval_power655(picks, result):
    main_res = set(result[:6])
    bonus = result[6] if len(result) > 6 else None
    matches = len(set(picks).intersection(main_res))
    if matches == 6:
        return 30_000_000_000, 6
    elif matches == 5 and bonus and bonus in picks:
        return 3_000_000_000, 5.5
    elif matches == 5:
        return 40_000_000, 5
    elif matches == 4:
        return 500_000, 4
    elif matches == 3:
        return 50_000, 3
    return 0, matches

start_idx = 100
total_tested_draws = len(draws) - start_idx

random.seed(42)

def predict_random(history):
    return sorted(random.sample(range(1, 56), 6))

def predict_hot_all(history):
    counts = Counter()
    for d in history:
        counts.update(d['result'][:6])
    return [num for num, _ in counts.most_common(6)]

def predict_hot_w(history, w=30):
    counts = Counter()
    for d in history[-w:]:
        counts.update(d['result'][:6])
    picks = [num for num, _ in counts.most_common(6)]
    if len(picks) < 6:
        rem = [x for x in range(1, 56) if x not in picks]
        picks.extend(rem[:6 - len(picks)])
    return sorted(picks)

def predict_cold(history):
    last_seen = {i: -1 for i in range(1, 56)}
    for idx, d in enumerate(history):
        for num in d['result'][:6]:
            last_seen[num] = idx
    sorted_cold = sorted(range(1, 56), key=lambda x: last_seen[x])
    return sorted(sorted_cold[:6])

def predict_delta(history):
    deltas = []
    for d in history[-50:]:
        s = sorted(d['result'][:6])
        deltas.append(s[0])
        for i in range(1, 6):
            deltas.append(s[i] - s[i-1])
    d_counts = Counter(deltas)
    top_deltas = [d for d, _ in d_counts.most_common(20)]
    for _ in range(500):
        sample_d = random.sample(top_deltas, 6)
        nums = []
        curr = sample_d[0]
        nums.append(curr)
        valid = True
        for d in sample_d[1:]:
            curr += d
            if curr > 55 or curr in nums:
                valid = False
                break
            nums.append(curr)
        if valid and len(nums) == 6:
            return sorted(nums)
    return predict_random(history)

def predict_markov(history):
    trans = defaultdict(Counter)
    for i in range(len(history) - 1):
        prev = history[i]['result'][:6]
        nxt = history[i+1]['result'][:6]
        for p in prev:
            for n in nxt:
                trans[p][n] += 1
    last_res = history[-1]['result'][:6]
    scores = Counter()
    for p in last_res:
        for n in range(1, 56):
            scores[n] += trans[p][n]
    return sorted([n for n, _ in scores.most_common(6)])

def predict_gaussian_z(history):
    w = 50
    sub = history[-w:]
    freq = Counter()
    for d in sub:
        freq.update(d['result'][:6])
    mean_f = 6.0 * w / 55.0
    stdev_f = math.sqrt(w * (6.0/55.0) * (49.0/55.0))
    z_scores = {n: (freq[n] - mean_f)/stdev_f for n in range(1, 56)}
    return sorted(sorted(range(1, 56), key=lambda x: z_scores[x], reverse=True)[:6])

def predict_ensemble(history):
    h_all = predict_hot_w(history, 50)
    cold = predict_cold(history)
    mkv = predict_markov(history)
    gz = predict_gaussian_z(history)
    score = Counter()
    for n in h_all: score[n] += 2
    for n in cold: score[n] += 1.5
    for n in mkv: score[n] += 2.5
    for n in gz: score[n] += 2
    return sorted([n for n, _ in score.most_common(6)])

strategies = {
    '1. Random (Baseline)': predict_random,
    '2. Hot All-Time': predict_hot_all,
    '3. Hot Window W=30': lambda h: predict_hot_w(h, 30),
    '4. Hot Window W=50': lambda h: predict_hot_w(h, 50),
    '5. Cold / Gan Max-Gap': predict_cold,
    '6. Delta System': predict_delta,
    '7. Markov Chain (k=1)': predict_markov,
    '8. Gaussian Z-Score': predict_gaussian_z,
    '9. Multi-Factor Ensemble': predict_ensemble,
}

results = {}
for name, func in strategies.items():
    total_cost = 0
    total_payout = 0
    consecutive_losses = 0
    max_consecutive_losses = 0
    wins_count = 0
    equity = 0
    peak_equity = 0
    max_drawdown = 0
    match_distribution = Counter()
    
    # Financial bankroll tracking for initial capitals: 1M, 5M, 10M
    bankroll_1m = 1_000_000
    ruin_step_1m = None
    bankroll_5m = 5_000_000
    ruin_step_5m = None
    bankroll_10m = 10_000_000
    ruin_step_10m = None

    for idx in range(start_idx, len(draws)):
        hist = draws[:idx]
        current_draw = draws[idx]
        picks = func(hist)
        
        ticket_cost = 10_000
        total_cost += ticket_cost
        
        # update bankrolls
        if bankroll_1m >= ticket_cost:
            bankroll_1m -= ticket_cost
        elif ruin_step_1m is None:
            ruin_step_1m = idx - start_idx

        if bankroll_5m >= ticket_cost:
            bankroll_5m -= ticket_cost
        elif ruin_step_5m is None:
            ruin_step_5m = idx - start_idx

        if bankroll_10m >= ticket_cost:
            bankroll_10m -= ticket_cost
        elif ruin_step_10m is None:
            ruin_step_10m = idx - start_idx
        
        payout, m_level = eval_power655(picks, current_draw['result'])
        total_payout += payout
        match_distribution[m_level] += 1

        if ruin_step_1m is None:
            bankroll_1m += payout
        if ruin_step_5m is None:
            bankroll_5m += payout
        if ruin_step_10m is None:
            bankroll_10m += payout
        
        if payout > 0:
            wins_count += 1
            consecutive_losses = 0
        else:
            consecutive_losses += 1
            if consecutive_losses > max_consecutive_losses:
                max_consecutive_losses = consecutive_losses
        
        net_round = payout - ticket_cost
        equity += net_round
        if equity > peak_equity:
            peak_equity = equity
        dd = peak_equity - equity
        if dd > max_drawdown:
            max_drawdown = dd
            
    pnl = total_payout - total_cost
    roi = (pnl / total_cost) * 100
    win_rate = (wins_count / total_tested_draws) * 100
    results[name] = {
        'total_cost': total_cost,
        'total_payout': total_payout,
        'pnl': pnl,
        'roi': roi,
        'win_rate': win_rate,
        'max_loss_streak': max_consecutive_losses,
        'max_drawdown': max_drawdown,
        'ruin_1m': ruin_step_1m,
        'ruin_5m': ruin_step_5m,
        'ruin_10m': ruin_step_10m,
        'matches': dict(match_distribution)
    }

print('=== KET QUA BACKTEST CHI TIET TREN POWER 6/55 (1.322 KY QUAY) ===')
header = f"{'Chien Luoc':<25} | {'Von Da Chi':<12} | {'Tien Trung':<12} | {'Lo/Lai PnL':<12} | {'ROI (%)':<8} | {'Win%':<6} | {'Max Chuoi Thua':<14} | {'Max Drawdown':<12} | {'Chay Von 1M':<11} | {'Chay Von 5M':<11}"
print(header)
print('-' * len(header))
for name, r in results.items():
    ruin1 = f"Ky {r['ruin_1m']}" if r['ruin_1m'] else "Con ton"
    ruin5 = f"Ky {r['ruin_5m']}" if r['ruin_5m'] else "Con ton"
    line = f"{name:<25} | {r['total_cost']:>10,}d | {r['total_payout']:>10,}d | {r['pnl']:>10,}d | {r['roi']:>6.1f}% | {r['win_rate']:>5.2f}% | {r['max_loss_streak']:>11} ky | {r['max_drawdown']:>10,}d | {ruin1:>11} | {ruin5:>11}"
    print(line)

print('\nPhan bo so luong trung thuong chi tiet:')
for name, r in results.items():
    m = r['matches']
    g_ba = m.get(3, 0)
    g_nhi = m.get(4, 0)
    g_nhat = m.get(5, 0)
    jp2 = m.get(5.5, 0)
    jp1 = m.get(6, 0)
    print(f"  {name:<25}: G.Ba (3 so): {g_ba:2d} | G.Nhi (4 so): {g_nhi:2d} | G.Nhat (5 so): {g_nhat:2d} | JP2: {jp2} | JP1: {jp1}")
