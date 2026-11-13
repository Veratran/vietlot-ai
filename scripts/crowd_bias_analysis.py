import json
from collections import defaultdict

with open('D:/vietlot-ai/data/power655.jsonl', 'r', encoding='utf-8') as f:
    draws = [json.loads(line) for line in f if line.strip()]

calendar_counts = []
g_ba_winners = []
g_nhi_winners = []
g_nhat_winners = []

for d in draws:
    res = d['result'][:6]
    c_cal = sum(1 for x in res if x <= 31)
    w = d.get('winners', {})
    w_ba = w.get('Giải ba', 0)
    w_nhi = w.get('Giải nhì', 0)
    w_nhat = w.get('Giải nhất', 0)
    if w_ba > 0:
        calendar_counts.append(c_cal)
        g_ba_winners.append(w_ba)
        g_nhi_winners.append(w_nhi)
        g_nhat_winners.append(w_nhat)

avg_ba = defaultdict(list)
avg_nhi = defaultdict(list)
avg_nhat = defaultdict(list)
for c, ba, nhi, nhat in zip(calendar_counts, g_ba_winners, g_nhi_winners, g_nhat_winners):
    avg_ba[c].append(ba)
    avg_nhi[c].append(nhi)
    avg_nhat[c].append(nhat)

print("Total draws analyzed:", len(calendar_counts))
print("=== THIEN LECH TAM LY DAM DONG THEO NGAY SINH (SO <= 31) ===")
header = "{:15} | {:10} | {:20} | {:20} | {:20}".format("So bong <= 31", "So ky quay", "Giai Ba TB (nguoi)", "Giai Nhi TB (nguoi)", "Giai Nhat TB (nguoi)")
print(header)
print("-" * len(header))
for c in sorted(avg_ba.keys()):
    n_k = len(avg_ba[c])
    m_ba = sum(avg_ba[c]) / n_k
    m_nhi = sum(avg_nhi[c]) / n_k
    m_nhat = sum(avg_nhat[c]) / n_k
    print("{:15} | {:10} | {:20.0f} | {:20.1f} | {:20.2f}".format(str(c) + " bong", n_k, m_ba, m_nhi, m_nhat))
