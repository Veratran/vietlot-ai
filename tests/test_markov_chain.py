#!/usr/bin/env python3
"""
Unit tests for HigherOrderMarkovChain (scripts/markov_chain.py)
─────────────────────────────────────────────────────────────
Kiểm thử toàn diện mô hình Markov Chain bậc cao (k=1 và k=2):
  1. Kiểm thử chuyển tiếp bậc 1 và bậc 2.
  2. Bất biến chuẩn hóa tuyệt đối: Tổng xác suất mỗi hàng trạng thái == 1.0 (< 1e-6).
  3. Xử lý trạng thái chưa từng gặp (unseen states) & chống bẫy chia cho 0 (zero-division guard).
  4. Huấn luyện và dự đoán trên bộ dữ liệu thực tế data/power655.jsonl.
  5. Xử lý lỗi, serialization và xuất ma trận chuyển tiếp.
"""

import json
import math
from pathlib import Path
import pytest

from scripts.markov_chain import HigherOrderMarkovChain


@pytest.fixture
def sample_draws_small():
    """Tập dữ liệu nhỏ phục vụ kiểm thử logic xác suất xác định."""
    return [
        [1, 2, 3],
        [4, 5, 6],
        [1, 2, 7],
        [3, 4, 8],
        [5, 6, 9],
    ]


@pytest.fixture
def real_power655_draws():
    """Đọc dữ liệu thực tế từ data/power655.jsonl (nếu có)."""
    p = Path(__file__).resolve().parent.parent / "data" / "power655.jsonl"
    if not p.exists():
        pytest.skip(f"Không tìm thấy file dữ liệu thực tế: {p}")
    draws = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                draws.append(json.loads(line))
    # Dữ liệu power655 lưu mới nhất trước -> đảo ngược sang cũ nhất trước
    return draws[::-1]


# ----------------------------------------------------------------------
# 1. Khởi tạo & tham số mô hình
# ----------------------------------------------------------------------

def test_initialization_defaults():
    mc = HigherOrderMarkovChain(order=2, total_numbers=55)
    assert mc.order == 2
    assert mc.total_numbers == 55
    assert mc.alpha == 0.1
    assert len(mc.lambdas) == 3
    assert math.isclose(sum(mc.lambdas), 1.0, abs_tol=1e-7)
    assert not mc.is_fitted


def test_initialization_order1():
    mc = HigherOrderMarkovChain(order=1, total_numbers=45, alpha=0.2, lambdas=(0.8, 0.2))
    assert mc.order == 1
    assert mc.total_numbers == 45
    assert mc.alpha == 0.2
    assert len(mc.lambdas) == 2
    assert math.isclose(sum(mc.lambdas), 1.0, abs_tol=1e-7)


def test_initialization_invalid_arguments():
    with pytest.raises(ValueError, match="Bậc Markov không hợp lệ"):
        HigherOrderMarkovChain(order=3)

    with pytest.raises(ValueError, match="Bậc Markov không hợp lệ"):
        HigherOrderMarkovChain(order=0)

    with pytest.raises(ValueError, match="phải >= 2"):
        HigherOrderMarkovChain(order=1, total_numbers=1)

    with pytest.raises(ValueError, match="không được âm"):
        HigherOrderMarkovChain(order=1, total_numbers=45, alpha=-0.5)


def test_fit_minimum_draws_requirement():
    mc1 = HigherOrderMarkovChain(order=1, total_numbers=10)
    with pytest.raises(ValueError, match="Cần tối thiểu 2 kỳ quay"):
        mc1.fit([[1, 2]])

    mc2 = HigherOrderMarkovChain(order=2, total_numbers=10)
    with pytest.raises(ValueError, match="Cần tối thiểu 3 kỳ quay"):
        mc2.fit([[1, 2], [3, 4]])


# ----------------------------------------------------------------------
# 2. Kiểm thử chuyển tiếp Bậc 1 và Bậc 2
# ----------------------------------------------------------------------

def test_order1_transitions_logic(sample_draws_small):
    mc = HigherOrderMarkovChain(order=1, total_numbers=10, alpha=0.1, lambdas=(0.9, 0.1))
    mc.fit(sample_draws_small)

    # Trong sample_draws:
    # Kỳ 0: [1, 2, 3] -> Kỳ 1: [4, 5, 6]
    # Do đó từ số 1 có bước chuyển tới 4, 5, 6
    p_1_to_4 = mc.get_transition_probability(1, 4)
    p_1_to_10 = mc.get_transition_probability(1, 10)  # Số 10 chưa từng xuất hiện

    assert p_1_to_4 > p_1_to_10, "Xác suất chuyển tiếp tới số đã thấy (4) phải lớn hơn số chưa thấy (10)"

    # Tổng xác suất từ số 1 sang tất cả các số 1..10 phải là 1.0
    dist_1 = mc.get_transition_distribution(1)
    assert math.isclose(sum(dist_1.values()), 1.0, abs_tol=1e-6)
    assert len(dist_1) == 10


def test_order2_transitions_logic():
    # Chuỗi xác định 3 kỳ:
    # Kỳ 0: [1, 2]
    # Kỳ 1: [3, 4]
    # Kỳ 2: [5, 6]
    draws = [[1, 2], [3, 4], [5, 6]]
    mc = HigherOrderMarkovChain(order=2, total_numbers=10, alpha=0.1, lambdas=(0.7, 0.2, 0.1))
    mc.fit(draws)

    # Cặp (1, 3) có chuyển tiếp trực tiếp sang 5 và 6
    p_1_3_to_5 = mc.get_transition_probability((1, 3), 5)
    p_1_3_to_6 = mc.get_transition_probability((1, 3), 6)
    p_1_3_to_9 = mc.get_transition_probability((1, 3), 9)  # 9 chưa từng thấy

    assert math.isclose(p_1_3_to_5, p_1_3_to_6, abs_tol=1e-5)
    assert p_1_3_to_5 > p_1_3_to_9, "Xác suất (1, 3) -> 5 phải lớn hơn (1, 3) -> 9"

    # Hàng phân phối xác suất của (1, 3) phải chuẩn hóa về 1.0
    dist = mc.get_transition_distribution((1, 3))
    assert math.isclose(sum(dist.values()), 1.0, abs_tol=1e-6)


# ----------------------------------------------------------------------
# 3. Bất biến chuẩn hóa (Strict Row Normalization Invariant)
# ----------------------------------------------------------------------

def test_strict_row_normalization_invariant_order1(sample_draws_small):
    mc = HigherOrderMarkovChain(order=1, total_numbers=10)
    mc.fit(sample_draws_small)

    assert mc.verify_normalization_invariant(tolerance=1e-6) is True

    # Kiểm tra trực tiếp từng hàng
    for u in range(1, 11):
        dist = mc.get_transition_distribution((u,))
        total = sum(dist.values())
        assert abs(total - 1.0) < 1e-6, f"Hàng u={u} có tổng = {total} != 1.0"


def test_strict_row_normalization_invariant_order2(sample_draws_small):
    mc = HigherOrderMarkovChain(order=2, total_numbers=10)
    mc.fit(sample_draws_small)

    assert mc.verify_normalization_invariant(tolerance=1e-6) is True

    # Kiểm tra các cặp đã quan sát
    for pair in mc.transitions_order2.keys():
        dist = mc.get_transition_distribution(pair)
        total = sum(dist.values())
        assert abs(total - 1.0) < 1e-6, f"Cặp đã thấy {pair} có tổng = {total} != 1.0"


# ----------------------------------------------------------------------
# 4. Trạng thái chưa từng xuất hiện (Unseen States) & Zero-Division Guards
# ----------------------------------------------------------------------

def test_unseen_state_order1():
    # Huấn luyện chỉ trên các số 1..3 trong không gian 10 số
    draws = [[1, 2], [2, 3], [1, 3]]
    mc = HigherOrderMarkovChain(order=1, total_numbers=10, alpha=0.1)
    mc.fit(draws)

    # Số 9 chưa từng xuất hiện trong tập huấn luyện
    dist_unseen = mc.get_transition_distribution(9)
    assert len(dist_unseen) == 10
    assert math.isclose(sum(dist_unseen.values()), 1.0, abs_tol=1e-6)

    # Unseen state fallback về phân phối đều 1/10
    for w in range(1, 11):
        assert math.isclose(dist_unseen[w], 0.1, abs_tol=1e-5)


def test_unseen_state_order2_backoff():
    draws = [[1, 2], [3, 4], [5, 6]]
    mc = HigherOrderMarkovChain(order=2, total_numbers=10, alpha=0.1, lambdas=(0.7, 0.2, 0.1))
    mc.fit(draws)

    # Cặp (9, 10) hoàn toàn chưa từng xuất hiện
    dist_unseen_pair = mc.get_transition_distribution((9, 10))
    assert math.isclose(sum(dist_unseen_pair.values()), 1.0, abs_tol=1e-6)

    # Mọi xác suất phải strictly > 0 (không bị bẫy xác suất 0)
    for w in range(1, 11):
        assert dist_unseen_pair[w] > 0.0

    # Cặp (1, 9) trong đó 1 đã thấy nhưng 9 chưa thấy ở kỳ 2
    dist_half_unseen = mc.get_transition_distribution((1, 9))
    assert math.isclose(sum(dist_half_unseen.values()), 1.0, abs_tol=1e-6)
    for w in range(1, 11):
        assert dist_half_unseen[w] > 0.0


def test_zero_division_guard_alpha_zero():
    # Khi alpha = 0.0 (chỉ dùng Jelinek-Mercer và phân phối đều cho unseen)
    draws = [[1, 2], [3, 4], [5, 6]]
    mc = HigherOrderMarkovChain(order=1, total_numbers=10, alpha=0.0, lambdas=(0.8, 0.2))
    mc.fit(draws)

    # Kiểm tra số 8 (chưa từng thấy) không sinh lỗi chia cho 0
    dist = mc.get_transition_distribution(8)
    assert math.isclose(sum(dist.values()), 1.0, abs_tol=1e-6)


# ----------------------------------------------------------------------
# 5. Phương thức dự đoán: predict_next_probabilities & predict_top_k
# ----------------------------------------------------------------------

def test_predict_next_probabilities_order1(sample_draws_small):
    mc = HigherOrderMarkovChain(order=1, total_numbers=10)
    mc.fit(sample_draws_small)

    recent = [[5, 6, 9]]
    probs = mc.predict_next_probabilities(recent)

    assert isinstance(probs, dict)
    assert len(probs) == 10
    assert math.isclose(sum(probs.values()), 1.0, abs_tol=1e-6)

    for w in range(1, 11):
        assert 0.0 < probs[w] < 1.0


def test_predict_next_probabilities_order2(sample_draws_small):
    mc = HigherOrderMarkovChain(order=2, total_numbers=10)
    mc.fit(sample_draws_small)

    recent = [[3, 4, 8], [5, 6, 9]]
    probs = mc.predict_next_probabilities(recent)

    assert isinstance(probs, dict)
    assert len(probs) == 10
    assert math.isclose(sum(probs.values()), 1.0, abs_tol=1e-6)


def test_predict_next_probabilities_fallback():
    mc = HigherOrderMarkovChain(order=2, total_numbers=10)
    mc.fit([[1, 2], [3, 4], [5, 6]])

    # Trường hợp 1: recent_draws rỗng -> fallback về phân phối đều
    probs_empty = mc.predict_next_probabilities([])
    assert math.isclose(sum(probs_empty.values()), 1.0, abs_tol=1e-6)
    for w in range(1, 11):
        assert math.isclose(probs_empty[w], 0.1, abs_tol=1e-5)

    # Trường hợp 2: chỉ có 1 kỳ gần nhất trong mô hình bậc 2 -> fallback bậc 1
    probs_one_draw = mc.predict_next_probabilities([[5, 6]])
    assert math.isclose(sum(probs_one_draw.values()), 1.0, abs_tol=1e-6)


def test_predict_top_k(sample_draws_small):
    mc = HigherOrderMarkovChain(order=2, total_numbers=10)
    mc.fit(sample_draws_small)

    recent = [[3, 4, 8], [5, 6, 9]]
    top4 = mc.predict_top_k(recent, k=4)

    assert len(top4) == 4
    assert len(set(top4)) == 4  # Không trùng lặp
    for num in top4:
        assert 1 <= num <= 10


# ----------------------------------------------------------------------
# 6. Kiểm thử trên tập dữ liệu thực tế: data/power655.jsonl
# ----------------------------------------------------------------------

def test_fit_and_predict_real_power655(real_power655_draws):
    assert len(real_power655_draws) > 1000

    # 1. Huấn luyện bậc 1
    mc_o1 = HigherOrderMarkovChain(order=1, total_numbers=55, alpha=0.1)
    mc_o1.fit(real_power655_draws)
    assert mc_o1.verify_normalization_invariant() is True

    # 2. Huấn luyện bậc 2
    mc_o2 = HigherOrderMarkovChain(order=2, total_numbers=55, alpha=0.1)
    mc_o2.fit(real_power655_draws)
    assert mc_o2.verify_normalization_invariant() is True

    # 3. Dự đoán cho kỳ tiếp theo
    recent_2 = real_power655_draws[-2:]
    probs = mc_o2.predict_next_probabilities(recent_2)
    assert len(probs) == 55
    assert math.isclose(sum(probs.values()), 1.0, abs_tol=1e-6)

    top6 = mc_o2.predict_top_k(recent_2, k=6)
    assert len(top6) == 6
    assert len(set(top6)) == 6
    for b in top6:
        assert 1 <= b <= 55


# ----------------------------------------------------------------------
# 7. Xử lý ngoại lệ, Serialization & Xuất Ma Trận
# ----------------------------------------------------------------------

def test_unfitted_model_errors():
    mc = HigherOrderMarkovChain(order=2, total_numbers=55)

    with pytest.raises(RuntimeError, match="chưa được huấn luyện"):
        mc.get_transition_probability((1, 2), 3)

    with pytest.raises(RuntimeError, match="chưa được huấn luyện"):
        mc.get_transition_distribution((1, 2))

    with pytest.raises(RuntimeError, match="chưa được huấn luyện"):
        mc.predict_next_probabilities([[1, 2, 3]])

    with pytest.raises(RuntimeError, match="chưa được huấn luyện"):
        mc.verify_normalization_invariant()


def test_invalid_target_number():
    mc = HigherOrderMarkovChain(order=1, total_numbers=10)
    mc.fit([[1, 2], [3, 4]])

    # next_num ngoài dải 1..10 trả về 0.0
    assert mc.get_transition_probability(1, 0) == 0.0
    assert mc.get_transition_probability(1, 11) == 0.0


def test_serialization_and_json(sample_draws_small, tmp_path):
    mc = HigherOrderMarkovChain(order=2, total_numbers=10)
    mc.fit(sample_draws_small)

    d = mc.to_dict()
    assert d["order"] == 2
    assert d["total_numbers"] == 10
    assert d["is_fitted"] is True
    assert "matrix_order1" in d

    json_file = tmp_path / "markov_test.json"
    mc.save_json(json_file)
    assert json_file.exists()

    with open(json_file, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["order"] == 2
    assert loaded["total_numbers"] == 10
