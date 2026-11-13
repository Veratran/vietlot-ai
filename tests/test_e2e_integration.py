#!/usr/bin/env python3
"""
Test Suite: Milestone 5 E2E Pipeline Integration Test
──────────────────────────────────────────────────────
Kiểm thử tích hợp đầu cuối (End-to-End Pipeline Integration) cho toàn bộ 4 module:
  - Higher-Order Markov Chain Engine (cpeoples/powerpredict)
  - Gaussian Z-Score Distribution & Cycle Analysis Engine (khiemdoan/xsmb)
  - Monte Carlo Baseline Simulator (kenchudigital/ML-SixMark-Lab)
  - Formal Finite State Machine & TLA+ Invariants (/statechart)

Được tích hợp đầy đủ vào:
  - scripts/build_predictions.py
  - scripts/build_api.py
"""
import copy
import json
import pathlib
import sys
import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
SCRIPTS_DIR = ROOT / "scripts"
DATA_DIR = ROOT / "data"
API_DIR = ROOT / "api"
SPEC_DIR = ROOT / "spec"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from data_utils import cutoff_future_data, parse_date
from markov_chain import HigherOrderMarkovChain
from gaussian_cycle import GaussianCycleAnalyzer
from monte_carlo_baseline import MonteCarloBaselineSimulator
from pipeline_fsm import LotteryPipelineStateMachine, PipelineState
import build_predictions
import build_api


@pytest.fixture
def sample_draws_sequence():
    """Tạo chuỗi lượt quay giả lập 35 kỳ với ngày tăng dần."""
    draws = []
    for day in range(1, 36):
        date_str = f"2026-11-{day:02d}" if day <= 30 else f"2026-12-{day-30:02d}"
        draws.append({
            "id": f"{day:05d}",
            "date": date_str,
            "nums": [
                ((day * 3 + 1) % 55) + 1,
                ((day * 5 + 7) % 55) + 1,
                ((day * 7 + 13) % 55) + 1,
                ((day * 11 + 17) % 55) + 1,
                ((day * 13 + 19) % 55) + 1,
                ((day * 17 + 23) % 55) + 1,
            ],
            "spec": ((day * 2 + 5) % 55) + 1,
        })
    return draws


# ==============================================================================
# TEST SUITE 1: Tích Hợp scripts/build_predictions.py
# ==============================================================================

def test_predictions_pipeline_execution(tmp_path):
    """
    Kiểm tra thực thi build_predictions.py với file tạm,
    đảm bảo các trường markov, gaussian, monte_carlo, fsm_verification được sinh đầy đủ.
    """
    tmp_pred = tmp_path / "predictions_e2e_test.jsonl"
    
    # Chạy build với file đích tạm thời trên game power645 để kiểm thử tốc độ cao
    build_predictions.build(
        cutoff_date="2026-11-01",
        inclusive=True,
        pred_file=str(tmp_pred),
        game="power645",
        mc_simulations=100,
    )
    
    assert tmp_pred.exists(), "Tệp predictions tạm không được tạo!"
    records = build_predictions.load_jsonl(tmp_pred)
    assert len(records) > 0, "Không có bản ghi dự đoán nào được sinh ra!"
    
    for r in records:
        assert r["game"] == "power645"
        assert "based_on_id" in r
        assert "picks" in r
        assert "ai" in r["picks"]
        
        # Kiểm tra đầy đủ 4 động cơ tích hợp
        assert "markov" in r, "Thiếu trường markov"
        assert "gaussian" in r, "Thiếu trường gaussian"
        assert "monte_carlo" in r, "Thiếu trường monte_carlo"
        assert "fsm_verification" in r, "Thiếu trường fsm_verification"
        
        # 1. Markov verification
        assert "order_1_prob" in r["markov"]
        assert "order_2_prob" in r["markov"]
        assert "top_picks" in r["markov"]
        assert r["markov"]["normalization_invariant_verified"] is True
        
        # 2. Gaussian verification
        assert "z_scores" in r["gaussian"]
        assert "outlier_hot" in r["gaussian"]
        assert "outlier_cold" in r["gaussian"]
        assert "stationary" in r["gaussian"]
        assert "cycle_metrics" in r["gaussian"]
        
        # 3. Monte Carlo verification
        assert "simulations" in r["monte_carlo"]
        assert "mean_hits_per_draw" in r["monte_carlo"]
        assert "roi_mean" in r["monte_carlo"]
        
        # 4. FSM verification
        fsm_meta = r["fsm_verification"]
        assert fsm_meta["zero_future_leakage_verified"] is True
        assert fsm_meta["bankroll_solvency_verified"] is True
        assert fsm_meta["bankroll"] >= 0


def test_predictions_temporal_cutoff_invariant(tmp_path):
    """
    Khẳng định bất biến ZeroFutureLeakage:
    Khi build_predictions chạy với cutoff_date, mọi record đều có based_on_date < cutoff_date.
    """
    tmp_pred = tmp_path / "pred_cutoff_invariant.jsonl"
    cutoff = "2026-06-01"
    
    build_predictions.build(
        cutoff_date=cutoff,
        inclusive=False,
        pred_file=str(tmp_pred),
        game="power645",
        mc_simulations=100,
    )
    
    records = build_predictions.load_jsonl(tmp_pred)
    assert len(records) > 0
    for r in records:
        if r.get("based_on_date"):
            assert r["based_on_date"] < cutoff, (
                f"LỖI RÒ RỈ: based_on_date {r['based_on_date']} >= cutoff {cutoff}"
            )


# ==============================================================================
# TEST SUITE 2: Tích Hợp scripts/build_api.py
# ==============================================================================

def test_build_api_generates_quantitative_endpoints():
    """
    Kiểm tra build_api.py xuất đúng các endpoint JSON định lượng:
    - api/markov.json, api/gaussian.json, api/monte_carlo.json
    - api/{game}/markov.json, api/{game}/gaussian.json, api/{game}/monte_carlo.json
    """
    # 1. Các file tổng hợp toàn hệ thống
    assert (API_DIR / "markov.json").exists(), "Thiếu api/markov.json"
    assert (API_DIR / "gaussian.json").exists(), "Thiếu api/gaussian.json"
    assert (API_DIR / "monte_carlo.json").exists(), "Thiếu api/monte_carlo.json"
    
    with (API_DIR / "markov.json").open("r", encoding="utf-8") as f:
        markov_data = json.load(f)
        assert "games" in markov_data
        assert "power655" in markov_data["games"]
        assert "top_candidates" in markov_data["games"]["power655"]
        assert markov_data["games"]["power655"]["normalization_invariant_verified"] is True
        
    with (API_DIR / "gaussian.json").open("r", encoding="utf-8") as f:
        gauss_data = json.load(f)
        assert "games" in gauss_data
        assert "power655" in gauss_data["games"]
        assert "outlier_hot" in gauss_data["games"]["power655"]
        assert "band_2sigma" in gauss_data["games"]["power655"]
        
    with (API_DIR / "monte_carlo.json").open("r", encoding="utf-8") as f:
        mc_data = json.load(f)
        assert "games" in mc_data
        assert "power655" in mc_data["games"]
        assert "mean_hits_per_draw" in mc_data["games"]["power655"]
        assert "roi_mean" in mc_data["games"]["power655"]
        
    # 2. Các file endpoint theo từng game
    for game in ["power655", "power645", "power535", "keno"]:
        game_api = API_DIR / game
        assert (game_api / "markov.json").exists(), f"Thiếu {game}/markov.json"
        assert (game_api / "gaussian.json").exists(), f"Thiếu {game}/gaussian.json"
        assert (game_api / "monte_carlo.json").exists(), f"Thiếu {game}/monte_carlo.json"


# ==============================================================================
# TEST SUITE 3: FSM State Machine & TLA+ Invariant Integration
# ==============================================================================

def test_fsm_lifecycle_state_transitions(sample_draws_sequence):
    """
    Kiểm tra FSM chuyển trạng thái tuần tự và bảo toàn bất biến:
    IDLE -> INGESTED -> CUTOFF_VERIFIED -> FEATURES_READY -> PREDICTED -> EVALUATED
    """
    fsm = LotteryPipelineStateMachine(initial_bankroll=100.0, ticket_cost=10.0, pick_size=6)
    assert fsm.state == PipelineState.IDLE
    
    # 1. Ingest
    cutoff = "2026-11-20"
    fsm_data = [{"date": d["date"], "result": d["nums"]} for d in sample_draws_sequence]
    fsm.ingest_data(fsm_data, cutoff_date=cutoff)
    assert fsm.state == PipelineState.INGESTED
    
    # 2. Cutoff verify
    fsm.verify_cutoff()
    assert fsm.state == PipelineState.CUTOFF_VERIFIED
    assert not fsm.future_leak
    assert all(d["date"] < cutoff for d in fsm.loaded_draws)
    
    # 3. Feature extraction
    hist_nums = [d["result"] for d in fsm.loaded_draws]
    mc = HigherOrderMarkovChain(order=2, total_numbers=55)
    mc.fit(hist_nums)
    gauss = GaussianCycleAnalyzer(total_numbers=55)
    gauss.fit(hist_nums)
    sim = MonteCarloBaselineSimulator(total_numbers=55, pick_count=6, game_type="power655")
    
    fsm.extract_features(markov_model=mc, gaussian_model=gauss, mc_model=sim)
    assert fsm.state == PipelineState.FEATURES_READY
    assert fsm.markov_state == "CALCULATED"
    assert fsm.zscore_state == "COMPUTED"
    
    # 4. Generate prediction
    fsm.generate_prediction(predictor_func=lambda d: [1, 2, 3, 4, 5, 6])
    assert fsm.state == PipelineState.PREDICTED
    assert len(fsm.picks) == 6
    assert fsm.picks_count == 6
    assert fsm.bankroll == 90.0  # Đã trừ 10 tiền vé
    
    # 5. Evaluate
    fsm.evaluate_result(actual_draw=[1, 2, 3, 10, 20, 30], prize_payout=50)
    assert fsm.state == PipelineState.EVALUATED
    assert fsm.bankroll == 140.0  # 90 + 50
    assert fsm.current_round == 1


def test_fsm_insolvency_safety_halt():
    """
    Kiểm tra cơ chế an toàn: nếu vốn không đủ mua vé, FSM lập tức chuyển sang HALTED_INSOLVENT
    và trả về danh sách rỗng một cách an toàn mà không bị âm vốn.
    """
    fsm = LotteryPipelineStateMachine(initial_bankroll=5.0, ticket_cost=10.0, pick_size=6)
    fsm.ingest_data([{"date": "2026-11-01", "result": [1, 2, 3, 4, 5, 6]}], cutoff_date="2026-11-02")
    fsm.verify_cutoff()
    fsm.extract_features("dummy_markov", "dummy_gauss", "dummy_mc")
    
    # Gọi hàm dự đoán khi không đủ tiền
    picks = fsm.generate_prediction(predictor_func=lambda d: [1, 2, 3, 4, 5, 6])
    assert picks == []
    assert fsm.state == PipelineState.HALTED_INSOLVENT
    assert fsm.bankroll == 5.0  # Không bị âm
