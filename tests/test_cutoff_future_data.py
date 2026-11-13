#!/usr/bin/env python3
"""
Test Suite: Cắt bỏ dữ liệu tương lai (Future Data Cutoff & Zero Data Leakage)
─────────────────────────────────────────────────────────────────────────────
Kiểm thử tính năng cô lập dữ liệu theo trục thời gian, đảm bảo mô hình AI
và các thuật toán dự đoán / backtest không bị rò rỉ dữ liệu tương lai (look-ahead bias).

Kịch bản kiểm thử trọng tâm (Test 1):
  - Tập dữ liệu gồm 30 ngày của tháng 12 (2026-12-01 đến 2026-12-30).
  - Khi kiểm thử tại mốc ngày 11/12 (target/test date 11/12):
    • Chế độ Exclusive (mặc định): Cắt bỏ toàn bộ dữ liệu từ 11/12 trở đi (11/12 đến 30/12).
      Chỉ giữ lại các ngày trước đó (01/12 đến 10/12, gồm 10 ngày).
    • Chế độ Inclusive: Giữ lại đến hết ngày 11/12 (11 ngày), cắt sạch từ 12/12 đến 30/12 (19 ngày).
"""
import sys
import pathlib
import json
import subprocess
from datetime import date, datetime
import pytest

# Thiết lập đường dẫn import tới thư mục scripts
ROOT = pathlib.Path(__file__).resolve().parent.parent
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from data_utils import cutoff_future_data, parse_date, infer_default_year
from build_predictions import (
    adapt,
    compute_scores,
    predict_num,
    ACFG,
    FILES,
    load_jsonl,
)


@pytest.fixture
def december_30_days_data():
    """
    Tạo fixture giả lập dữ liệu 30 ngày liên tiếp của tháng 12/2026 (2026-12-01 -> 2026-12-30).
    Mỗi ngày là 1 lượt quay hoàn chỉnh có ID, bộ số chính và số đặc biệt.
    """
    rows = []
    for day in range(1, 31):
        d_str = f"2026-12-{day:02d}"
        rows.append({
            "id": f"D{day:04d}",
            "date": d_str,
            "result": [
                ((day * 3 + 1) % 55) + 1,
                ((day * 5 + 7) % 55) + 1,
                ((day * 7 + 13) % 55) + 1,
                ((day * 11 + 17) % 55) + 1,
                ((day * 13 + 19) % 55) + 1,
                ((day * 17 + 23) % 55) + 1,
                ((day * 2 + 5) % 55) + 1,
            ],
            "winners": {"Jackpot 1": 0, "Jackpot 2": 0}
        })
    return rows


# ==============================================================================
# TEST CASE 1: Cắt data future - 30 ngày tháng 12, test ngày 11/12
# ==============================================================================

def test_cutoff_30_days_december_exclusive(december_30_days_data):
    """
    KỊCH BẢN CHÍNH (User Request):
    Dữ liệu có 30 ngày của tháng 12.
    Test ngày 11/12 (dự đoán lượt quay ngày 11/12):
    Cắt hết future data từ 11/12 trở đi (inclusive=False).
    - Kỳ vọng:
      • 20 ngày từ 11/12 đến 30/12 bị cắt bỏ hoàn toàn.
      • Chỉ giữ lại đúng 10 ngày từ 01/12 đến 10/12.
      • Ngày lớn nhất trong tập còn lại là 2026-12-10 (< 2026-12-11).
    """
    cutoff_date = "2026-12-11"
    filtered = cutoff_future_data(december_30_days_data, cutoff_date=cutoff_date, inclusive=False)

    # 1. Khẳng định số lượng bản ghi còn lại chính xác là 10 ngày (01/12 -> 10/12)
    assert len(filtered) == 10, f"Kỳ vọng 10 ngày (01-10/12), thực tế nhận {len(filtered)}"

    # 2. Khẳng định mọi bản ghi đều có ngày nhỏ hơn ngày cutoff (11/12)
    assert all(r["date"] < "2026-12-11" for r in filtered)

    # 3. Khẳng định ngày gần nhất (lớn nhất) trong tập kết quả là ngày 10/12
    max_date = max(r["date"] for r in filtered)
    assert max_date == "2026-12-10", f"Ngày lớn nhất phải là 2026-12-10, thực tế: {max_date}"

    # 4. Khẳng định thứ tự mặc định là mới nhất -> cũ nhất (reverse=True)
    assert filtered[0]["date"] == "2026-12-10"
    assert filtered[-1]["date"] == "2026-12-01"

    # 5. Khẳng định tuyệt đối KHÔNG có bất kỳ ngày nào từ 11/12 đến 30/12 tồn tại
    pruned_dates = [f"2026-12-{d:02d}" for d in range(11, 31)]
    remaining_dates = {r["date"] for r in filtered}
    for p_date in pruned_dates:
        assert p_date not in remaining_dates, f"LỖI RÒ RỈ DỮ LIỆU: Ngày tương lai {p_date} vẫn còn trong tập dữ liệu!"


def test_cutoff_30_days_december_inclusive(december_30_days_data):
    """
    Kịch bản Inclusive:
    Cắt tại cuối ngày 11/12 (đã có kết quả ngày 11/12, dự đoán cho ngày 12/12).
    - Kỳ vọng:
      • 19 ngày từ 12/12 đến 30/12 bị cắt bỏ.
      • Giữ lại 11 ngày từ 01/12 đến 11/12.
      • Ngày lớn nhất trong tập còn lại là 2026-12-11.
    """
    cutoff_date = "2026-12-11"
    filtered = cutoff_future_data(december_30_days_data, cutoff_date=cutoff_date, inclusive=True)

    assert len(filtered) == 11, f"Kỳ vọng 11 ngày (01-11/12), thực tế nhận {len(filtered)}"
    assert all(r["date"] <= "2026-12-11" for r in filtered)
    assert max(r["date"] for r in filtered) == "2026-12-11"
    assert filtered[0]["date"] == "2026-12-11"

    # Không còn dữ liệu từ ngày 12/12 trở đi
    remaining_dates = {r["date"] for r in filtered}
    for day in range(12, 31):
        future_day = f"2026-12-{day:02d}"
        assert future_day not in remaining_dates, f"Ngày tương lai {future_day} bị rò rỉ!"


def test_cutoff_with_vietnamese_date_shorthand(december_30_days_data):
    """
    Kiểm tra nhập ngày dạng chuỗi tiếng Việt phổ biến: '11/12' hoặc '11/12/2026'.
    Hàm parse_date và cutoff_future_data phải tự động nhận diện và cắt đồng nhất.
    """
    # Test định dạng '11/12' (tự động suy luận năm 2026 từ dữ liệu)
    res_shorthand = cutoff_future_data(december_30_days_data, "11/12", inclusive=False)
    assert len(res_shorthand) == 10
    assert max(r["date"] for r in res_shorthand) == "2026-12-10"

    # Test định dạng '11/12/2026' (đầy đủ ngày/tháng/năm)
    res_full = cutoff_future_data(december_30_days_data, "11/12/2026", inclusive=False)
    assert len(res_full) == 10

    # Test định dạng ISO '2026-12-11'
    res_iso = cutoff_future_data(december_30_days_data, "2026-12-11", inclusive=False)
    assert len(res_iso) == 10

    # Kết quả của cả 3 cách viết phải hoàn toàn giống hệt nhau
    assert res_shorthand == res_full == res_iso


# ==============================================================================
# TEST CASE 2: Chứng minh toán học triệt tiêu Data Leakage (Leakage Invariant)
# ==============================================================================

def test_zero_future_leakage_in_prediction_scoring(december_30_days_data):
    """
    CHỨNG MINH TOÁN HỌC KHÔNG CÓ DATA LEAKAGE:
    1. Lấy dữ liệu chuẩn lịch sử (01/12 đến 10/12), tính điểm và dự đoán.
    2. Tạo một tập dữ liệu bị 'đầu độc tương lai' (future data contamination):
       Thêm các lượt quay từ 11/12 đến 30/12 với các số cực đoan cố ý lặp lại.
    3. Áp dụng cutoff_future_data('11/12', inclusive=False) lên tập bị đầu độc.
    4. Tính điểm và dự đoán trên tập sau khi cắt.
    5. Khẳng định: Điểm số, bộ số AI, Hot, Cold, Correlation dự đoán
       HOÀN TOÀN TRÙNG KHỚP 100% với tập lịch sử sạch.
       => Chứng minh dữ liệu tương lai dù có biến động thế nào cũng không thể
          ảnh hưởng hay rò rỉ vào kết quả dự đoán kỳ 11/12.
    """
    ac = ACFG["power655"]

    # 1. Dataset sạch: chỉ có 10 ngày trước 11/12 (01/12 -> 10/12) sắp xếp mới -> cũ
    clean_history = cutoff_future_data(
        [r for r in december_30_days_data if r["date"] < "2026-12-11"],
        "2026-12-11",
        inclusive=False,
    )
    clean_draws = adapt(clean_history, ac)
    clean_scores, clean_spec_sc = compute_scores(clean_draws, ac)
    clean_picks, clean_spec = predict_num(clean_draws, ac)

    # 2. Dataset bị đầu độc dữ liệu tương lai (ngày 11/12 đến 30/12 bị cố ý nhồi số 50, 51, 52, 53, 54, 55)
    contaminated_data = [dict(r) for r in december_30_days_data]
    for r in contaminated_data:
        if r["date"] >= "2026-12-11":
            r["result"] = [50, 51, 52, 53, 54, 55, 50]  # Giả mạo tương lai

    # 3. Cắt dữ liệu tương lai tại mốc 11/12
    sanitized_rows = cutoff_future_data(contaminated_data, "2026-12-11", inclusive=False)
    sanitized_draws = adapt(sanitized_rows, ac)
    sanitized_scores, sanitized_spec_sc = compute_scores(sanitized_draws, ac)
    sanitized_picks, sanitized_spec = predict_num(sanitized_draws, ac)

    # 4. Kiểm tra đẳng thức tuyệt đối
    assert clean_picks["ai"] == sanitized_picks["ai"], "AI picks bị sai lệch do rò rỉ future data!"
    assert clean_picks["hot"] == sanitized_picks["hot"], "Hot picks bị sai lệch!"
    assert clean_picks["cold"] == sanitized_picks["cold"], "Cold picks bị sai lệch!"
    assert clean_picks["corr"] == sanitized_picks["corr"], "Corr picks bị sai lệch!"
    assert clean_spec == sanitized_spec, "Special ball bị sai lệch!"

    # Kiểm tra điểm số AI chi tiết cho từng con số
    for num in range(1, ac["max"] + 1):
        assert abs(clean_scores[num]["ai"] - sanitized_scores[num]["ai"]) < 1e-9


# ==============================================================================
# TEST CASE 3: Kiểm định tính cô lập thời gian trong vòng lặp Backtest
# ==============================================================================

def test_backtest_temporal_isolation_invariant(december_30_days_data):
    """
    Kiểm tra vòng lặp backtest:
    Khi dự đoán cho lượt quay đích (target draw T), cửa sổ lịch sử huấn luyện (hist)
    bắt buộc CHỈ được phép chứa các lượt quay có ngày quay diễn ra TRƯỚC target draw.
    Tuyệt đối không được chứa chính target draw hoặc bất kỳ lượt quay tương lai nào.
    """
    ac = ACFG["power655"]
    sorted_rows = sorted(december_30_days_data, key=lambda r: (r["date"], r["id"]), reverse=True)
    draws = adapt(sorted_rows, ac)  # sắp xếp mới -> cũ (30/12 -> 01/12)

    # Giả lập vòng lặp backtest trong build_predictions.py
    for j in range(len(draws) - 10):
        target = draws[j]
        hist = draws[j + 1 : j + 1 + 50]

        target_date = target["date"]
        # Khẳng định 1: Target date không nằm trong hist
        hist_dates = [h["date"] for h in hist]
        assert target_date not in hist_dates, f"Kỳ đích {target_date} bị đưa vào dữ liệu huấn luyện backtest!"

        # Khẳng định 2: Toàn bộ ngày trong hist phải nhỏ hơn target_date
        for h_date in hist_dates:
            assert h_date < target_date, f"Lượt quay lịch sử {h_date} không được lớn hơn hoặc bằng kỳ đích {target_date}!"


# ==============================================================================
# TEST CASE 4: Kiểm thử trên dữ liệu thực tế (Real Dataset) của Power 6/55
# ==============================================================================

def test_cutoff_on_real_power655_dataset():
    """
    Kiểm tra cắt dữ liệu trên tệp thật data/power655.jsonl.
    Cắt tại mốc '2026-06-01' (01/06/2026):
    - Toàn bộ bản ghi trả về phải có date < 2026-06-01.
    - Không có bất kỳ bản ghi nào của nửa cuối năm 2026 lọt vào.
    """
    power_file = ROOT / "data" / FILES["power655"]
    if not power_file.exists():
        pytest.skip(f"Không tìm thấy file {power_file}")

    rows = load_jsonl(power_file)
    assert len(rows) > 100, "Dữ liệu Power 6/55 phải có hơn 100 bản ghi"

    cutoff_date = "2026-06-01"
    filtered = cutoff_future_data(rows, cutoff_date=cutoff_date, inclusive=False)

    assert len(filtered) > 0, "Phải có các bản ghi lịch sử trước 01/06/2026"
    assert all(r["date"] < cutoff_date for r in filtered)

    # Bản ghi mới nhất trong tập đã cắt phải nhỏ hơn 2026-06-01
    latest_date = filtered[0]["date"]
    assert latest_date < cutoff_date
    print(f"\n[Real Dataset Power655] Cắt tại {cutoff_date} -> Bản ghi mới nhất: {latest_date}, Tổng còn: {len(filtered)}")


# ==============================================================================
# TEST CASE 5: Kiểm tra các trường hợp biên & dữ liệu bất thường (Edge Cases)
# ==============================================================================

def test_cutoff_edge_cases(december_30_days_data):
    """Kiểm tra độ bền vững với các trường hợp biên."""
    # 1. Danh sách rỗng
    assert cutoff_future_data([], "2026-12-11") == []

    # 2. Cutoff trước toàn bộ dữ liệu -> Trả về rỗng
    early_cut = cutoff_future_data(december_30_days_data, "2026-11-01", inclusive=False)
    assert len(early_cut) == 0

    # 3. Cutoff sau toàn bộ dữ liệu -> Giữ lại tất cả 30 ngày
    late_cut = cutoff_future_data(december_30_days_data, "2027-01-01", inclusive=False)
    assert len(late_cut) == 30

    # 4. Dữ liệu đầu vào bị đảo xáo thứ tự ngày (unsorted)
    shuffled_data = list(reversed(december_30_days_data))
    filtered_shuffled = cutoff_future_data(shuffled_data, "2026-12-11", inclusive=False, reverse=True)
    assert len(filtered_shuffled) == 10
    assert filtered_shuffled[0]["date"] == "2026-12-10"
    assert filtered_shuffled[-1]["date"] == "2026-12-01"

    # 5. Dữ liệu chứa bản ghi lỗi (thiếu date hoặc date rỗng)
    dirty_data = [
        {"id": "BAD1", "date": None},
        {"id": "BAD2", "date": ""},
        {"id": "OK1", "date": "2026-12-05"},
    ]
    res_dirty = cutoff_future_data(dirty_data, "2026-12-11", inclusive=False)
    assert len(res_dirty) == 1
    assert res_dirty[0]["id"] == "OK1"


# ==============================================================================
# TEST CASE 6: Kiểm tra chạy qua dòng lệnh CLI (scripts/build_predictions.py)
# ==============================================================================

def test_cli_build_predictions_cutoff(tmp_path):
    """
    Kiểm tra chạy script build_predictions.py từ dòng lệnh với tham số --cutoff-date.
    Ghi kết quả ra tệp tạm và kiểm tra tất cả bản ghi tạo ra đều dựa trên ngày trước cutoff.
    """
    tmp_pred_file = tmp_path / "test_predictions.jsonl"
    cutoff_arg = "2026-05-01"

    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "build_predictions.py"),
        "--cutoff-date", cutoff_arg,
        "--pred-file", str(tmp_pred_file),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    assert result.returncode == 0, f"Script gặp lỗi khi chạy: {result.stderr}"

    assert tmp_pred_file.exists(), "File kết quả dự đoán không được tạo!"
    pred_records = load_jsonl(tmp_pred_file)
    assert len(pred_records) > 0, "Phải có ít nhất 1 bản ghi dự đoán được tạo"

    # Kiểm tra mọi dự đoán đều dựa trên các lượt quay trước cutoff_arg
    for rec in pred_records:
        based_on_date = rec.get("based_on_date")
        if based_on_date:
            assert based_on_date < cutoff_arg, f"Dự đoán dựa trên ngày tương lai: {based_on_date} >= {cutoff_arg}"
