#!/usr/bin/env python3
"""
VietLot AI — Data & Temporal Utilities
──────────────────────────────────────
Cung cấp các hàm chuẩn hóa ngày tháng và cắt bỏ dữ liệu tương lai (future data cutoff)
nhằm ngăn ngừa look-ahead bias và data leakage trong huấn luyện và backtesting mô hình.
"""
from datetime import datetime, timezone, timedelta, date
from typing import List, Dict, Any, Union, Optional

VN_TZ = timezone(timedelta(hours=7))


def parse_date(val: Union[str, date, datetime], default_year: Optional[int] = None) -> str:
    """
    Chuẩn hóa chuỗi hoặc đối tượng ngày sang định dạng chuẩn YYYY-MM-DD.

    Hỗ trợ:
      - datetime.date hoặc datetime.datetime -> YYYY-MM-DD
      - YYYY-MM-DD, YYYY/MM/DD (vd: '2026-12-11', '2026/12/11')
      - DD/MM/YYYY, DD-MM-YYYY (vd: '11/12/2026', '11-12-2026')
      - DD/MM hoặc D/M (vd: '11/12' -> dùng default_year hoặc năm hiện tại VN)

    Returns:
        str: Chuỗi ngày theo chuẩn 'YYYY-MM-DD'.
    """
    if isinstance(val, (datetime, date)):
        return val.strftime("%Y-%m-%d")

    s = str(val).strip()
    if not s:
        raise ValueError("Giá trị ngày không được để trống")

    if "T" in s:
        s = s.split("T")[0]
    if " " in s:
        s = s.split(" ")[0]

    s = s.replace("/", "-")
    parts = s.split("-")

    if len(parts) == 3:
        # Trường hợp YYYY-MM-DD
        if len(parts[0]) == 4:
            return f"{int(parts[0]):04d}-{int(parts[1]):02d}-{int(parts[2]):02d}"
        # Trường hợp DD-MM-YYYY
        elif len(parts[2]) == 4:
            return f"{int(parts[2]):04d}-{int(parts[1]):02d}-{int(parts[0]):02d}"
    elif len(parts) == 2:
        # Trường hợp định dạng ngắn ngày/tháng (DD/MM theo văn hóa VN)
        year = default_year or datetime.now(VN_TZ).year
        day = int(parts[0])
        month = int(parts[1])
        return f"{int(year):04d}-{month:02d}-{day:02d}"

    raise ValueError(f"Không thể phân tích định dạng ngày: '{val}'")


def infer_default_year(rows: List[Dict[str, Any]]) -> int:
    """
    Suy luận năm điển hình từ danh sách các bản ghi lượt quay.
    Nếu không tìm thấy, fallback về năm hiện tại theo múi giờ Việt Nam.
    """
    for r in rows:
        d = str(r.get("date") or r.get("draw_date") or "").strip()
        if len(d) >= 4 and d[:4].isdigit():
            return int(d[:4])
        if "/" in d:
            parts = d.split("/")
            if len(parts) == 3 and len(parts[2]) == 4 and parts[2].isdigit():
                return int(parts[2])
    return datetime.now(VN_TZ).year


def cutoff_future_data(
    rows: List[Dict[str, Any]],
    cutoff_date: Union[str, date, datetime],
    inclusive: bool = False,
    default_year: Optional[int] = None,
    reverse: bool = True,
) -> List[Dict[str, Any]]:
    """
    Cắt bỏ toàn bộ dữ liệu tương lai (future data) từ mốc cutoff_date trở đi
    để ngăn ngừa triệt để hiện tượng look-ahead bias và rò rỉ dữ liệu (data leakage).

    Quy tắc nghiệp vụ:
      - inclusive = False (mặc định cho kiểm thử dự đoán kỳ cutoff_date):
        Cắt toàn bộ dữ liệu từ cutoff_date trở đi (loại bỏ date >= cutoff_date).
        Chỉ giữ lại các lượt quay diễn ra trước thời điểm cutoff (date < cutoff_date).
      - inclusive = True (dự đoán kỳ kế tiếp sau khi đã biết kết quả cutoff_date):
        Cắt toàn bộ dữ liệu sau cutoff_date (loại bỏ date > cutoff_date).
        Giữ lại các lượt quay đến hết ngày cutoff (date <= cutoff_date).

    Args:
        rows: Danh sách các bản ghi lượt quay (mỗi bản ghi có trường 'date' hoặc 'draw_date').
        cutoff_date: Mốc ngày cutoff (hỗ trợ 'YYYY-MM-DD', 'DD/MM/YYYY', 'DD/MM', date/datetime).
        inclusive: Cờ bao gồm ngày cutoff hay loại bỏ từ ngày cutoff.
        default_year: Năm mặc định nếu cutoff_date ở dạng thiếu năm ('DD/MM').
        reverse: Nếu True, sắp xếp danh sách kết quả mới -> cũ theo (date, id).

    Returns:
        List[Dict[str, Any]]: Danh sách bản ghi lịch sử hợp lệ, sạch 100% dữ liệu tương lai.
    """
    if not rows:
        return []

    year = default_year or infer_default_year(rows)
    cutoff_iso = parse_date(cutoff_date, default_year=year)

    filtered = []
    for r in rows:
        raw_d = r.get("date") or r.get("draw_date")
        if not raw_d:
            continue
        try:
            d_iso = parse_date(raw_d, default_year=year)
        except ValueError:
            continue

        if inclusive:
            if d_iso <= cutoff_iso:
                filtered.append(r)
        else:
            if d_iso < cutoff_iso:
                filtered.append(r)

    if reverse:
        filtered.sort(key=lambda x: (str(x.get("date", "")), str(x.get("id", ""))), reverse=True)

    return filtered
