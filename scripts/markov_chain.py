#!/usr/bin/env python3
"""
VietLot AI — Higher-Order Markov Chain Transition Matrix Engine
────────────────────────────────────────────────────────────────
Triển khai mô hình Markov Chain bậc cao (k=1 và k=2) phân tích xác suất chuyển tiếp
giữa các kỳ quay số Vietlott.

Đặc tính kỹ thuật:
  • Bậc k=1: Xác suất chuyển tiếp cặp số u -> v giữa 2 kỳ liên tiếp.
  • Bậc k=2: Xác suất chuyển tiếp bộ 2 số (u1, u2) -> v qua 3 kỳ liên tiếp.
  • Bất biến chuẩn hóa (Normalization Invariant): Tổng xác suất mỗi hàng trạng thái
    điều kiện PHẢI bằng 1.0 tuyệt đối (sai số nổi < 1e-6).
  • Làm mịn Lidstone / Laplace (alpha=0.1) ngăn chặn xác suất 0.
  • Nội suy Jelinek-Mercer & Backoff về bậc thấp hơn và phân phối đều (uniform)
    cho các trạng thái chưa từng xuất hiện (unseen states).
  • Zero-dependency core: Chỉ sử dụng thư viện chuẩn Python (math, collections, typing, json).
"""

from collections import Counter, defaultdict
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


class HigherOrderMarkovChain:
    """Mô hình Markov Chain bậc cao k in {1, 2} cho chuỗi kết quả xổ số."""

    def __init__(
        self,
        order: int = 2,
        total_numbers: int = 55,
        alpha: float = 0.1,
        lambdas: Tuple[float, ...] = (0.7, 0.2, 0.1),
    ):
        """Khởi tạo mô hình Markov bậc cao.

        Args:
            order: Bậc Markov (1 hoặc 2). Mặc định là 2.
            total_numbers: Tổng số bóng trong không gian số (vd: 55 cho Power 6/55, 45 cho Mega 6/45).
            alpha: Hệ số làm mịn Lidstone (Laplace smoothing). Mặc định 0.1.
            lambdas: Trọng số nội suy Jelinek-Mercer.
                     - Cho bậc 2: (lambda2, lambda1, lambda0) cho (bậc 2, bậc 1, phân phối đều).
                     - Cho bậc 1: (lambda1, lambda0) cho (bậc 1, phân phối đều).
        """
        if order not in (1, 2):
            raise ValueError(f"Bậc Markov không hợp lệ: order={order}. Chỉ hỗ trợ bậc 1 và bậc 2.")
        if total_numbers < 2:
            raise ValueError(f"Tổng số bóng total_numbers={total_numbers} phải >= 2.")
        if alpha < 0:
            raise ValueError(f"Hệ số alpha={alpha} không được âm.")

        self.order = order
        self.total_numbers = total_numbers
        self.alpha = float(alpha)

        # Chuẩn hóa trọng số Jelinek-Mercer lambdas
        self.lambdas = self._normalize_lambdas(order, lambdas)

        # Cấu trúc đếm tần suất chuyển tiếp
        # Bậc 1: u -> Counter(v: count)
        self.transitions_order1: Dict[int, Counter] = defaultdict(Counter)
        self.state_totals_order1: Dict[int, int] = defaultdict(int)

        # Bậc 2: (u1, u2) -> Counter(w: count)
        self.transitions_order2: Dict[Tuple[int, int], Counter] = defaultdict(Counter)
        self.state_totals_order2: Dict[Tuple[int, int], int] = defaultdict(int)

        # Đếm tần suất đơn lẻ (unigram)
        self.unigram_counts: Counter = Counter()
        self.total_draws_fitted: int = 0
        self.is_fitted: bool = False

        # Cache phân phối có điều kiện để tối ưu hóa truy vấn
        self._distribution_cache: Dict[Any, Dict[int, float]] = {}

    @staticmethod
    def _normalize_lambdas(order: int, lambdas: Tuple[float, ...]) -> Tuple[float, ...]:
        """Chuẩn hóa trọng số lambdas sao cho tổng = 1.0."""
        if not lambdas:
            if order == 1:
                return (0.85, 0.15)
            return (0.7, 0.2, 0.1)

        if order == 1:
            if len(lambdas) == 2:
                s = sum(lambdas)
                if s <= 0:
                    return (0.85, 0.15)
                return (lambdas[0] / s, lambdas[1] / s)
            elif len(lambdas) >= 3:
                # Nếu người dùng truyền 3 trọng số cho bậc 1, gộp trọng số bậc cao
                w1 = lambdas[0] + lambdas[1]
                w0 = lambdas[2]
                s = w1 + w0
                if s <= 0:
                    return (0.85, 0.15)
                return (w1 / s, w0 / s)
            else:
                return (0.85, 0.15)
        else:  # order == 2
            if len(lambdas) == 3:
                s = sum(lambdas)
                if s <= 0:
                    return (0.7, 0.2, 0.1)
                return (lambdas[0] / s, lambdas[1] / s, lambdas[2] / s)
            elif len(lambdas) == 2:
                # Mở rộng thành 3 trọng số hợp lý
                return (lambdas[0] * 0.8, lambdas[0] * 0.2, lambdas[1])
            else:
                return (0.7, 0.2, 0.1)

    def _extract_draw_numbers(self, draw: Union[List[int], Dict[str, Any]]) -> List[int]:
        """Trích xuất và lọc danh sách số hợp lệ trong kỳ quay."""
        if isinstance(draw, dict):
            raw_nums = draw.get("result") or draw.get("numbers") or []
        elif isinstance(draw, (list, tuple)):
            raw_nums = draw
        else:
            return []

        valid = []
        for n in raw_nums:
            try:
                val = int(n)
                if 1 <= val <= self.total_numbers:
                    valid.append(val)
            except (ValueError, TypeError):
                continue
        # Loại bỏ trùng lặp trong cùng 1 kỳ quay nhưng giữ thứ tự ổn định
        return sorted(set(valid))

    def fit(self, draws: List[Union[List[int], Dict[str, Any]]]) -> "HigherOrderMarkovChain":
        """Huấn luyện ma trận chuyển tiếp Markov trên danh sách các kỳ quay lịch sử.

        Args:
            draws: Danh sách các kỳ quay theo thứ tự thời gian tăng dần (cũ nhất -> mới nhất).
                   Mỗi kỳ quay là danh sách số int hoặc dict chứa trường 'result'.

        Returns:
            self: Trả về chính đối tượng HigherOrderMarkovChain sau khi khớp dữ liệu.
        """
        cleaned_draws: List[List[int]] = []
        for d in draws:
            nums = self._extract_draw_numbers(d)
            if nums:
                cleaned_draws.append(nums)

        min_required = self.order + 1
        if len(cleaned_draws) < min_required:
            raise ValueError(
                f"Cần tối thiểu {min_required} kỳ quay hợp lệ để huấn luyện mô hình Markov bậc {self.order}. "
                f"Hiện chỉ có {len(cleaned_draws)} kỳ hợp lệ."
            )

        # Xóa dữ liệu huấn luyện cũ
        self.transitions_order1.clear()
        self.state_totals_order1.clear()
        self.transitions_order2.clear()
        self.state_totals_order2.clear()
        self.unigram_counts.clear()
        self._distribution_cache.clear()

        # Đếm tần suất đơn lẻ
        for d in cleaned_draws:
            for num in d:
                self.unigram_counts[num] += 1

        # Xây dựng ma trận chuyển tiếp bậc 1 (u in draw[t] -> v in draw[t+1])
        t_len = len(cleaned_draws)
        for t in range(t_len - 1):
            curr_draw = cleaned_draws[t]
            next_draw = cleaned_draws[t + 1]
            for u in curr_draw:
                for v in next_draw:
                    self.transitions_order1[u][v] += 1
                    self.state_totals_order1[u] += 1

        # Xây dựng ma trận chuyển tiếp bậc 2 ((u1, u2) in draw[t-1] x draw[t] -> w in draw[t+1])
        if self.order >= 2:
            for t in range(1, t_len - 1):
                prev_draw = cleaned_draws[t - 1]
                curr_draw = cleaned_draws[t]
                next_draw = cleaned_draws[t + 1]
                for u in prev_draw:
                    for v in curr_draw:
                        pair = (u, v)
                        for w in next_draw:
                            self.transitions_order2[pair][w] += 1
                            self.state_totals_order2[pair] += 1

        self.total_draws_fitted = t_len
        self.is_fitted = True
        return self

    def _get_order1_distribution(self, u: int) -> Dict[int, float]:
        """Tính vector phân phối xác suất bậc 1 từ số u sang tất cả các số 1..total_numbers."""
        cache_key = ("o1", u)
        if cache_key in self._distribution_cache:
            return self._distribution_cache[cache_key]

        total_u = self.state_totals_order1.get(u, 0)
        n = self.total_numbers
        uniform_p = 1.0 / n

        probs: Dict[int, float] = {}

        if total_u > 0:
            # Có dữ liệu quan sát: Làm mịn Lidstone kết hợp nội suy Jelinek-Mercer
            lambda1, lambda0 = self.lambdas[0], self.lambdas[1]
            denom = total_u + n * self.alpha
            for w in range(1, n + 1):
                cnt = self.transitions_order1[u].get(w, 0)
                p_mle = (cnt + self.alpha) / denom
                probs[w] = lambda1 * p_mle + lambda0 * uniform_p
        else:
            # Chưa từng thấy số u: Fallback về phân phối đều
            for w in range(1, n + 1):
                probs[w] = uniform_p

        # Chuẩn hóa chính xác về tổng 1.0
        sum_p = sum(probs.values())
        if sum_p > 0:
            for w in probs:
                probs[w] /= sum_p

        self._distribution_cache[cache_key] = probs
        return probs

    def get_transition_distribution(self, prev_numbers: Union[Tuple[int, ...], List[int], int]) -> Dict[int, float]:
        """Lấy toàn bộ phân phối xác suất chuyển tiếp có điều kiện sang các số 1..total_numbers.

        Đảm bảo bất biến chuẩn hóa: sum(distribution.values()) == 1.0.

        Args:
            prev_numbers: Trạng thái tiền điều kiện.
                          - Bậc 1: tuple (u,) hoặc int u.
                          - Bậc 2: tuple (u1, u2).

        Returns:
            Dict[int, float]: Dict ánh xạ từ số w (1..total_numbers) đến xác suất chuyển tiếp.
        """
        if not self.is_fitted:
            raise RuntimeError("Mô hình chưa được huấn luyện. Hãy gọi fit() trước khi truy vấn.")

        # Chuẩn hóa prev_numbers về tuple
        if isinstance(prev_numbers, int):
            prev_tuple = (prev_numbers,)
        else:
            prev_tuple = tuple(prev_numbers)

        cache_key = (self.order, prev_tuple)
        if cache_key in self._distribution_cache:
            return self._distribution_cache[cache_key]

        n = self.total_numbers
        uniform_p = 1.0 / n

        if self.order == 1 or len(prev_tuple) == 1:
            u = prev_tuple[-1]
            dist = self._get_order1_distribution(u)
            self._distribution_cache[cache_key] = dist
            return dist

        # Bậc 2: prev_tuple = (u1, u2)
        u1, u2 = prev_tuple[-2], prev_tuple[-1]
        pair = (u1, u2)
        dist_o1 = self._get_order1_distribution(u2)

        total_pair = self.state_totals_order2.get(pair, 0)
        probs: Dict[int, float] = {}

        lambda2, lambda1, lambda0 = self.lambdas[0], self.lambdas[1], self.lambdas[2]

        if total_pair > 0:
            # Bigram đã quan sát: Nội suy 3 thành phần (bậc 2 Lidstone + bậc 1 + phân phối đều)
            denom = total_pair + n * self.alpha
            for w in range(1, n + 1):
                cnt = self.transitions_order2[pair].get(w, 0)
                p2_mle = (cnt + self.alpha) / denom
                probs[w] = lambda2 * p2_mle + lambda1 * dist_o1[w] + lambda0 * uniform_p
        else:
            # Unseen Bigram State: Backoff về bậc 1 và phân phối đều
            sum_lower = lambda1 + lambda0
            if sum_lower > 0:
                w1 = lambda1 / sum_lower
                w0 = lambda0 / sum_lower
                for w in range(1, n + 1):
                    probs[w] = w1 * dist_o1[w] + w0 * uniform_p
            else:
                for w in range(1, n + 1):
                    probs[w] = dist_o1[w]

        # Chuẩn hóa hàng chính xác về 1.0
        sum_p = sum(probs.values())
        if sum_p > 0:
            for w in probs:
                probs[w] /= sum_p

        self._distribution_cache[cache_key] = probs
        return probs

    def get_transition_probability(
        self, prev_numbers: Union[Tuple[int, ...], List[int], int], next_num: int
    ) -> float:
        """Lấy xác suất chuyển tiếp từ trạng thái prev_numbers sang next_num.

        Args:
            prev_numbers: Trạng thái điều kiện (tuple (u,) hoặc (u1, u2)).
            next_num: Số đích tiếp theo (1..total_numbers).

        Returns:
            float: Xác suất chuyển tiếp có điều kiện P(next_num | prev_numbers).
        """
        if not (1 <= next_num <= self.total_numbers):
            return 0.0

        dist = self.get_transition_distribution(prev_numbers)
        return dist.get(next_num, 0.0)

    def predict_next_probabilities(
        self, recent_draws: List[Union[List[int], Dict[str, Any]]]
    ) -> Dict[int, float]:
        """Dự đoán phân phối xác suất cho kỳ quay kế tiếp dựa trên các kỳ quay gần nhất.

        Args:
            recent_draws: Danh sách các kỳ quay gần đây nhất theo thứ tự thời gian.
                          - Nếu order=1: Chỉ cần 1 kỳ gần nhất (recent_draws[-1]).
                          - Nếu order=2: Dùng 2 kỳ gần nhất (recent_draws[-2], recent_draws[-1]).

        Returns:
            Dict[int, float]: Phân phối xác suất chuẩn hóa trên toàn bộ dải số [1..total_numbers],
                              đảm bảo sum(probs.values()) == 1.0.
        """
        if not self.is_fitted:
            raise RuntimeError("Mô hình chưa được huấn luyện. Hãy gọi fit() trước khi dự đoán.")

        cleaned_recent = [self._extract_draw_numbers(d) for d in recent_draws]
        cleaned_recent = [d for d in cleaned_recent if d]

        n = self.total_numbers
        uniform_p = 1.0 / n

        if not cleaned_recent:
            # Không có dữ liệu kỳ gần nhất -> phân phối đều
            return {num: uniform_p for num in range(1, n + 1)}

        accumulated: Dict[int, float] = {num: 0.0 for num in range(1, n + 1)}

        if self.order == 1 or len(cleaned_recent) < 2:
            # Dự đoán bậc 1: Trung bình cộng phân phối từ tất cả các bóng trong kỳ gần nhất
            last_draw = cleaned_recent[-1]
            if not last_draw:
                return {num: uniform_p for num in range(1, n + 1)}

            for ball in last_draw:
                dist = self.get_transition_distribution((ball,))
                for w, p in dist.items():
                    accumulated[w] += p

            div = len(last_draw)
            for w in accumulated:
                accumulated[w] /= div

        else:
            # Dự đoán bậc 2: Trung bình cộng phân phối từ tất cả các cặp (u in draw[-2], v in draw[-1])
            prev_draw = cleaned_recent[-2]
            last_draw = cleaned_recent[-1]
            if not prev_draw or not last_draw:
                # Fallback bậc 1
                fallback_draw = last_draw or prev_draw
                for ball in fallback_draw:
                    dist = self.get_transition_distribution((ball,))
                    for w, p in dist.items():
                        accumulated[w] += p
                div = len(fallback_draw)
                for w in accumulated:
                    accumulated[w] /= div
            else:
                pair_count = 0
                for u in prev_draw:
                    for v in last_draw:
                        pair_count += 1
                        dist = self.get_transition_distribution((u, v))
                        for w, p in dist.items():
                            accumulated[w] += p

                for w in accumulated:
                    accumulated[w] /= pair_count

        # Chuẩn hóa triệt để để loại bỏ sai số dấu chấm động
        total_sum = sum(accumulated.values())
        if total_sum > 0:
            for w in accumulated:
                accumulated[w] /= total_sum

        return accumulated

    def predict_top_k(
        self, recent_draws: List[Union[List[int], Dict[str, Any]]], k: int = 6
    ) -> List[int]:
        """Lấy danh sách top k số có xác suất chuyển tiếp cao nhất cho kỳ tiếp theo.

        Args:
            recent_draws: Các kỳ quay gần đây nhất.
            k: Số lượng bóng cần chọn (vd: 6 cho Power 6/55 hoặc Mega 6/45).

        Returns:
            List[int]: Danh sách k số có xác suất cao nhất, sắp xếp tăng dần.
        """
        probs = self.predict_next_probabilities(recent_draws)
        # Sắp xếp theo xác suất giảm dần, nếu bằng nhau sắp xếp theo số tăng dần
        sorted_pairs = sorted(probs.items(), key=lambda item: (-item[1], item[0]))
        top_k = [num for num, _ in sorted_pairs[:k]]
        return sorted(top_k)

    def verify_normalization_invariant(self, tolerance: float = 1e-6) -> bool:
        """Kiểm tra và xác nhận bất biến chuẩn hóa xác suất trên toàn bộ không gian trạng thái.

        Quy tắc: Mọi hàng phân phối xác suất có điều kiện PHẢI có tổng bằng 1.0 (sai số < tolerance).

        Args:
            tolerance: Ngưỡng dung sai số học dấu chấm động (mặc định 1e-6).

        Returns:
            bool: True nếu toàn bộ trạng thái đều thỏa mãn chuẩn hóa.

        Raises:
            AssertionError: Nếu có bất kỳ trạng thái nào vi phạm bất biến chuẩn hóa.
        """
        if not self.is_fitted:
            raise RuntimeError("Mô hình chưa được huấn luyện. Không thể kiểm tra bất biến.")

        # 1. Kiểm tra toàn bộ trạng thái bậc 1 (u in 1..total_numbers)
        for u in range(1, self.total_numbers + 1):
            dist = self.get_transition_distribution((u,))
            total_p = sum(dist.values())
            if not math.isclose(total_p, 1.0, abs_tol=tolerance):
                raise AssertionError(
                    f"[Bất biến bị vi phạm] Trạng thái bậc 1 u={u}: tổng xác suất = {total_p:.8f} != 1.0"
                )

        # 2. Kiểm tra toàn bộ trạng thái bậc 2 đã quan sát
        if self.order >= 2:
            for pair in list(self.transitions_order2.keys()):
                dist = self.get_transition_distribution(pair)
                total_p = sum(dist.values())
                if not math.isclose(total_p, 1.0, abs_tol=tolerance):
                    raise AssertionError(
                        f"[Bất biến bị vi phạm] Trạng thái bậc 2 đã quan sát {pair}: "
                        f"tổng xác suất = {total_p:.8f} != 1.0"
                    )

            # 3. Kiểm tra các trạng thái bậc 2 chưa từng quan sát (unseen states / boundary cases)
            test_unseen = [
                (1, 1),
                (self.total_numbers, self.total_numbers),
                (1, self.total_numbers),
                (self.total_numbers, 1),
            ]
            for u in (2, 7, 13, 29):
                for v in (3, 8, 14, 30):
                    if u <= self.total_numbers and v <= self.total_numbers:
                        test_unseen.append((u, v))

            for pair in test_unseen:
                dist = self.get_transition_distribution(pair)
                total_p = sum(dist.values())
                if not math.isclose(total_p, 1.0, abs_tol=tolerance):
                    raise AssertionError(
                        f"[Bất biến bị vi phạm] Trạng thái bậc 2 chưa quan sát {pair}: "
                        f"tổng xác suất = {total_p:.8f} != 1.0"
                    )

        return True

    def get_transition_matrix(self, order: int = 1) -> Dict[str, Dict[int, float]]:
        """Xuất ma trận chuyển tiếp dưới dạng dict phục vụ API hoặc serialize.

        Args:
            order: Bậc muốn xuất (1 hoặc 2).

        Returns:
            Dict[str, Dict[int, float]]: Ma trận chuyển tiếp.
        """
        matrix: Dict[str, Dict[int, float]] = {}
        if order == 1:
            for u in range(1, self.total_numbers + 1):
                matrix[str(u)] = self.get_transition_distribution((u,))
        elif order == 2:
            # Xuất các cặp đã quan sát để tránh bùng nổ kích thước N x N
            for pair in self.transitions_order2.keys():
                key = f"{pair[0]}_{pair[1]}"
                matrix[key] = self.get_transition_distribution(pair)
        return matrix

    def to_dict(self) -> Dict[str, Any]:
        """Serialize mô hình sang dict."""
        return {
            "order": self.order,
            "total_numbers": self.total_numbers,
            "alpha": self.alpha,
            "lambdas": list(self.lambdas),
            "total_draws_fitted": self.total_draws_fitted,
            "is_fitted": self.is_fitted,
            "matrix_order1": self.get_transition_matrix(order=1),
        }

    def save_json(self, file_path: Union[str, Path]) -> None:
        """Lưu ma trận Markov ra file JSON."""
        p = Path(file_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)


def main():
    """Hàm chạy thử nhanh khi thực thi trực tiếp script."""
    data_path = Path(__file__).resolve().parent.parent / "data" / "power655.jsonl"
    if not data_path.exists():
        print(f"File dữ liệu không tồn tại: {data_path}")
        return

    draws = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                draws.append(json.loads(line))

    # Dữ liệu power655 lưu mới nhất trước -> đảo ngược sang cũ nhất trước
    chronological_draws = draws[::-1]
    print(f"Đọc {len(chronological_draws)} kỳ quay từ {data_path.name}")

    mc = HigherOrderMarkovChain(order=2, total_numbers=55, alpha=0.1)
    mc.fit(chronological_draws)
    print("Huấn luyện Markov bậc 2 hoàn tất.")

    # Xác minh bất biến
    assert mc.verify_normalization_invariant()
    print("Bất biến chuẩn hóa (sum == 1.0) thỏa mãn 100%!")

    # Dự đoán cho kỳ tiếp theo
    recent = chronological_draws[-2:]
    probs = mc.predict_next_probabilities(recent)
    top_picks = mc.predict_top_k(recent, k=6)
    print(f"Top 6 bóng dự đoán Markov bậc 2 cho kỳ tiếp theo: {top_picks}")
    print(f"Tổng xác suất kiểm tra: {sum(probs.values()):.8f}")


if __name__ == "__main__":
    main()
