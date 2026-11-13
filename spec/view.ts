export const title = "Lottery Analytics & Prediction Pipeline";

export function state(vars: Record<string, any>) {
  if (vars.phase === "IDLE") return "IDLE (Chờ kích hoạt)";
  if (vars.phase === "INGESTED") return "INGESTED (30 ngày tháng 12)";
  if (vars.phase === "CUTOFF_VERIFIED") return "CUTOFF_VERIFIED (Đã cắt future data < 11/12)";
  if (vars.phase === "FEATURES_READY") return "FEATURES_READY (Markov & Z-Score sẵn sàng)";
  if (vars.phase === "PREDICTED") return "PREDICTED (Đã chọn 6 số tối ưu)";
  if (vars.phase === "EVALUATED") {
    return vars.bankroll > 9 ? "EVALUATED_WIN (Trúng giải)" : "EVALUATED_MISS (Trật vé)";
  }
  return vars.phase;
}

export function step(action: string) {
  if (action === "IngestData") return "Nạp dữ liệu";
  if (action === "ApplyCutoff") return "Cắt dữ liệu >= 11/12";
  if (action === "ExtractFeatures") return "Trích xuất đặc trưng";
  if (action === "GeneratePrediction") return "Dự đoán & Đặt cược";
  if (action === "EvaluateDraw") return "Đối soát kết quả";
  return action;
}
