# Reward function analysis

Nguồn: `smart_buildings/smart_control/reward/setpoint_energy_carbon_regret.py` và
`base_setpoint_energy_carbon_reward.py`; tham số trong `config.gin` của scenario.

## 1. Công thức mỗi bước
```
reward_t = (0.6 * productivity_regret  -  0.2 * energy_cost_norm  -  0.2 * carbon_norm) / (0.6 + 0.2 + 0.2)
```
- Trọng số trong config: productivity 0.6, energy 0.2, carbon 0.2 (tổng 1.0).
- Kiểm chứng số: reward = công thức trên, sai lệch tối đa 2e-10 trên 288 bước
  (`reward_components.csv`). Đoạn đó là Chủ nhật không có người nên số hạng comfort bằng 0
  và chỉ phần năng lượng và carbon được kiểm tra bằng số; số hạng comfort dưới đây đọc từ code.

## 2. Số hạng comfort (productivity regret)
Với mỗi zone z, độ lệch khỏi band:
```
dev_z = max(0, T_z - cooling_setpoint_z, heating_setpoint_z - T_z)        [K]
shortfall(dev) = dev / 5                         nếu dev <= 5
               = 1 + ((dev - 5) / 5)^2           nếu dev > 5   (KHÔNG bị chặn trên)
```
| dev (K) | 1 | 2.5 | 5 | 7.5 | 10 | 15 | 20 |
|---|---|---|---|---|---|---|---|
| shortfall | 0.2 | 0.5 | 1.0 | 1.25 | 2.0 | 5.0 | 10.0 |

Gộp các zone (min_productivity = 0):
```
productivity_regret = - sum_z( occupancy_z * shortfall_z ) / sum_z( occupancy_z )      (<= 0)
productivity_regret = 0                                              nếu tổng số người = 0
```
Hệ quả:
1. **Chỉ phạt comfort khi có người.** Khi tòa không có người, số hạng comfort bằng 0 dù nhiệt độ ra sao.
2. Là trung bình có trọng số theo số người của từng zone. Zone đông người quan trọng hơn.
3. Giá trị không bị chặn: một zone lạnh hơn band 15 K đã tính shortfall 5. Một zone bị bỏ mặc có thể
   chi phối cả episode (đúng như tài liệu của ban tổ chức).
4. Band dùng là heating/cooling setpoint của zone theo lịch: comfort (294, 297) K trong giờ làm việc,
   eco (289, 298) K ngoài giờ (building_train_1).

## 3. Số hạng năng lượng và carbon
- energy_cost_norm = (chi phí điện + chi phí gas) / (chi phí tại công suất tối đa danh nghĩa).
  Công suất điện tối đa danh nghĩa 320 kW và gas 820 kW (building_train_1; ghi chú trong config: 80 kW và 205 kW mỗi AHU x 4 AHU).
- carbon_norm tính tương tự từ lượng carbon thải ra.
- Công suất thực KHÔNG bị chặn bởi mức tối đa, nên các số hạng này có thể lớn hơn 1.
- Điện gồm quạt (blower), điều hòa AHU (trị tuyệt đối), bơm nước; gas là phần đốt của boiler (sưởi).

## 4. Điểm chấm
`score = trung bình trên các scenario của |total_reward / 8640|`, thấp hơn thì tốt hơn
(`score_leaderboard.py`). Baseline không tham gia công thức, chỉ in để tham chiếu.
Tổng reward ở 8640 bước chỉ so sánh được với các lần chạy cùng số bước.

## 5. Quan sát thực tế (building_train_1 / fairbanks_1 / pid_water / 8640 bước)
- total_reward = -2607.41 (trùng `evaluate.py`). `baseline_returns.json` ghi -1828.90: máy này không
  tái lập baseline chính xác (atlanta_2: -212.35 so với -208.21).
- Ngày 1 (Chủ nhật): reward bằng 0. Từ thứ Hai, mỗi ngày mất khoảng -100 đến -200; cuối tháng khoảng -10 đến -70.
- Có 3277 bước ở chế độ comfort (38%). Trong đó 88% số cặp zone-bước thấp hơn cận dưới của band,
  thiếu trung bình 2.5 K và tối đa 12.2 K; 37% số bước có ít nhất một zone thiếu hơn 5 K (vùng bậc hai).
- Bước tệ nhất -1.56; 5% bước tệ nhất đóng góp 17.5% tổng.
- Ngoài trời từ -42 đến 1 độ C. Ở scenario này pid_water kém pid và fixed_setpoint (theo baseline_returns.json).

## 6. Hàm ý cho thiết kế policy
- Ưu tiên comfort khi có người; ngoài giờ chỉ cần tiết kiệm năng lượng (AHU tắt khi không ai).
- Tránh để một zone lệch lớn, vì phạt tăng bậc hai.
- Dự báo nhiệt độ zone kế tiếp (delta_T) để nhận ra sớm zone sắp vượt band là hợp lý với cấu trúc reward này.
- Chưa kiểm chứng: dạng chính xác của occupancy theo zone (không có trong observation, chỉ có tổng num_occupants).
