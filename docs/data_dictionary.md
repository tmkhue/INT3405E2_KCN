# Data dictionary: observation, action, normalization

Phạm vi xác minh: mục đánh dấu **[OK]** được kiểm chứng bằng số liệu trên episode
`building_train_1 / fairbanks_1 / pid_water / 8640 bước`. Đối chiếu thêm trên các episode 600 bước
(`pid_water`, bước đầu là Chủ nhật, chưa có người): building_train_2 / atlanta_3, building_train_4 / atlanta_5,
building_train_5 / atlanta_6, building_train_7 / atlanta_8, building_train_8 / atlanta_1.

Kết quả test (`tests/test_units.py`, 18/18 pass trên 6 tòa): bố cục action, đổi đơn vị hai chiều,
quan hệ action[t] -> obs[t+1], sàn damper 10%, sàn static pressure 200 Pa, đồng hồ 5 phút mỗi bước.
`tests/test_mapping.py` (6/6 pass trên 6 tòa): kích thước action = 3K+2+2V, chỉ số action phủ đủ và không trùng, zone_temp_indices khớp tên trường, mỗi zone thuộc đúng một AHU.
Giới hạn: bốn tòa mới chỉ kiểm trên 600 bước, nên chưa thấy hành vi lúc có người (comfort mode, num_occupants). Mỗi tòa chỉ kiểm một weather.

Commit starter-kit: `1fb75606d12e016463fbc63de704967c5b2bf9d5`

## 1. Quy ước chung
- K = số AHU, V = số VAV (mỗi VAV một zone). Độ dài vector thay đổi theo tòa nhà. **Không hardcode.**
  | Tòa | K | V | obs | action liên tục | action rời rạc | tầng |
  |---|---|---|---|---|---|---|
  | building_train_1 | 4 | 39 | 352 | 92 | 5 | 3 |
  | building_train_2 | 9 | 85 | 703 | 199 | 10 | 2 |
  | building_train_4 | 6 | 54 | 472 | 128 | 7 | 3 |
  | building_train_5 | 8 | 79 | 652 | 184 | 9 | 4 |
  | building_train_7 | 4 | 36 | 334 | 86 | 5 | 2 |
  | building_train_8 | 3 | 30 | 283 | 71 | 4 | 2 |
  Công thức: liên tục = 3K + 2 + 2V, rời rạc = K + 1 **[OK]** (test_mapping.py).
- `VAV_N` tương ứng `room_N` trong `ahu_zone_assignments` **[OK]**.
- Thứ tự VAV trong **action** theo bảng chữ cái (VAV_1, VAV_10, VAV_11, ..., VAV_19, VAV_2, VAV_20, ...)
  **[OK]**. Luôn dùng `zone_action_indices["VAV_N"]`, không dùng N làm chỉ số.
- Action tại bước t được áp dụng, và kết quả hiện ra ở observation của bước t+1 **[OK]**.
  Vì vậy obs[t] chứa damper/reheat/setpoint của action[t-1], không phải action[t].

## 2. Bố cục observation (ví dụ building_train_1, 352 phần tử)
Chỉ để minh họa, KHÔNG hardcode. Dựng bảng chỉ số từ `observation_field_names` (src/features.py).

| Chỉ số | Nội dung |
|---|---|
| 0 đến 233 | 39 zone x 6 trường, xen kẽ theo từng VAV (thứ tự VAV theo bảng chữ cái): reheat_valve, supply_air_damper, supply_air_flowrate_setpoint, zone cooling setpoint, zone heating setpoint, zone_air_temperature |
| 234 đến 293 | 4 AHU x 15 trường, theo từng AHU: cooling_request_count, discharge_fan_speed, exhaust_fan_run_command, outside_air_flowrate, outside_air_temperature, recirculation, supervisor_run_command, supply_air_flowrate sensor và setpoint, static_pressure sensor và setpoint, supply_air_temperature sensor và setpoint, supply_fan_run_command, supply_fan_speed |
| 294 đến 300 | hot-water system (hws): differential_pressure sensor và setpoint, heating_request_count, run_status, supervisor_run_command, supply_water_temperature sensor và setpoint |
| 301 đến 312 | hod_cos_000 .. 011 |
| 313 đến 324 | hod_sin_000 .. 011 |
| 325 đến 336 | dow_cos_000 .. 011 |
| 337 đến 348 | dow_sin_000 .. 011 |
| 349, 350, 351 | comfort_mode_now, comfort_mode_soon, num_occupants |

## 3. Chuẩn hóa observation **[OK]**
Chỉ một số trường được chuẩn hóa. `observation_normalization_metadata` liệt kê 13 nhóm nhưng
KHÔNG phản ánh trường nào thực sự được áp dụng. Quy tắc thật (src/units.py, NORMALIZED_OBS_RULES):

| Trường | Trong vector | Về đơn vị gốc |
|---|---|---|
| VAV_*_zone_air_temperature_sensor | chuẩn hóa | T[K] = obs x 5 + 295 (khớp timeseries.csv, sai lệch < 1e-4 K) |
| VAV_*_supply_air_damper_percentage_command | [-1, 1] | native = obs x 0.5 + 0.5 |
| VAV_*_reheat_valve_percentage_command | [-1, 1] | native = obs x 0.5 + 0.5 |
| hws_differential_pressure_setpoint | chuẩn hóa | native = obs x 6 + 10 |
| hws_run_status, hws_supervisor_run_command | {-1, 1} | native = obs x 0.5 + 0.5 (bằng 2 x action rời rạc - 1) |

Các trường **còn lại là đơn vị gốc**, không chuẩn hóa:
- Kelvin: zone cooling/heating setpoint, outside_air_temperature, supply_air_temperature (sensor/setpoint), hws supply_water_temperature.
- Pa: static_pressure (sensor/setpoint), hws_differential_pressure_sensor.
- 0 đến 1: fan speed, recirculation. Đếm: *_request_count, num_occupants.
- {0, 1}: ahs_ahu_*_supervisor_run_command và các run_command khác của AHU.
- Lưu lượng (*_flowrate_*): m³/s theo docstring của `air_handler.py`; chưa kiểm bằng số liệu **[CHƯA XÁC MINH BẰNG SỐ LIỆU]**.

## 4. Thời gian và lịch **[OK]**
- hod_cos_000, hod_sin_000: giờ hiện tại, góc = giờ x 15 độ. Khớp đồng hồ mô phỏng chính xác (bước 5 phút).
- hod_*_k (k = 0..11): giờ trong ngày tại thời điểm t + 2k giờ, tức 12 mốc cách 2 giờ trải 24 giờ tới.
- dow_*_000: chỉ số ngày trong tuần (thứ Hai = 0), góc = ngày x 360/7, đổi tại 0h.
  dow_*_k: chỉ số ngày tại thời điểm t + 14k giờ (12 mốc phủ một tuần).
- Scenario fairbanks_1 bắt đầu 06:00 Chủ nhật 2023-01-01. Ngày đầu không có người, không có phạt comfort.
  Các scenario khác chưa kiểm tra giờ bắt đầu **[CHƯA]**.
- comfort_temp_window = (294.0, 297.0) K, eco_temp_window = (289.0, 298.0) K (building_train_1).
  morning_start_hour = 6, evening_start_hour = 19, time_zone = US/Pacific.
- Setpoint của zone trong obs lúc reset (297 và 294) là giá trị khởi tạo. Không dùng bước 0 làm feature.
- comfort_mode_now/soon ở {0, 1}. num_occupants là số người (0 đến 74 trong episode đã thu). Có 19 bước
  comfort_mode = 0 mà vẫn có người và 333 bước comfort_mode = 1 mà không có người, nên hai thứ không đồng nhất.

## 5. Action
### Rời rạc, int32, K + 1 phần tử, giá trị đúng 0 hoặc 1
Phần tử 0..K-1: bật/tắt AHU; phần tử cuối: bật/tắt hot-water (hws_supervisor_run_command).
Môi trường ném lỗi với giá trị khác 0 hoặc 1. Trong episode pid_water, AHU 1 tắt 46% số bước.

### Liên tục, float32, 3K + 2 + 2V phần tử, miền [-1, 1]
Giá trị 0 là TRUNG ĐIỂM của khoảng, không phải tắt. Bố cục xen kẽ, đọc từ metadata:

| Nhóm | Chỉ số lấy từ | Miền gốc |
|---|---|---|
| AHU supply air temperature | meta["temp_indices"] (K phần tử) | 285 đến 305 K |
| AHU static pressure | meta["pressure_indices"][:K] | 0 đến 2000 Pa |
| hws differential pressure | meta["pressure_indices"][K] (phần tử cuối) | 0 đến 20 |
| AHU recirculation | meta["recirculation_indices"] | 0 đến 1 (simulator giới hạn 0.70) |
| hws supply water temperature | meta["water_indices"] (1 phần tử) | 310 đến 350 K |
| VAV damper, reheat | meta["zone_action_indices"]["VAV_N"] | 0 đến 1 |

Chuyển đổi: dùng `ActionLayout.to_native` / `to_normalized` trong src/units.py.

### Sàn do simulator áp đặt (bạn đặt thấp hơn thì chỉ nhận sàn)
- Damper VAV tối thiểu 10% (normalized -0.8): obs[t+1] = max(action[t], 0.1) **[OK]**.
- AHU đang chạy giữ static pressure tối thiểu 200 Pa: obs[t+1] = max(action[t], 200)
  khi AHU bật và 0 khi tắt **[OK trên 6 tòa]**.
- Recirculation tối đa 0.70.
- reheat obs[t+1] = action[t] (không có sàn) **[OK]**. AHU supply air temp setpoint obs[t+1] = action[t] **[OK]**.

## 6. Trường hằng hoặc ít thông tin
- VAV_*_supply_air_flowrate_setpoint = 2.0 ở mọi bước của episode đã thu (39 trường hằng). Bỏ khỏi feature ML.

## 7. Việc còn lại
- Chạy `scripts/inspect_all_buildings.py` và kiểm tra các quy tắc còn chưa xác minh ở mục 3 và mục 5 trên building_train_4, _5, _7, _8.
- Đơn vị lưu lượng không khí, tên chính xác phần tử của `floor_plan` và `zone_map` chưa được xem xét.
