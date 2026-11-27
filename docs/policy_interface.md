# Policy interface (đầu vào, đầu ra)

Nguồn: `smart_buildings/evaluate.py` (initialize_policy, get_policy_action, validate_action, run_policy),
`smart_buildings/smart_control/environment/hybrid_action_environment.py` và docstring của `transfer_policy.py`.

## 1. Vòng đời
1. Evaluator nạp môi trường, tạo `SimulationTracker`, rồi gán `env.metrics_path = None` và
   `env._occupancy_normalization_constant = 125.0`.
2. Policy được tạo **một lần cho mỗi scenario**: thử `Policy(building_meta_data, action_spec)`, nếu lỗi `TypeError`
   thì `Policy(building_meta_data)`, rồi `Policy()`.
3. `tracker.reset()` rồi mỗi bước: `action = policy.get_action(obs)` (hoặc policy là hàm gọi được),
   `validate_action`, `tracker.step(action)`. Obs đầu tiên là obs của `reset()`.
4. **Không có hàm reset cho policy.** Trạng thái nội bộ (nếu có) phải khởi tạo trong `__init__`; một
   instance mới mỗi scenario. Không để trạng thái rò giữa các episode.
5. Không đổi tên lớp mà không cập nhật `POLICY_MAP` trong evaluate.py.

## 2. Đầu vào
- `obs`: mảng numpy float32 một chiều, độ dài `len(building_meta_data["observation_field_names"])`
  (352 ở building_train_1, 703 ở building_train_2, 283 ở building_train_8 theo số đo của `building_summary.md`; docstring của transfer_policy.py ghi 271 là lỗi thời).
- Một số trường chuẩn hóa, số khác đơn vị gốc: xem `docs/data_dictionary.md` và `src/units.py`.
- `building_meta_data` (dict) gồm các khóa: n_discrete_action, n_continuous_action, temp_indices, pressure_indices,
  water_indices, recirculation_indices, zone_action_indices, zone_temp_indices, observation_field_names, floor_plan,
  zone_map, ahu_zone_assignments, num_floors, morning_start_hour, evening_start_hour, comfort_temp_window,
  eco_temp_window, time_zone, holidays, observation_normalization_metadata, action_normalization_metadata.
- **Không có** reward, thông tin bước trước ngoài những gì obs phản ánh, tên tòa nhà, thời tiết hay tháng.
  Dùng tên hoặc seed của cấu hình ẩn là vi phạm luật (docstring). Giờ và thứ trong tuần suy từ hod_*/dow_*.
- Bước t: obs chứa damper/reheat/setpoint của action[t-1] (đã qua sàn của simulator).

## 3. Đầu ra
```python
{"continuous_action": np.ndarray float32 shape (3K + 2 + 2V,),   # trong [-1, 1]
 "discrete_action":   np.ndarray int32   shape (K + 1,)}         # đúng 0 hoặc 1
```
- `validate_action` chỉ ép kiểu dtype và kiểm tra shape. Nó **không chặn miền [-1, 1]**.
  Tầng dưới (`BoundedActionNormalizer.setpoint_value`, ACTION_TOLERANCE = 1e-5) **ném `ValueError`** nếu giá trị
  ra ngoài [-1, 1] quá 1e-5: môi trường không tự cắt mà báo lỗi và dừng episode. Policy **bắt buộc** tự
  `np.clip(..., -1, 1)` trước khi trả action **[OK, đọc từ code]**.
- Giá trị rời rạc khác 0 và 1 làm môi trường ném `NotImplementedError`.
- Action 0 liên tục là trung điểm, không phải tắt. Muốn tắt AHU thì đặt `discrete_action = 0`.
- Sàn của simulator: damper tối thiểu 10%, AHU đang chạy có static pressure tối thiểu, recirculation tối đa 0.70.

## 4. Quy tắc thiết kế
- Không hardcode độ dài hay chỉ số; dùng `BuildingIndex` và `ActionLayout`.
- Nếu model lỗi, feature thiếu hoặc NaN: trả về action của controller nền (pid_water).
- Mọi action trả ra phải qua `check_action` (src/policy_contract.py) trong test.
- Thời gian mô phỏng khoảng 0.1 đến 0.2 giây mỗi bước; hạn mức Kaggle xem mục 6.

## 6. Ràng buộc nộp bài trên Kaggle (Code Requirements) **[OK]**
- Nộp qua Notebook (code competition). Nút Submit chỉ bật khi notebook sau khi commit thỏa:
  CPU <= 9 giờ, GPU <= 9 giờ, **tắt internet**; dữ liệu ngoài công khai và model pre-trained được phép.
- Hệ quả: không `pip install` trong notebook; trọng số model và mã phụ phải đóng gói thành Kaggle Dataset
  rồi gắn vào notebook. Không huấn luyện trong notebook nộp bài, chỉ nạp model đã train.
- Ngân sách thời gian (ƯỚC LƯỢNG, chưa biết số scenario ẩn): mô phỏng đo được khoảng 0.09 đến 0.17 s/bước
  trên máy bạn, tòa lớn chậm hơn. Một episode 8640 bước mất khoảng 13 đến 25 phút, cộng khoảng 1 phút nạp.
  Nếu chấm khoảng 30 scenario tuần tự thì riêng mô phỏng đã chiếm khoảng 7 đến 12 giờ so với trần 9 giờ.
  Vì vậy thời gian suy luận của policy phải rất nhỏ (mục tiêu < 5 ms mỗi bước, không chạy mạng lớn
  hay tối ưu hóa mỗi bước) và nên đo bằng `seconds_per_step` trong summary.json.
  CPU của Kaggle có thể chậm hơn máy bạn; cần thử thật trên một notebook.

## 5. Khung policy tối thiểu
```python
import numpy as np
from src.features import BuildingIndex
from src.units import ActionLayout, ObsUnits


class MyPolicy:
    def __init__(self, building_meta_data, action_spec):
        self.meta = building_meta_data
        self.index = BuildingIndex(building_meta_data)
        self.layout = ActionLayout(building_meta_data)
        self.units = ObsUnits(building_meta_data)

    def get_action(self, obs):
        native = self.units.to_native(obs)                 # float64 copy, obs not modified
        cont = self.layout.neutral()                       # float32, all zeros
        disc = np.ones(self.layout.n_disc, dtype=np.int32)
        # ... compute cont and disc here ...
        cont = np.clip(cont, -1.0, 1.0).astype(np.float32)
        return {"continuous_action": cont, "discrete_action": disc}
```