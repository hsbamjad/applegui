import sys
import threading, queue, time

from pathlib import Path
import sys
from datetime import datetime

LOCAL_ULTRALYTICS_ROOT = Path(
    r"C:\Yuyuan\apple_sorting_project\apple_integration_code\ultralytics_perseanal_learning"
)

sys.path.insert(0, str(LOCAL_ULTRALYTICS_ROOT))

import ultralytics
print("Using ultralytics from:", ultralytics.__file__)

from ultralytics import YOLO

from initial_test_packet import SourceStream, packet_queue

from online_test_packet import detection_loop, image_saver_loop, run_lanes
import eBUS as eb
import os


sys.path.append(r"C:\Yuyuan\apple_sorting_project\apple_integration_code\apple_integration\lib")
import PvSampleUtils as psu

# 只选择一个检测通道

DETECTION_CHANNEL = "Source0"

# 可选: "rgb", "nir1", "nir2"
DISPLAY_MODE = "rgb"
def main():
    connection_id = psu.PvSelectDevice()
    result, device = eb.PvDevice.CreateAndConnect(connection_id)
    params = device.GetParameters()
    params.Get("AcquisitionMode").SetValue("Continuous")
    params.Get("TriggerMode").SetValue("Off")
    params.Get("AcquisitionFrameRate").SetValue(15)

    for name in ["Source0", "Source1", "Source2"]:
        params.Get("SourceSelector").SetValue(name)
        params.Get("ExposureTime").SetValue(3000)

    TARGET_SOURCES = ["Source0", "Source1", "Source2"]

    sources = []
    selector = device.GetParameters().GetEnum("SourceSelector")
    result, count = selector.GetEntriesCount()

    print(f"🔍 Found {count} sources on device.")

    for i in range(count):
        result, entry = selector.GetEntryByIndex(i)
        if result.IsOK() and entry:
            result, name = entry.GetName()
            if result.IsOK():
                if name in TARGET_SOURCES:
                    print(f"✅ Opening stream for {name}...")
                    stream = SourceStream(device, connection_id, name, detection_channel=["Source0", "Source1", "Source2"], display_mode=DISPLAY_MODE)
                    if stream.open():
                        sources.append(stream)
                        print(f"   ✔️ Stream {name} opened successfully.")
                    else:
                        print(f"   ❌ Failed to open stream for {name}.")
                else:
                    print(f"⏭️ Skipping {name} (not in TARGET_SOURCES)")

    # 启动选定的通道
    for s in sources:
        print(f"🚀 Starting acquisition for {s.source_name} ...")
        s.start_acquisition()
        s.start_thread(show_window=False)  # 不显示窗口可改为 False
    # YOLO 模型
    # model_grade = YOLO(
    #     'C:/Users/Tommy/OneDrive - Michigan State University/data/system_integration/training_data/2025_9_30/model/rg-nir1/grading/weights/best.engine',
    #     task='segment'
    # )
    # model_grade = YOLO(r"C:\Yuyuan\apple_sorting_project\apple_integration_code\model\sge model\epoch183.pt")
    model_path = r"C:\Yuyuan\apple_sorting_project\apple_integration_code\model\sge model\epoch183.pt"
    model_grade = YOLO(model_path)
    file_path = r"C:\Yuyuan\apple_sorting_project\apple_integration_code\apple_integration_experiment\Batch 16\Repeat 3\output.json"
    experiment_id = datetime.now().strftime("experiment_%Y%m%d_%H%M%S")

    output_root = r"C:\Yuyuan\apple_sorting_project\apple_integration_code\apple_integration_experiment\Batch 16\Repeat 3"

    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    class_names = ["Class_1", "Class_2", "Class_3"]
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255)]

    lane_queues = [queue.Queue(), queue.Queue(), queue.Queue()]
    save_queue = queue.Queue(maxsize=20)
    stop_event = threading.Event()
    save_stop_event = threading.Event()
    fps_stats = {
        "saving_fps": 0.0,
        "lock": threading.Lock()
    }

    save_root = os.path.join(output_root, experiment_id, "images")
    save_interval = 1

    t1 = threading.Thread(
    target=detection_loop,
    args=(
        packet_queue,
        lane_queues,
        stop_event,
        model_grade,
        file_path,
        class_names,
        colors
    ),
    kwargs={
        "output_root": output_root,
        "experiment_id": experiment_id,
        "model_path": model_path,
        "speed": "unknown",
        "cultivar": "unknown",
        "true_label_map": None,
        "save_queue": save_queue,
        "fps_stats": fps_stats
    }
)
    t2 = threading.Thread(target=run_lanes, args=(lane_queues,))
    t3 = threading.Thread(
        target=image_saver_loop,
        args=(save_queue, save_stop_event, save_root, save_interval, fps_stats)
    )
    t1.start()
    t2.start()
    t3.start()

    try:
        t1.join()
    except KeyboardInterrupt:
        print("⛔ Ctrl+C，通知线程退出")
        stop_event.set()

    # 检测线程停止生产保存任务后，再通知保存线程排空队列并退出。
    t1.join()
    save_stop_event.set()

    t2.join()
    t3.join()

    # 停相机（最后）
    for s in sources:
        s.stop_thread()
        s.stop_acquisition()
        s.close()


if __name__ == "__main__":
    main()
