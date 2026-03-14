import os
import argparse
import subprocess
import multiprocessing

def get_all_sequences(root_dir, dark=False):
    seq_paths = []
    Seqs = ["Seq_0", "Seq_1"]
    
    for warehouse in sorted(os.listdir(root_dir)):
        warehouse_path = os.path.join(root_dir, warehouse)
        if not os.path.isdir(warehouse_path):
            continue
        for seq in Seqs:
            seq_path = os.path.join(warehouse_path, seq)
            if os.path.isdir(seq_path):
                seq_paths.append((warehouse, seq, seq_path))
    return seq_paths

def run_worker(gpu_id, tasks, save_base_dir, dataset, stride, mode, align, flow, light):
    for warehouse, seq, seq_path in tasks:
        save_dir = os.path.join(save_base_dir, warehouse, seq)
        os.makedirs(save_dir, exist_ok=True)
        if align:
            
            cmd = [
                "python", "unaligned_cd_dir.py",
                "--scene-dir", seq_path,
                "--save-dir", save_dir,
                "--dataset", dataset,
                "--stride", str(stride),
                "--mode", mode,
                "--align",
                "--iou_thresh", "0.35"
            ]
        else:
            if flow:
                cmd = [
                    "python", "flow_cd_dir.py",
                    "--scene-dir", seq_path,
                    "--save-dir", save_dir,
                    "--dataset", dataset,
                    "--stride", str(stride),
                    "--mode", "initial",
                    "--iou_thresh", "0.65",
                ]
                print(f"run flow_load_json.py for {seq_path}")
            else:
                cmd = [
                    "python", "unaligned_cd_dir.py",
                    "--scene-dir", seq_path,
                    "--save-dir", save_dir,
                    "--dataset", dataset,
                    "--stride", str(stride),
                    "--mode", mode,
                    "--iou_thresh", "0.35",
                  
                ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        print(f"[GPU {gpu_id}] Launching {warehouse}/{seq}")
        subprocess.run(cmd, env=env)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root-dir', required=True, help='Root dir containing Warehouse_x/Seq_y')
    parser.add_argument('--save-dir', required=True, help='Output base dir')
    parser.add_argument('--dataset', default='changesim', help='Dataset name')
    parser.add_argument('--gpus', default='6, 7', help='Comma-separated list of GPU ids')
    parser.add_argument('--stride', default=5, type=int)
    parser.add_argument('--mode', default='occupy', choices=['initial', 'occupy'], help='Mode of operation')
    parser.add_argument('--align', action='store_true', help='Whether to align images')
    parser.add_argument('--flow', action='store_true', help='Whether use flow match')
    parser.add_argument('--dark', action='store_true', help='whether evaluate dark')
    parser.add_argument('--light', action='store_true', help='whether evaluate dark')
    args = parser.parse_args()

    available_gpus = list(map(int, args.gpus.split(',')))
    seq_paths = get_all_sequences(args.root_dir, args.dark)

    # 按GPU数量分配任务
    gpu_task_lists = [[] for _ in available_gpus]
    for i, task in enumerate(seq_paths):
        gpu_task_lists[i % len(available_gpus)].append(task)

    processes = []
    for gpu_id, task_list in zip(available_gpus, gpu_task_lists):
        p = multiprocessing.Process(target=run_worker, args=(gpu_id, task_list, args.save_dir, 
                                                             args.dataset, args.stride, args.mode, 
                                                             args.align, args.flow, args.light))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

if __name__ == '__main__':
    main()
