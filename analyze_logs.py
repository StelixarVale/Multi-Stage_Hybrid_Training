import json
import os
import glob
from collections import defaultdict
from datetime import datetime


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_ROOTS = {
    "DQN": os.path.join(BASE_DIR, "python_scripts", "DQN", "log"),
    "PPO": os.path.join(BASE_DIR, "python_scripts", "PPO", "log"),
    "SAC": os.path.join(BASE_DIR, "python_scripts", "SAC", "log"),
}

STAGES = ["catch_log", "tai_log"]


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_float_list(data, key):
    vals = data.get(key, [])
    if not isinstance(vals, list):
        vals = [vals]
    result = []
    for v in vals:
        try:
            result.append(float(v))
        except (ValueError, TypeError):
            continue
    return result


def calc_stats(arr):
    if not arr:
        return {"count": 0, "mean": 0, "max": 0, "min": 0, "sum": 0}
    return {
        "count": len(arr),
        "mean": sum(arr) / len(arr),
        "max": max(arr),
        "min": min(arr),
        "sum": sum(arr),
    }


def analyze_algo_stage(algo_name, stage_name, log_files):
    if not log_files:
        return None

    all_returns = []
    all_goals = []
    all_losses = []
    total_episodes = 0
    total_actions = 0
    action_counter = defaultdict(int)
    start_time = None
    end_time = None

    for fpath in sorted(log_files):
        try:
            data = load_json(fpath)
        except Exception:
            print(f"  [警告] 无法解析: {fpath}")
            continue

        returns = safe_float_list(data, "return_all")
        goals = safe_float_list(data, "goal")
        losses = safe_float_list(data, "loss")

        n = max(len(returns), len(goals))
        if n == 0:
            continue

        all_returns.extend(returns)
        all_goals.extend(goals)
        all_losses.extend(losses)
        total_episodes += n

        action_list = data.get("action_list", [])
        if isinstance(action_list, list):
            for seq in action_list:
                if isinstance(seq, list):
                    total_actions += len(seq)
                    for a in seq:
                        try:
                            action_counter[int(a)] += 1
                        except (ValueError, TypeError):
                            continue

        try:
            t = data.get("start time")
            if t:
                t = datetime.strptime(t, "%Y-%m-%d %H:%M:%S")
                if start_time is None or t < start_time:
                    start_time = t
        except Exception:
            pass

        save_times = data.get("save time", [])
        if isinstance(save_times, list) and save_times:
            for st in save_times:
                try:
                    st_dt = datetime.strptime(st, "%Y-%m-%d %H:%M:%S")
                    if end_time is None or st_dt > end_time:
                        end_time = st_dt
                except Exception:
                    continue

    if total_episodes == 0:
        return None

    duration_str = ""
    if start_time and end_time:
        duration = end_time - start_time
        total_seconds = int(duration.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        duration_str = f"{hours}h {minutes}m {seconds}s"

    return {
        "total_episodes": total_episodes,
        "return_stats": calc_stats(all_returns),
        "goal_stats": calc_stats(all_goals),
        "loss_stats": calc_stats(all_losses) if all_losses else None,
        "total_actions": total_actions,
        "avg_actions_per_episode": total_actions / total_episodes if total_episodes > 0 else 0,
        "action_distribution": dict(sorted(action_counter.items())),
        "duration": duration_str,
        "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S") if start_time else "",
        "end_time": end_time.strftime("%Y-%m-%d %H:%M:%S") if end_time else "",
    }


def collect_log_files(algo_name):
    root = LOG_ROOTS.get(algo_name)
    if not root or not os.path.isdir(root):
        return {}
    result = {}
    for stage in STAGES:
        stage_dir = os.path.join(root, stage)
        if os.path.isdir(stage_dir):
            files = glob.glob(os.path.join(stage_dir, "*.json"))
            result[stage] = files
        else:
            result[stage] = []
    return result


def print_separator(char="=", width=72):
    print(char * width)


def print_stats_block(label, stats, indent=2, is_catch=False, tai_episodes=0):
    if stats is None:
        return
    prefix = " " * indent
    print(f"{prefix}{label}:")
    print(f"{prefix}  总 episode 数:  {stats['total_episodes']}")
    r = stats["return_stats"]
    print(f"{prefix}  累计奖励 (return):  均值 {r['mean']:.2f}  |  最大 {r['max']:.2f}  |  最小 {r['min']:.2f}  |  样本数 {r['count']}")
    if is_catch and tai_episodes > 0 and stats['total_episodes'] > 0:
        grasp_rate = tai_episodes / stats['total_episodes'] * 100
        print(f"{prefix}  进入踩踏阶段次数:    {tai_episodes}")
        print(f"{prefix}  抓取成功率:          {grasp_rate:.2f}%  ({tai_episodes}/{stats['total_episodes']})")
    if stats["loss_stats"] and stats["loss_stats"]["count"] > 0:
        l = stats["loss_stats"]
        print(f"{prefix}  Loss:              均值 {l['mean']:.6f}  |  样本数 {l['count']}")
    print(f"{prefix}  总动作数:          {stats['total_actions']}")
    print(f"{prefix}  平均每 episode 动作数: {stats['avg_actions_per_episode']:.1f}")
    if stats["action_distribution"]:
        print(f"{prefix}  动作分布:          {stats['action_distribution']}")
    if stats["duration"]:
        print(f"{prefix}  训练时长:          {stats['duration']}")
        print(f"{prefix}  起止时间:          {stats['start_time']}  ~  {stats['end_time']}")


def generate_comparison(all_results):
    print_separator()
    print("跨算法对比")
    print_separator()

    print(f"\n--- 抓取阶段 (catch_log) 对比 ---")
    print(f"{'算法':<6} {'抓取Episodes':>12} {'Return均值':>10} {'Return最大':>10} {'进入踩踏':>10} {'抓取成功率':>10}")
    print("-" * 64)
    for algo in ["DQN", "PPO", "SAC"]:
        catch_s = all_results.get(algo, {}).get("catch_log")
        tai_s = all_results.get(algo, {}).get("tai_log")
        if catch_s:
            r = catch_s["return_stats"]
            tai_ep = tai_s["total_episodes"] if tai_s else 0
            rate = tai_ep / catch_s["total_episodes"] * 100 if catch_s["total_episodes"] > 0 else 0
            print(f"{algo:<6} {catch_s['total_episodes']:>12} {r['mean']:>10.2f} {r['max']:>10.2f} {tai_ep:>10} {rate:>9.2f}%")
        else:
            print(f"{algo:<6} {'(无数据)':>12}")

    print(f"\n--- 踩踏阶段 (tai_log) 对比 ---")
    print(f"{'算法':<6} {'踩踏Episodes':>12} {'Return均值':>10} {'Return最大':>10} {'踩踏成功':>10} {'成功率':>10}")
    print("-" * 62)
    for algo in ["DQN", "PPO", "SAC"]:
        tai_s = all_results.get(algo, {}).get("tai_log")
        if tai_s:
            r = tai_s["return_stats"]
            g = tai_s["goal_stats"]
            print(f"{algo:<6} {tai_s['total_episodes']:>12} {r['mean']:>10.2f} {r['max']:>10.2f} {int(g['sum']):>10} {g['mean']*100:>9.2f}%")
        else:
            print(f"{algo:<6} {'(无数据)':>12}")
    print()


def main():
    print_separator()
    print("强化学习训练日志分析报告")
    print(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print_separator()

    all_results = {}

    for algo_name in ["DQN", "PPO", "SAC"]:
        print(f"\n{'=' * 72}")
        print(f"  {algo_name} 算法")
        print(f"{'=' * 72}")

        log_files_by_stage = collect_log_files(algo_name)
        algo_results = {}

        for stage in STAGES:
            stage_name_display = "抓取阶段 (catch)" if stage == "catch_log" else "踩踏阶段 (tai)"
            files = log_files_by_stage.get(stage, [])
            print(f"\n  [{stage_name_display}] 发现 {len(files)} 个日志文件:")
            for f in files:
                print(f"    - {os.path.basename(f)}")

            stats = analyze_algo_stage(algo_name, stage, files)
            if stats:
                algo_results[stage] = stats
            else:
                print("    (无有效数据)")

        # 先打印抓取结果（此时尚未知道 tai_episodes）
        catch_stats = algo_results.get("catch_log")
        tai_stats = algo_results.get("tai_log")
        tai_ep_count = tai_stats["total_episodes"] if tai_stats else 0

        if catch_stats:
            print_stats_block("统计结果", catch_stats, is_catch=True, tai_episodes=tai_ep_count)
        if tai_stats:
            print_stats_block("统计结果", tai_stats, is_catch=False)

        all_results[algo_name] = algo_results

    generate_comparison(all_results)

    print_separator()
    print("分析完成")
    print_separator()


if __name__ == "__main__":
    main()
