"""
数据集工具：从 designs_with_metrics.jsonl 加载数据并构建图
"""
import json
import os
import re
import importlib
import torch
from torch_geometric.data import Data
from typing import List, Tuple, Dict, Optional
import numpy as np
from pathlib import Path


def _torch_load_compat(path: str, map_location="cpu"):
    """
    兼容 PyTorch 2.6+ 的 torch.load 默认 weights_only=True 行为。
    对于图对象（torch_geometric.data.Data）必须使用 weights_only=False。
    """
    def _is_weights_only_error(exc: Exception) -> bool:
        msg = str(exc)
        return "Weights only load failed" in msg and "Unsupported global" in msg

    def _resolve_global(global_name: str):
        parts = global_name.split(".")
        for i in range(len(parts), 0, -1):
            module_name = ".".join(parts[:i])
            try:
                obj = importlib.import_module(module_name)
            except Exception:
                continue
            for attr in parts[i:]:
                if not hasattr(obj, attr):
                    obj = None
                    break
                obj = getattr(obj, attr)
            if obj is not None:
                return obj
        return None

    def _load_with_auto_safe_globals():
        add_safe_globals = getattr(torch.serialization, "add_safe_globals", None)
        if add_safe_globals is None:
            return None

        last_exc = None
        for _ in range(12):
            try:
                return torch.load(path, map_location=map_location, weights_only=True)
            except Exception as e:
                last_exc = e
                msg = str(e)
                match = re.search(r"Unsupported global: GLOBAL ([\w\.]+)", msg)
                if not match:
                    break
                obj = _resolve_global(match.group(1))
                if obj is None:
                    break
                add_safe_globals([obj])
        if last_exc is not None:
            raise last_exc
        return None

    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        # 兼容旧版本 PyTorch（没有 weights_only 参数）
        pass
    except Exception as e:
        # 若环境强制 weights_only=True（如 TORCH_FORCE_WEIGHTS_ONLY_LOAD=1），
        # 自动根据报错逐步 allowlist 依赖类。
        if _is_weights_only_error(e):
            loaded = _load_with_auto_safe_globals()
            if loaded is not None:
                return loaded
        raise

    try:
        return torch.load(path, map_location=map_location)
    except Exception as e:
        if _is_weights_only_error(e):
            loaded = _load_with_auto_safe_globals()
            if loaded is not None:
                return loaded
        raise


def mape_loss(pred, true):
    """Mean Absolute Percentage Error"""
    epsilon = 1e-6
    return torch.mean(torch.abs((true - pred) / (true + epsilon)))


def load_designs_from_jsonl(jsonl_file: str) -> List[Dict]:
    """加载 designs_with_metrics.jsonl 文件"""
    designs = []
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                designs.append(json.loads(line))
    return designs


def generate_dataset(
    dataset_dir: str,
    dataset_files: List[str] = None,
    print_info: bool = True
) -> List[Tuple[Data, str]]:
    """
    从目录加载已经处理好的图数据
    
    Args:
        dataset_dir: 图数据所在目录
        dataset_files: 文件列表（如果为None则加载所有.pt文件）
        print_info: 是否打印信息
    
    Returns:
        List of (graph, code) tuples
    """
    if dataset_files is None:
        dataset_files = [f for f in os.listdir(dataset_dir) if f.endswith('.pt')]
    
    pairs = []
    
    for file in dataset_files:
        file_path = os.path.join(dataset_dir, file)
        try:
            graph = _torch_load_compat(file_path, map_location="cpu")
            
            # 假设 code 存储在 graph 的某个属性中，或者单独加载
            # 这里简化处理，直接使用 None
            code = getattr(graph, 'code', None)
            
            pairs.append((graph, code))
        
        except Exception as e:
            if print_info:
                print(f"加载 {file} 失败: {e}")
            continue
    
    if print_info:
        print(f"成功加载 {len(pairs)} 个图")
    
    return pairs


def split_dataset(
    dataset_list: List[Tuple[Data, str]],
    shuffle: bool = True,
    seed: int = 42,
    test_ratio: float = 0.2,
    return_graphs: bool = True
):
    """
    划分训练集和测试集
    
    Args:
        dataset_list: 数据集列表
        shuffle: 是否打乱
        seed: 随机种子
        test_ratio: 测试集比例
        return_graphs: 是否返回原始图列表
    
    Returns:
        如果 return_graphs=True: (train_pairs, test_pairs, train_graphs, test_graphs)
        否则: (train_pairs, test_pairs)
    """
    if shuffle:
        np.random.seed(seed)
        indices = np.random.permutation(len(dataset_list))
    else:
        indices = np.arange(len(dataset_list))
    
    split_idx = int(len(dataset_list) * (1 - test_ratio))
    
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    
    train_pairs = [dataset_list[i] for i in train_indices]
    test_pairs = [dataset_list[i] for i in test_indices]
    
    if return_graphs:
        train_graphs = [pair[0] for pair in train_pairs]
        test_graphs = [pair[0] for pair in test_pairs]
        return train_pairs, test_pairs, train_graphs, test_graphs
    else:
        return train_pairs, test_pairs


def create_dataset_from_jsonl(
    jsonl_file: str,
    output_dir: str,
    use_fhls: bool = True,
    max_samples: int = None
):
    """
    从 designs_with_metrics.jsonl 创建图数据集
    
    Args:
        jsonl_file: 输入的 JSONL 文件
        output_dir: 输出图的目录
        use_fhls: 是否使用 -fhls 编译选项
        max_samples: 最大样本数（用于调试）
    """
    from llvm_ir_to_graph import code_to_graph
    from tqdm import tqdm
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载设计
    designs = load_designs_from_jsonl(jsonl_file)
    
    if max_samples:
        designs = designs[:max_samples]
    
    print(f"加载了 {len(designs)} 个设计")
    
    successful = 0
    failed = 0
    
    for idx, design in enumerate(tqdm(designs, desc="构建图数据集")):
        try:
            # 提取信息
            algo_name = design.get('algo_name', 'unknown')
            design_id = design.get('design_id', f'design_{idx}')
            code = design.get('output', '')
            
            metadata = design.get('metadata', {})
            latency = float(metadata.get('Worst-caseLatency', 0))
            
            # 其他 PPA 指标
            lut = float(metadata.get('LUT', 0))
            ff = float(metadata.get('FF', 0))
            dsp = float(metadata.get('DSP', 0))
            bram = float(metadata.get('BRAM', 0))
            uram = float(metadata.get('URAM', 0))
            cp = float(metadata.get('CP(ns)', 0))
            power = float(metadata.get('Power(W)', 0))
            
            if not code or latency == 0:
                failed += 1
                continue
            
            # 构建图
            graph = code_to_graph(code, use_fhls=use_fhls)
            
            # 添加标签和元信息
            graph.y = torch.tensor(
                [[latency, lut, ff, dsp, bram, uram, cp, power]],
                dtype=torch.float
            )
            
            graph.algo_name = algo_name
            graph.design_id = design_id
            graph.code = code  # 保存代码用于后续训练
            
            # 保存图
            safe_algo_name = algo_name.replace('/', '_').replace(' ', '_')
            output_file = os.path.join(output_dir, f'{safe_algo_name}_{design_id}.pt')
            torch.save(graph, output_file)
            
            successful += 1
        
        except Exception as e:
            print(f"\n处理 {algo_name}/{design_id} 失败: {e}")
            failed += 1
    
    print(f"\n✅ 成功: {successful}, ❌ 失败: {failed}")
    print(f"数据集保存到: {output_dir}")
    
    return successful, failed


def create_dataset_from_design_package(
    design_package_dir: str,
    metrics_jsonl: str,
    output_dir: str,
    max_designs: Optional[int] = None,
):
    """
    从 design_package 目录直接创建图数据集（优先使用现成 .ll 文件）。

    目录结构要求:
      design_package/<algo_name>/<design_id>/

    其中每个 design 目录应至少包含:
      - 一个 .ll 文件
      - 一个 .c 或 .cpp 文件（用于保存 code 文本）

    标签从 metrics_jsonl 的 metadata 中读取:
      - Worst-caseLatency, LUT, FF, DSP, BRAM_18K, URAM, CP(ns), Power(W)
    """
    from llvm_ir_to_graph import LLVMIRGraphBuilder

    design_root = Path(design_package_dir)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    if not design_root.exists():
        raise FileNotFoundError(f"design_package 目录不存在: {design_package_dir}")
    if not Path(metrics_jsonl).exists():
        raise FileNotFoundError(f"metrics jsonl 不存在: {metrics_jsonl}")

    # 建立 metadata 映射:
    # 1) 精确 key: (algo_name, design_id, source_name)
    # 2) 回退 key: (algo_name, design_id)
    metric_map_with_source: Dict[Tuple[str, str, str], Dict] = {}
    metric_map_fallback: Dict[Tuple[str, str], Dict] = {}
    with open(metrics_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            meta = obj.get("metadata", {})
            algo = str(meta.get("algo_name", "")).strip()
            did = str(meta.get("design_id", "")).strip()
            source = str(meta.get("source_name", "")).strip()
            if not algo or not did:
                continue
            # 同 key 若重复，保留首条
            metric_map_fallback.setdefault((algo, did), meta)
            if source:
                metric_map_with_source.setdefault((algo, did, source), meta)

    builder = LLVMIRGraphBuilder()

    total_seen = 0
    saved = 0
    skipped_no_metric = 0
    skipped_no_ll = 0
    skipped_no_code = 0
    output_name_collision = 0
    failed = 0

    # 兼容两种结构:
    # 1) design_package/<algo>/<design_id>/
    # 2) design_package/<suite>/<algo>/<design_id>/   (当前仓库使用该结构)
    design_items: List[Tuple[str, str, Path]] = []
    lv1_dirs = sorted([p for p in design_root.iterdir() if p.is_dir()])
    for lv1 in lv1_dirs:
        lv1_sub = sorted([p for p in lv1.iterdir() if p.is_dir()])
        if any(p.name.startswith("design_") for p in lv1_sub):
            # 结构 1
            for design_dir in lv1_sub:
                if design_dir.name.startswith("design_"):
                    design_items.append(("(root)", lv1.name, design_dir))
            continue

        # 尝试结构 2
        for algo_dir in lv1_sub:
            design_dirs = sorted([p for p in algo_dir.iterdir() if p.is_dir()])
            for design_dir in design_dirs:
                if design_dir.name.startswith("design_"):
                    design_items.append((lv1.name, algo_dir.name, design_dir))

    for suite_name, algo_name, design_dir in design_items:
        if max_designs is not None and total_seen >= max_designs:
            break
        total_seen += 1
        if total_seen % 10000 == 0:
            print(
                f"[progress] seen={total_seen}, saved={saved}, "
                f"skip_metric={skipped_no_metric}, skip_no_ll={skipped_no_ll}, "
                f"skip_no_code={skipped_no_code}, failed={failed}"
            )

        design_id = design_dir.name
        meta = (
            metric_map_with_source.get((algo_name, design_id, suite_name))
            or metric_map_fallback.get((algo_name, design_id))
        )
        if meta is None:
            skipped_no_metric += 1
            continue

        ll_files = sorted(design_dir.glob("*.ll"))
        if not ll_files:
            skipped_no_ll += 1
            continue

        code_file = None
        c_files = sorted(design_dir.glob("*.c"))
        cpp_files = sorted(design_dir.glob("*.cpp"))
        if c_files:
            code_file = c_files[0]
        elif cpp_files:
            code_file = cpp_files[0]
        if code_file is None:
            skipped_no_code += 1
            continue

        try:
            graph = builder.ir_to_graph(str(ll_files[0]))
            code = code_file.read_text(encoding="utf-8", errors="ignore")

            latency = float(meta.get("Worst-caseLatency") or 0.0)
            lut = float(meta.get("LUT") or 0.0)
            ff = float(meta.get("FF") or 0.0)
            dsp = float(meta.get("DSP") or 0.0)
            bram = float(meta.get("BRAM_18K") or 0.0)
            uram = float(meta.get("URAM") or 0.0)
            cp = float(meta.get("CP(ns)") or 0.0)
            power = float(meta.get("Power(W)") or 0.0)

            graph.y = torch.tensor(
                [[latency, lut, ff, dsp, bram, uram, cp, power]],
                dtype=torch.float,
            )
            graph.suite_name = suite_name
            graph.algo_name = algo_name
            graph.design_id = design_id
            graph.code = code
            graph.ll_file = str(ll_files[0])
            graph.code_file = str(code_file)

            safe_suite = suite_name.replace("/", "_").replace(" ", "_")
            safe_algo = algo_name.replace("/", "_").replace(" ", "_")
            out_file = output_root / f"{safe_suite}__{safe_algo}_{design_id}.pt"
            if out_file.exists():
                output_name_collision += 1
                stem = out_file.stem
                suffix = out_file.suffix
                idx = 2
                while True:
                    candidate = output_root / f"{stem}__dup{idx}{suffix}"
                    if not candidate.exists():
                        out_file = candidate
                        break
                    idx += 1
            torch.save(graph, out_file)
            saved += 1
        except Exception as e:
            print(f"处理失败 {algo_name}/{design_id}: {e}")
            failed += 1

        if max_designs is not None and total_seen >= max_designs:
            break

    print("\n从 design_package 构建图数据集完成")
    print(f"  扫描 design 数: {total_seen}")
    print(f"  成功保存: {saved}")
    print(f"  跳过(无指标): {skipped_no_metric}")
    print(f"  跳过(无.ll): {skipped_no_ll}")
    print(f"  跳过(无代码): {skipped_no_code}")
    print(f"  文件名冲突回退: {output_name_collision}")
    print(f"  失败: {failed}")
    print(f"  输出目录: {output_dir}")

    return {
        "total_seen": total_seen,
        "saved": saved,
        "skipped_no_metric": skipped_no_metric,
        "skipped_no_ll": skipped_no_ll,
        "skipped_no_code": skipped_no_code,
        "output_name_collision": output_name_collision,
        "failed": failed,
        "output_dir": str(output_root),
    }


def load_graph_dataset(
    graph_dir: str,
    target_metric: str = 'latency',
    normalize: bool = True
) -> List[Tuple[Data, str]]:
    """
    加载已保存的图数据集
    
    Args:
        graph_dir: 图数据所在目录
        target_metric: 目标指标 (latency, lut, ff, dsp, bram, uram, cp, power)
        normalize: 是否归一化标签
    
    Returns:
        List of (graph, code) tuples
    """
    metric_index = {
        'latency': 0, 'lut': 1, 'ff': 2, 'dsp': 3,
        'bram': 4, 'uram': 5, 'cp': 6, 'power': 7
    }
    
    target_idx = metric_index.get(target_metric, 0)
    
    # 加载所有图
    graph_files = [f for f in os.listdir(graph_dir) if f.endswith('.pt')]
    
    pairs = []
    all_targets = []
    
    for file in graph_files:
        file_path = os.path.join(graph_dir, file)
        try:
            graph = _torch_load_compat(file_path, map_location="cpu")
            
            # 提取目标指标
            if hasattr(graph, 'y') and graph.y is not None:
                target_value = graph.y[0, target_idx].item()
                all_targets.append(target_value)
                
                # 提取代码
                code = getattr(graph, 'code', None)
                
                pairs.append((graph, code))
        
        except Exception as e:
            print(f"加载 {file} 失败: {e}")
            continue
    
    # 归一化标签
    if normalize and pairs:
        mean = np.mean(all_targets)
        std = np.std(all_targets) + 1e-6
        
        print(f"标签统计 ({target_metric}):")
        print(f"  均值: {mean:.2f}")
        print(f"  标准差: {std:.2f}")
        print(f"  最小值: {min(all_targets):.2f}")
        print(f"  最大值: {max(all_targets):.2f}")
        
        for graph, code in pairs:
            original_y = graph.y.clone()
            graph.y[:, target_idx] = (graph.y[:, target_idx] - mean) / std
            
            # 保存原始值和归一化参数
            graph.y_original = original_y
            graph.y_mean = mean
            graph.y_std = std
    
    print(f"成功加载 {len(pairs)} 个图")
    
    return pairs


if __name__ == '__main__':
    # 测试数据集创建
    print("=" * 80)
    print("测试数据集创建")
    print("=" * 80)
    
    # 示例：从 JSONL 创建图数据集
    _here = Path(__file__).resolve().parent
    jsonl_file = str(_here / 'dataset' / 'designs_with_metrics_train_test_merged.jsonl')
    output_dir = str(_here / 'dataset' / 'graphs')
    
    if Path(jsonl_file).exists():
        print(f"\n创建数据集从: {jsonl_file}")
        print(f"输出到: {output_dir}")
        print("\n⚠️  这可能需要较长时间，建议先用小样本测试...")
        print("提示：设置 max_samples=10 进行快速测试\n")
        
        # 测试：只处理前 5 个样本
        # create_dataset_from_jsonl(jsonl_file, output_dir, use_fhls=True, max_samples=5)
        
        print("如需完整运行，请取消注释上面的代码")
    else:
        print(f"⚠️  文件不存在: {jsonl_file}")
    
    print("\n" + "=" * 80)
    print("使用示例:")
    print("  # 1. 创建图数据集")
    print("  create_dataset_from_jsonl('designs.jsonl', 'graphs/', use_fhls=True)")
    print()
    print("  # 2. 加载图数据集")
    print("  pairs = load_graph_dataset('graphs/', target_metric='latency')")
    print()
    print("  # 3. 划分训练/测试集")
    print("  train_pairs, test_pairs, _, _ = split_dataset(pairs, test_ratio=0.2)")
