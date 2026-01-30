import json
import os
import time
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from dataclasses import dataclass, field
from collections import defaultdict

from api import PaLSim

C = 0.1
B = 5.0
TOP_K = 5  # cut off


@dataclass
class HistoryState:
    """history state"""
    history: List[Tuple[str, str]]  # [(st1, bp1), (st2, bp2), ...]
    prob: float  # joint prob


class ContextCache:
    """Context cache manager
    
    two types of cache:
    1. distribution cache (player, context) → joint_distribution
       - same context uses same player model prediction result
       - can be reused across strokes in rally
    
    2. winrate cache (player, context) → winrate_matrix
       - winrate only depends on context, not current action
       - can be reused across strokes in rally
    """
    def __init__(self):
        self.dist_cache: Dict[Tuple, Dict] = {}  # distribution cache
        self.winrate_cache: Dict[Tuple, np.ndarray] = {}  # winrate cache
        self.dist_hits = 0
        self.dist_misses = 0
        self.wr_hits = 0
        self.wr_misses = 0
    
    def _get_key(self, player: str, st_t2, bp_t2, st_t1, bp_t1) -> Tuple:
        """generate cache key, containing player info"""
        return (player, st_t2, bp_t2, st_t1, bp_t1)
    
    def get_distribution(self, api: PaLSim, st_t2, bp_t2, st_t1, bp_t1) -> Dict:
        """get distribution (with cache)"""
        key = self._get_key(api.player, st_t2, bp_t2, st_t1, bp_t1)
        if key in self.dist_cache:
            self.dist_hits += 1
            return self.dist_cache[key]
        
        self.dist_misses += 1
        result = api.predict_distribution(
            st_t2=st_t2, bp_t2=bp_t2,
            st_t1=st_t1, bp_t1=bp_t1
        )
        self.dist_cache[key] = result
        return result
    
    def get_winrate(self, api: PaLSim, joint: np.ndarray, st_t2, bp_t2, st_t1, bp_t1) -> np.ndarray:
        """get winrate matrix (with cache)
        
        note: winrate matrix only depends on (player, context), not joint distribution
        joint parameter is only used in the first calculation
        """
        key = self._get_key(api.player, st_t2, bp_t2, st_t1, bp_t1)
        if key in self.winrate_cache:
            self.wr_hits += 1
            return self.winrate_cache[key]
        
        self.wr_misses += 1
        wr_result = api.predict_winrate(
            joint.tolist(),
            st_t2=st_t2, bp_t2=bp_t2,
            st_t1=st_t1, bp_t1=bp_t1
        )
        winrate_matrix = np.array(wr_result['winrate_matrix'])
        self.winrate_cache[key] = winrate_matrix
        return winrate_matrix
    
    def clear(self):
        """clear cache (called at the start of each rally)"""
        self.dist_cache.clear()
        self.winrate_cache.clear()
    
    def get_stats(self) -> Dict:
        """get cache statistics"""
        total_dist = self.dist_hits + self.dist_misses
        total_wr = self.wr_hits + self.wr_misses
        total = total_dist + total_wr
        return {
            'distribution': {
                'hits': self.dist_hits,
                'misses': self.dist_misses,
                'hit_rate': self.dist_hits / total_dist if total_dist > 0 else 0
            },
            'winrate': {
                'hits': self.wr_hits,
                'misses': self.wr_misses,
                'hit_rate': self.wr_hits / total_wr if total_wr > 0 else 0
            },
            'total_hits': self.dist_hits + self.wr_hits,
            'total_misses': self.dist_misses + self.wr_misses,
            'total_hit_rate': (self.dist_hits + self.wr_hits) / total if total > 0 else 0
        }


def calculate_delta_tp(tp: float, C: float = 0.1, B: float = 5.0) -> float:
    """calculate the maximum probability increase for a stroke"""
    delta_percent = C + 4 * B * tp * (1 - tp)
    return delta_percent / 100


def parse_context(context_str: str) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """parse context string, extract the last two actions as t-2 and t-1"""
    if not context_str or context_str.strip() == "":
        return None, None, None, None
    
    parts = [p.strip() for p in context_str.split('|')]
    st_t2, bp_t2, st_t1, bp_t1 = None, None, None, None
    
    if len(parts) >= 1:
        last = parts[-1]
        try:
            action = last.split(':')[1]
            st_t1, bp_t1 = action.split('->')
        except:
            pass
    
    if len(parts) >= 2:
        second_last = parts[-2]
        try:
            action = second_last.split(':')[1]
            st_t2, bp_t2 = action.split('->')
        except:
            pass
    
    return st_t2, bp_t2, st_t1, bp_t1


def get_top_k_from_joint(joint: np.ndarray, st_labels: List[str], bp_labels: List[str], k: int) -> List[Dict]:
    """get top-k (ST,BP) combinations from joint distribution"""
    flat = [(i, j, joint[i, j]) for i in range(len(st_labels)) for j in range(len(bp_labels))]
    flat.sort(key=lambda x: x[2], reverse=True)
    top_k = []
    for i, j, prob in flat[:k]:
        if prob > 1e-6:
            top_k.append({'st': st_labels[i], 'bp': bp_labels[j], 'prob': float(prob)})
    return top_k


def forward_propagate_from_stroke(
    strokes: List[Dict],
    start_idx: int,
    player_to_api: Dict,
    cache: ContextCache,
    device: torch.device,
    top_k: int = TOP_K
) -> Dict:
    
    if start_idx >= len(strokes):
        return {}
    
    first_api = player_to_api[strokes[0]['player']][1]
    st_labels = first_api.st_labels
    bp_labels = first_api.bp_labels
    n_st, n_bp = len(st_labels), len(bp_labels)
    
    if start_idx == 0:
        init_context = (None, None, None, None)
    elif start_idx == 1:
        prev_stroke = strokes[0]
        init_context = (None, None, prev_stroke['st'], prev_stroke['bp'])
    else:
        prev2 = strokes[start_idx - 2]
        prev1 = strokes[start_idx - 1]
        init_context = (prev2['st'], prev2['bp'], prev1['st'], prev1['bp'])
    
    current_stroke = strokes[start_idx]
    current_player = current_stroke['player']
    model_name, api = player_to_api[current_player]
    
    st_t2, bp_t2, st_t1, bp_t1 = init_context
    dist_result = cache.get_distribution(api, st_t2, bp_t2, st_t1, bp_t1)
    current_joint = torch.tensor(dist_result['joint'], device=device, dtype=torch.float32)
    current_joint_np = current_joint.cpu().numpy()
    
    current_top_k_raw = get_top_k_from_joint(current_joint_np, st_labels, bp_labels, top_k)
    total_prob = sum(item['prob'] for item in current_top_k_raw)
    current_top_k = [{'st': item['st'], 'bp': item['bp'], 'prob': item['prob'] / total_prob if total_prob > 0 else 0}
                     for item in current_top_k_raw]
    
    propagation_chain = []
    
    prev_top_k = [(item['st'], item['bp'], item['prob']) for item in current_top_k]
    prev_prev_fixed = (init_context[2], init_context[3])
    
    for prop_idx in range(start_idx + 1, len(strokes)):
        prop_stroke = strokes[prop_idx]
        prop_player = prop_stroke['player']
        prop_model, prop_api = player_to_api[prop_player]
        
        if prop_idx == start_idx + 1:
            contexts_with_probs = []
            for st_t1, bp_t1, prob_t1 in prev_top_k:
                ctx = (prev_prev_fixed[0], prev_prev_fixed[1], st_t1, bp_t1)
                contexts_with_probs.append((ctx, prob_t1))
        else:
            contexts_with_probs = []
            for st_t2, bp_t2, prob_t2 in prev_prev_top_k:
                for st_t1, bp_t1, prob_t1 in prev_top_k:
                    flow_key = (st_t2, bp_t2, st_t1, bp_t1)
                    if flow_key in flow_probs:
                        ctx = (st_t2, bp_t2, st_t1, bp_t1)
                        contexts_with_probs.append((ctx, flow_probs[flow_key]))
            
            contexts_with_probs.sort(key=lambda x: x[1], reverse=True)
            contexts_with_probs = contexts_with_probs[:top_k]
            
            total_ctx_prob = sum(p for _, p in contexts_with_probs)
            if total_ctx_prob > 0:
                contexts_with_probs = [(ctx, p / total_ctx_prob) for ctx, p in contexts_with_probs]
        
        aggregated_joint = np.zeros((n_st, n_bp))
        context_joints = {}  
        
        for ctx, ctx_prob in contexts_with_probs:
            st_t2, bp_t2, st_t1, bp_t1 = ctx
            dist_result = cache.get_distribution(prop_api, st_t2, bp_t2, st_t1, bp_t1)
            joint = np.array(dist_result['joint'])
            context_joints[ctx] = joint
            aggregated_joint += ctx_prob * joint
        
        curr_top_k_raw = get_top_k_from_joint(aggregated_joint, st_labels, bp_labels, top_k)
        total_prob = sum(item['prob'] for item in curr_top_k_raw)
        curr_top_k = [{'st': item['st'], 'bp': item['bp'], 'prob': item['prob'] / total_prob if total_prob > 0 else 0}
                      for item in curr_top_k_raw]
        
        flow_matrix = defaultdict(float)  
        
        for ctx, ctx_prob in contexts_with_probs:
            st_t2, bp_t2, st_t1, bp_t1 = ctx
            joint = context_joints[ctx]
            
            for curr_item in curr_top_k:
                st_curr, bp_curr = curr_item['st'], curr_item['bp']
                i_curr = st_labels.index(st_curr)
                j_curr = bp_labels.index(bp_curr)
                
                flow_prob = ctx_prob * joint[i_curr, j_curr]
                flow_matrix[(st_t1, bp_t1, st_curr, bp_curr)] += flow_prob
        
        total_flow = sum(flow_matrix.values())
        flow_probs = {}  
        top5_flow = {}   
        
        prev_top_k_set = [(st, bp) for st, bp, _ in prev_top_k]
        curr_top_k_set = [(item['st'], item['bp']) for item in curr_top_k]
        
        for st_prev, bp_prev in prev_top_k_set:
            for st_curr, bp_curr in curr_top_k_set:
                key = (st_prev, bp_prev, st_curr, bp_curr)
                prob = flow_matrix.get(key, 0.0)
                ratio = prob / total_flow if total_flow > 0 else 0.0
                flow_probs[key] = ratio
                flow_key = f"({st_prev}->{bp_prev}) -> ({st_curr}->{bp_curr})"
                top5_flow[flow_key] = round(ratio, 4)
        
        weighted_joint = np.zeros((n_st, n_bp))
        for item in curr_top_k:
            i = st_labels.index(item['st'])
            j = bp_labels.index(item['bp'])
            weighted_joint[i, j] = item['prob']
        
        propagation_chain.append({
            'stroke_idx': prop_idx,
            'stroke_key': prop_stroke['key'],
            'player': prop_player,
            'model': prop_model,
            'num_contexts_evaluated': len(contexts_with_probs),
            'top_k_distribution': curr_top_k,
            'st_probs': {st: float(aggregated_joint[i, :].sum()) for i, st in enumerate(st_labels)},
            'bp_probs': {bp: float(aggregated_joint[:, j].sum()) for j, bp in enumerate(bp_labels)},
            'top5_to_top5_flow': top5_flow
        })
        
        prev_prev_top_k = prev_top_k
        prev_top_k = [(item['st'], item['bp'], item['prob']) for item in curr_top_k]
    
    return {
        'current_stroke': {
            'stroke_idx': start_idx,
            'stroke_key': current_stroke['key'],
            'player': current_player,
            'model': model_name,
            'context': {
                't-2': f"{init_context[0]}->{init_context[1]}" if init_context[0] else None,
                't-1': f"{init_context[2]}->{init_context[3]}" if init_context[2] else None
            },
            'top_k_distribution': current_top_k,
            'st_probs': {st: float(current_joint_np[i, :].sum()) for i, st in enumerate(st_labels)},
            'bp_probs': {bp: float(current_joint_np[:, j].sum()) for j, bp in enumerate(bp_labels)},
            'joint': current_joint_np
        },
        'forward_propagation': propagation_chain,
        'api': api,
        'st_labels': st_labels,
        'bp_labels': bp_labels
    }


def process_rally_file_v2(
    rally_path: Path, 
    output_dir: Path, 
    api_m: PaLSim, 
    api_w: PaLSim,
    cache: ContextCache
) -> Dict:
    cache.clear()
    
    with open(rally_path, 'r', encoding='utf-8') as f:
        rally_data = json.load(f)
    
    meta_info = rally_data['meta_info']
    rally_info = rally_data['rally_info']
    
    player_to_api = {}
    for player_key in ['player0', 'player1']:
        name = meta_info[player_key]['name']
        if name == 'W':
            player_to_api[player_key] = ('w', api_w)
        elif name == 'M':
            player_to_api[player_key] = ('m', api_m)
        else:
            player_to_api[player_key] = ('m', api_m)
    
    rally_name = rally_path.stem
    stroke_output_dir = output_dir / rally_name
    stroke_output_dir.mkdir(parents=True, exist_ok=True)
    
    strokes = []
    for stroke_key in sorted(rally_info.keys(), key=lambda x: int(x.replace('stroke', ''))):
        stroke_data = rally_info[stroke_key]
        strokes.append({
            'key': stroke_key,
            'player': stroke_data['player'],
            'st': stroke_data['strokeTech'],
            'bp': stroke_data['ballPlacement'] if stroke_data['ballPlacement'] else None
        })
    
    if not strokes:
        return {
            'rally_name': rally_name,
            'processed_strokes': 0,
            'output_dir': str(stroke_output_dir),
            'updated_rally_file': str(output_dir / f"{rally_name}_updated.json")
        }
    
    device = api_m.device
    
    processed_count = 0

    for stroke_idx, stroke in enumerate(strokes):
        stroke_key = stroke['key']
        actual_st = stroke['st']
        actual_bp = stroke['bp']
        player = stroke['player']
        
        prop_result = forward_propagate_from_stroke(
            strokes, stroke_idx, player_to_api, cache, device, TOP_K
        )
        
        if not prop_result:
            continue
        
        api = prop_result['api']
        st_labels = prop_result['st_labels']
        bp_labels = prop_result['bp_labels']
        current_info = prop_result['current_stroke']
        
        joint = current_info['joint']
        context = current_info['context']
        st_t2 = context['t-2'].split('->')[0] if context['t-2'] else None
        bp_t2 = context['t-2'].split('->')[1] if context['t-2'] else None
        st_t1 = context['t-1'].split('->')[0] if context['t-1'] else None
        bp_t1 = context['t-1'].split('->')[1] if context['t-1'] else None
        
        winrate_matrix = cache.get_winrate(api, joint, st_t2, bp_t2, st_t1, bp_t1)
        baseline_winrate = float(np.sum(joint * winrate_matrix))
        
        adjustments = []
        for i, st in enumerate(st_labels):
            for j, bp in enumerate(bp_labels):
                prob = float(joint[i, j])
                winrate = float(winrate_matrix[i, j])
                delta_tp = calculate_delta_tp(prob, C, B)
                winrate_change = delta_tp * (winrate - baseline_winrate)
                adjustments.append({
                    'st': st, 'bp': bp, 'prob': prob, 'winrate': winrate,
                    'winrate_change': winrate_change
                })
        adjustments.sort(key=lambda x: x['winrate_change'], reverse=True)
        for rank, adj in enumerate(adjustments, 1):
            adj['rank'] = rank
        
        usage_sorted = sorted(adjustments, key=lambda x: x['prob'], reverse=True)
        top5_usage = usage_sorted[:5]
        top5_by_winrate = sorted(top5_usage, key=lambda x: x['winrate'])
        frequent_but_losing = top5_by_winrate[:2]
        
        avg_winrate = float(np.mean([a['winrate'] for a in adjustments]))
        not_in_top5_usage = [a for a in adjustments if a not in top5_usage]
        rare_but_winning = [a for a in not_in_top5_usage if a['winrate'] > avg_winrate + 0.10]
        rare_but_winning.sort(key=lambda x: x['prob'])
        rare_but_winning = rare_but_winning[:5]
        
        actual_rank = None
        actual_info = None
        if actual_bp:
            for adj in adjustments:
                if adj['st'] == actual_st and adj['bp'] == actual_bp:
                    actual_rank = adj['rank']
                    actual_info = adj
                    break
        
        analysis = {
            'stroke_key': stroke_key,
            'stroke_index': stroke_idx,
            'player': player,
            'player_name': meta_info[player]['name'],
            'model_used': current_info['model'],
            'context': current_info['context'],
            'actual_action': {
                'st': actual_st,
                'bp': actual_bp,
                'rank': actual_rank,
                'info': actual_info
            },
            'baseline_winrate': baseline_winrate,
            'avg_winrate': avg_winrate,
            'current_stroke_analysis': {
                'top_k_distribution': current_info['top_k_distribution'],
                'st_probs': current_info['st_probs'],
                'bp_probs': current_info['bp_probs'],
                'joint_distribution': joint.tolist(),
                'winrate_matrix': winrate_matrix.tolist()
            },
            'top5_recommendations': adjustments[:5],
            'frequent_but_losing': frequent_but_losing,
            'rare_but_winning': rare_but_winning,
            'forward_propagation': prop_result['forward_propagation']
        }
        
        analysis_filename = f"{stroke_key}_analysis.json"
        analysis_path = stroke_output_dir / analysis_filename
        with open(analysis_path, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, ensure_ascii=False, indent=2)
        
        rally_info[stroke_key]['analysis_file'] = str(analysis_path.relative_to(output_dir.parent))
        processed_count += 1
    
    updated_rally_path = output_dir / f"{rally_name}_updated.json"
    with open(updated_rally_path, 'w', encoding='utf-8') as f:
        json.dump(rally_data, f, ensure_ascii=False, indent=2)
    
    return {
        'rally_name': rally_name,
        'processed_strokes': processed_count,
        'output_dir': str(stroke_output_dir),
        'updated_rally_file': str(updated_rally_path)
    }


def process_all_rallies_v2(input_dir: Path, output_dir: Path, device: str = 'cuda:1'):
    
    api_m = PaLSim(player='m', device=device)
    print(f"    TCA-GF + GQA-OS loaded")
    
    api_w = PaLSim(player='w', device=device)
    print(f"    TCA-GF + GQA-OS loaded")
    
    print(f"\n  All models loaded to: {device}")
    
    cache = ContextCache()
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    rally_files = sorted(input_dir.glob('*.json'))
    total_rallies = len(rally_files)
    print(f"\n[2/3] Found {total_rallies} rally files")
    
    timing_start_idx = max(0, total_rallies - 500)
    print(f"  Timing will start from rally {timing_start_idx + 1} (last 500 rallies)")
    
    print("\n[3/3] Processing rally files...")
    results = []
    timing_started = False
    timing_start_time = None
    timed_rally_count = 0
    timed_stroke_count = 0
    
    for i, rally_file in enumerate(rally_files):
        if i == timing_start_idx:
            timing_started = True
            timing_start_time = time.time()
        
        print(f"  [{i + 1}/{total_rallies}] Processing {rally_file.name}...", end='', flush=True)
        try:
            result = process_rally_file_v2(rally_file, output_dir, api_m, api_w, cache)
            results.append(result)
            print(f" -> {result['processed_strokes']} strokes")
            
            if timing_started:
                timed_rally_count += 1
                timed_stroke_count += result['processed_strokes']
                
        except Exception as e:
            print(f" -> error: {e}")
            import traceback
            traceback.print_exc()
            results.append({'rally_name': rally_file.stem, 'error': str(e)})
            if timing_started:
                timed_rally_count += 1
    
    timing_end_time = time.time()
    if timing_start_time is not None:
        total_timing_duration = timing_end_time - timing_start_time
        avg_time_per_rally = total_timing_duration / timed_rally_count if timed_rally_count > 0 else 0
        avg_time_per_stroke = total_timing_duration / timed_stroke_count if timed_stroke_count > 0 else 0
    else:
        total_timing_duration = 0
        avg_time_per_rally = 0
        avg_time_per_stroke = 0
    
    cache_stats = cache.get_stats()
    
    summary = {
        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'version': 'v2_forward_propagation_cached',
        'input_dir': str(input_dir),
        'output_dir': str(output_dir),
        'total_rallies': total_rallies,
        'top_k': TOP_K,
        'timing_info': {
            'timed_rallies': timed_rally_count,
            'timed_strokes': timed_stroke_count,
            'total_time_seconds': total_timing_duration,
            'avg_time_per_rally_seconds': avg_time_per_rally,
            'avg_time_per_stroke_seconds': avg_time_per_stroke
        },
        'cache_stats': cache_stats,
        'results': results
    }
    summary_path = output_dir / 'processing_summary.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print("\n" + "=" * 60)
    print("Processing completed!")
    print("=" * 60)
    print(f"Total rallies processed: {total_rallies}")
    print(f"Output directory: {output_dir}")
    print(f"Summary file: {summary_path}")
    
    print("\n" + "-" * 60)
    print("Timing statistics (last 500 rallies):")
    print("-" * 60)
    print(f"  Timed rallies: {timed_rally_count}")
    print(f"  Timed strokes: {timed_stroke_count}")
    print(f"  Total time: {total_timing_duration:.2f} seconds")
    print(f"  Average time per rally: {avg_time_per_rally:.4f} seconds")
    print(f"  Average time per stroke: {avg_time_per_stroke:.4f} seconds")
    if total_timing_duration > 0:
        print(f"  Throughput: {timed_stroke_count / total_timing_duration:.2f} strokes/second")
        
    return summary


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Process rally files with PaLSim V2')
    parser.add_argument('--input', '-i', type=str, default='../example_rally',
                        help='Input directory containing rally JSON files')
    parser.add_argument('--output', '-o', type=str, default='../example_rally_analysis_test',
                        help='Output directory for analysis results')
    parser.add_argument('--device', '-d', type=str, default='cuda:1',
                        help='GPU device to use (default: cuda:1)')
    args = parser.parse_args()
    
    pipeline_dir = Path(__file__).parent
    input_dir = pipeline_dir / args.input
    output_dir = pipeline_dir / args.output
    
    process_all_rallies_v2(input_dir, output_dir, device=args.device)


if __name__ == "__main__":
    main()
