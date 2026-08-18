import osmnx as ox
import networkx as nx
import numpy as np
from scipy import stats
import time
import heapq
import matplotlib.pyplot as plt
import csv


def calculate_GDF(
    name, 
    coords, 
    radius_m, 
    n_samples=500, 
    min_dist_m=100.0, 
    seed=17
):
    """
    Calculates Global Detour Factor using travel time (TT) (seconds).
    Heuristic time lower-bound = Euclidean Distance / Max Network Speed.
    """
    np.random.seed(seed)
    print(f"[{name}] Fetching network (radius = {radius_m}m)...")
    
    # Download graph
    G = ox.graph_from_point(coords, dist=radius_m, network_type="drive")
    G_proj = ox.project_graph(G)
    
    # Put in missing speeds and times
    G_proj = ox.add_edge_speeds(G_proj)
    G_proj = ox.add_edge_travel_times(G_proj)
    
    # Get Strongly Connected Component (SCC)
    G_scc = ox.truncate.largest_component(G_proj, strongly=True)
    
    # Find max speed (for h_Euc)
    speeds_kph = [
        d.get("speed_kph", 50.0) 
        for u, v, k, d in G_scc.edges(keys=True, data=True)
    ]
    v_max_kph = max(speeds_kph)
    v_max_mps = v_max_kph * (1000.0 / 3600.0) # km/h -> m/s
    
    nodes = list(G_scc.nodes())
    detour_ratios = []
    
    print(f"[{name}] Network max speed = {v_max_kph:.1f} km/h ({v_max_mps:.2f} m/s)")
    print(f"[{name}] Sampling {n_samples} OD pairs...")
    
    # Monte Carlo sampling
    while len(detour_ratios) < n_samples:
        u, v = np.random.choice(nodes, size=2, replace=False)
        
        # Coordinates (meters)
        x1, y1 = G_scc.nodes[u]["x"], G_scc.nodes[u]["y"]
        x2, y2 = G_scc.nodes[v]["x"], G_scc.nodes[v]["y"]
        
        d_euclid_m = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        # Get rid of really close pairs
        if d_euclid_m < min_dist_m:
            continue
            
        # Actual shortest graph TT (seconds)
        t_graph = nx.shortest_path_length(G_scc, u, v, weight="travel_time")
        
        # Admissible lower-bound straight-line TT (seconds)
        t_heur = d_euclid_m / v_max_mps
        
        detour_ratios.append(t_graph / t_heur)
        
    detour_ratios = np.array(detour_ratios)
    
    # Stat Summary
    mean_D = np.mean(detour_ratios)
    std_err = stats.sem(detour_ratios)
    ci_95 = stats.t.interval(0.95, df=n_samples-1, loc=mean_D, scale=std_err)
    
    print(f"[{name}] Done! |V|={len(G_scc.nodes())}, |E|={len(G_scc.edges())}")
    print(f"  -> D_hat (Time) = {mean_D:.4f} (95% CI: [{ci_95[0]:.4f}, {ci_95[1]:.4f}])\n")
    
    return {
        "tier": name,
        "nodes": len(G_scc.nodes()),
        "edges": len(G_scc.edges()),
        "v_max_kph": v_max_kph,
        "D_hat_time": mean_D,
        "ci_95": ci_95,
        "ratios": detour_ratios
    }


# =====================================================================
# 1. LANDMARK SELECTION & PRECOMPUTATION FOR ALT HEURISTIC
# =====================================================================

def select_farthest_landmarks(G, num_landmarks=16, seed=17):
    """
    Selects landmarks using the Farthest Landmarking (Farthest-Point) strategy.
    Measures graph distance using travel_time.
    """
    np.random.seed(seed)
    nodes = list(G.nodes())
    landmarks = [np.random.choice(nodes)]
    
    # Distance tracker from current set of landmarks
    min_dists = {n: float('inf') for n in nodes}
    
    for _ in range(1, num_landmarks):
        last_lm = landmarks[-1]
        # Single-source Dijkstra from the newest landmark
        sp_lengths = nx.single_source_dijkstra_path_length(G, last_lm, weight='travel_time')
        
        for n in nodes:
            if n in sp_lengths:
                min_dists[n] = min(min_dists[n], sp_lengths[n])
                
        # Pick node with maximum shortest-path distance to the closest landmark
        next_lm = max(min_dists, key=min_dists.get)
        landmarks.append(next_lm)
        
    return landmarks

def precompute_alt_distances(G, landmarks):
    """
    Precomputes d(L, n) and d(n, L) for all landmarks L and nodes n.
    """
    d_L_to_n = {}
    d_n_to_L = {}
    
    # Reverse graph for d(n, L) - distance FROM all nodes TO landmark
    G_rev = G.reverse(copy=True)
    
    for L in landmarks:
        d_L_to_n[L] = nx.single_source_dijkstra_path_length(G, L, weight='travel_time')
        d_n_to_L[L] = nx.single_source_dijkstra_path_length(G_rev, L, weight='travel_time')
        
    return d_L_to_n, d_n_to_L


# =====================================================================
# 2. HEURISTIC FUNCTIONS
# =====================================================================

def euclidean_time_heuristic(u, v, G, v_max_mps):
    """Admissible Euclidean time heuristic: d_euclid / v_max"""
    x1, y1 = G.nodes[u]['x'], G.nodes[u]['y']
    x2, y2 = G.nodes[v]['x'], G.nodes[v]['y']
    d_euclid = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return d_euclid / v_max_mps

def alt_heuristic(u, v, landmarks_subset, d_L_to_n, d_n_to_L):
    """
    ALT Heuristic using Triangle Inequality across a subset of active landmarks:
    h(u, v) = max_L { max( d(L, v) - d(L, u), d(u, L) - d(v, L) ) }
    """
    max_h = 0.0
    for L in landmarks_subset:
        # Lower bound 1: d(L, v) - d(L, u)
        if v in d_L_to_n[L] and u in d_L_to_n[L]:
            h1 = d_L_to_n[L][v] - d_L_to_n[L][u]
            if h1 > max_h:
                max_h = h1
                
        # Lower bound 2: d(u, L) - d(v, L)
        if u in d_n_to_L[L] and v in d_n_to_L[L]:
            h2 = d_n_to_L[L][u] - d_n_to_L[L][v]
            if h2 > max_h:
                max_h = h2
                
    return max_h


# =====================================================================
# 3. A* ALGORITHM
# =====================================================================

def a_star(G, source, target, heuristic_fn):
    """
    Runs A* and counts expanded nodes (popped from open set) and execution time.
    """
    start_time = time.perf_counter()
    
    open_set = []
    heapq.heappush(open_set, (0, source))
    
    g_score = {source: 0.0}
    f_score = {source: heuristic_fn(source, target)}
    
    nodes_expanded = 0
    closed_set = set()
    
    while open_set:
        _, current = heapq.heappop(open_set)
        
        if current in closed_set:
            continue
            
        closed_set.add(current)
        nodes_expanded += 1
        
        if current == target:
            runtime_ms = (time.perf_counter() - start_time) * 1000.0
            return nodes_expanded, runtime_ms, g_score[target]
            
        for neighbor, edge_data in G[current].items():
            # Handle multigraph edges by taking minimum travel_time
            weight = min(d.get('travel_time', 1.0) for d in edge_data.values())
            tentative_g = g_score[current] + weight
            
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                g_score[neighbor] = tentative_g
                f = tentative_g + heuristic_fn(neighbor, target)
                f_score[neighbor] = f
                heapq.heappush(open_set, (f, neighbor))
                
    runtime_ms = (time.perf_counter() - start_time) * 1000.0
    return nodes_expanded, runtime_ms, float('inf')


# =====================================================================
# 4. MULTI-LANDMARK TRIAL EXECUTION HARNESS WITH CSV EXPORT
# =====================================================================

def run_trials(targets, n_samples=500, min_dist_m=100.0, landmark_counts=[4, 8, 16], seed=17, csv_filename="ee_individual_trials_raw.csv"):
    all_results = []
    max_landmarks = max(landmark_counts)

    # Initialize CSV file and write header
    csv_header = [
        "trial_id",
        "tier_name",
        "origin_node",
        "target_node",
        "euclidean_dist_m",
        "euclidean_nodes",
        "euclidean_time_ms",
        "shortest_path_cost_s"
    ]
    
    # Dynamically add columns for each landmark configuration
    for L in landmark_counts:
        csv_header.extend([f"alt_L{L}_nodes", f"alt_L{L}_time_ms"])

    all_raw_rows = []
    global_trial_id = 1

    for t in targets:
        name, coords, radius_m = t["name"], t["coords"], t["radius_m"]
        print(f"\n=======================================================")
        print(f"PROCESSING TIER: {name}")
        print(f"=======================================================")
        
        # 1. Fetch & project network
        G = ox.graph_from_point(coords, dist=radius_m, network_type="drive")
        G_proj = ox.project_graph(G)
        G_proj = ox.add_edge_speeds(G_proj)
        G_proj = ox.add_edge_travel_times(G_proj)
        G_scc = ox.truncate.largest_component(G_proj, strongly=True)
        
        # Max speed calculation
        speeds_kph = [d.get("speed_kph", 50.0) for u, v, k, d in G_scc.edges(keys=True, data=True)]
        v_max_mps = max(speeds_kph) * (1000.0 / 3600.0)
        
        # 2. Select Maximum Required Landmarks & Precompute ALT Distances
        print(f"[{name}] Selecting max {max_landmarks} ALT landmarks and precomputing distances...")
        landmarks_full = select_farthest_landmarks(G_scc, num_landmarks=max_landmarks, seed=seed)
        d_L_to_n, d_n_to_L = precompute_alt_distances(G_scc, landmarks_full)
        
        # 3. Sample OD Pairs
        np.random.seed(seed)
        nodes = list(G_scc.nodes())
        od_pairs = []
        od_distances = []
        
        while len(od_pairs) < n_samples:
            u, v = np.random.choice(nodes, size=2, replace=False)
            x1, y1 = G_scc.nodes[u]["x"], G_scc.nodes[u]["y"]
            x2, y2 = G_scc.nodes[v]["x"], G_scc.nodes[v]["y"]
            d_euclid = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            if d_euclid >= min_dist_m:
                od_pairs.append((u, v))
                od_distances.append(d_euclid)
                
        # Structure to collect individual trial details for this tier
        tier_trial_records = [
            {
                "trial_id": global_trial_id + i,
                "tier_name": name,
                "origin_node": od_pairs[i][0],
                "target_node": od_pairs[i][1],
                "euclidean_dist_m": od_distances[i],
            }
            for i in range(n_samples)
        ]
        global_trial_id += n_samples

        # 4. Benchmark Baseline: Euclidean A*
        print(f"[{name}] Running Euclidean A* across {n_samples} pairs...")
        euclid_nodes, euclid_times = [], []
        
        for idx, (u, v) in enumerate(od_pairs):
            h_euclid = lambda curr, targ: euclidean_time_heuristic(curr, targ, G_scc, v_max_mps)
            e_n, e_t, sp_cost = a_star(G_scc, u, v, h_euclid)
            euclid_nodes.append(e_n)
            euclid_times.append(e_t)
            
            # Record individual trial metrics
            tier_trial_records[idx]["euclidean_nodes"] = e_n
            tier_trial_records[idx]["euclidean_time_ms"] = e_t
            tier_trial_records[idx]["shortest_path_cost_s"] = sp_cost
            
        mean_e_nodes = np.mean(euclid_nodes)
        mean_e_time = np.mean(euclid_times)
        
        tier_results = {
            "tier": name,
            "euclid_nodes_mean": mean_e_nodes,
            "euclid_time_ms_mean": mean_e_time,
            "alt_configs": {}
        }
        
        print(f"\n[EUCLIDEAN A* BASELINE] Mean Nodes: {mean_e_nodes:.1f} | Mean Runtime: {mean_e_time:.2f} ms\n")
        
        # 5. Sweep across active landmark counts L
        for L in landmark_counts:
            active_landmarks = landmarks_full[:L]
            print(f"[{name}] Benchmarking ALT A* with L = {L} landmarks...")
            
            alt_nodes, alt_times = [], []
            
            for idx, (u, v) in enumerate(od_pairs):
                h_alt = lambda curr, targ: alt_heuristic(curr, targ, active_landmarks, d_L_to_n, d_n_to_L)
                a_n, a_t, _ = a_star(G_scc, u, v, h_alt)
                alt_nodes.append(a_n)
                alt_times.append(a_t)
                
                # Record individual trial metrics for this L
                tier_trial_records[idx][f"alt_L{L}_nodes"] = a_n
                tier_trial_records[idx][f"alt_L{L}_time_ms"] = a_t
                
            mean_a_nodes = np.mean(alt_nodes)
            mean_a_time = np.mean(alt_times)
            node_red_pct = (1.0 - (mean_a_nodes / mean_e_nodes)) * 100.0
            speedup = mean_e_time / mean_a_time
            
            tier_results["alt_configs"][L] = {
                "nodes_mean": mean_a_nodes,
                "nodes_std": np.std(alt_nodes, ddof=1),
                "time_ms_mean": mean_a_time,
                "node_reduction_pct": node_red_pct,
                "speedup_factor": speedup
            }
            
            print(f"  -> ALT (L={L:2d}) | Mean Nodes: {mean_a_nodes:7.1f} (-{node_red_pct:5.1f}%) | Mean Time: {mean_a_time:5.2f} ms ({speedup:4.2f}x speedup)")

        all_results.append(tier_results)
        all_raw_rows.extend(tier_trial_records)

    # 6. Export all trial records to CSV
    print(f"\nExporting {len(all_raw_rows)} trial records to '{csv_filename}'...")
    with open(csv_filename, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_header)
        writer.writeheader()
        writer.writerows(all_raw_rows)
    print(f"Export complete!")

    return all_results


def plot_trial_results(trial_results):
    """Generates side-by-side comparison plots directly from the trial_results data structure."""
    plt.style.use(
        "seaborn-v0_8-whitegrid"
        if "seaborn-v0_8-whitegrid" in plt.style.available
        else "default"
    )
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

    tiers = [res["tier"].replace(" (", "\n(") for res in trial_results]
    landmark_counts = list(trial_results[0]["alt_configs"].keys())

    x = np.arange(len(tiers))
    total_bars = len(landmark_counts) + 1  # Euclidean + ALT configs
    width = 0.8 / total_bars

    # 1. Parse Baseline Euclidean Data
    e_nodes = [res["euclid_nodes_mean"] for res in trial_results]
    e_times = [res["euclid_time_ms_mean"] for res in trial_results]

    # Plot Baseline Euclidean Bars
    offset = -((total_bars - 1) / 2) * width
    ax1.bar(
        x + offset,
        e_nodes,
        width,
        label="Euclidean A*",
        color="#7f7f7f",
        edgecolor="black",
    )
    ax2.bar(
        x + offset,
        e_times,
        width,
        label="Euclidean A*",
        color="#7f7f7f",
        edgecolor="black",
    )

    colors = ["#6baed6", "#3182bd", "#08519c", "#08306b"]

    # 2. Dynamically Parse & Plot ALT Configurations
    for idx, L in enumerate(landmark_counts):
        alt_nodes = [
            res["alt_configs"][L]["nodes_mean"] for res in trial_results
        ]
        alt_times = [
            res["alt_configs"][L]["time_ms_mean"] for res in trial_results
        ]

        bar_offset = offset + (idx + 1) * width
        color = colors[idx % len(colors)]

        # Search Space Plot
        ax1.bar(
            x + bar_offset,
            alt_nodes,
            width,
            label=f"ALT (L={L})",
            color=color,
            edgecolor="black",
        )

        # Execution Time Plot
        ax2.bar(
            x + bar_offset,
            alt_times,
            width,
            label=f"ALT (L={L})",
            color=color,
            edgecolor="black",
        )

    # --- Formatting Plot 1: Nodes Expanded ---
    ax1.set_ylabel("Mean Nodes Expanded", fontsize=11, fontweight="bold")
    ax1.set_title(
        "Search Space Reduction vs. Network Tier", fontsize=12, pad=10
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(tiers, fontsize=10)
    ax1.legend(frameon=True)

    # --- Formatting Plot 2: Execution Time ---
    ax2.set_ylabel("Mean Runtime (ms)", fontsize=11, fontweight="bold")
    ax2.set_title("Execution Time Overhead Trade-off", fontsize=12, pad=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(tiers, fontsize=10)
    ax2.legend(frameon=True)

    plt.tight_layout()
    plt.savefig("EE_AStar_ALT_Benchmark_Results.png", dpi=300)
    plt.show()


# Test targets
targets = [
    {"name": "LOW (Manhattan Grid)", "coords": (40.7580, -73.9855), "radius_m": 1500},
    {"name": "MEDIUM (Paris Core)", "coords": (48.8566, 2.3522), "radius_m": 2500},
    {"name": "HIGH (Pittsburgh Rivers)", "coords": (40.4406, -79.9959), "radius_m": 3000},
]

# Execute across L = 4, 8, 16 landmark configurations and export trials to CSV
trial_results = run_trials(targets, n_samples=500, landmark_counts=[4, 8, 16], csv_filename="ee_individual_trials_raw.csv")

# Execution:
plot_trial_results(trial_results)
