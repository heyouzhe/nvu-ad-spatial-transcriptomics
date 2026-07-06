"""
Complete NVU two-level GNN training workflow
Project: NVU AD spatial transcriptomics
Purpose: hippocampus and cortex AD/Control classification with an auxiliary NVU-level loss
"""

# ══════════════════════════════════════════════════════════════
# 0. Imports & global configuration
# ══════════════════════════════════════════════════════════════
import os
import gc
import time
import pickle
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from scipy.spatial import KDTree
from scipy.sparse import issparse
from scipy.stats import mannwhitneyu
from sklearn.metrics import (
    roc_auc_score, classification_report, roc_curve, silhouette_score
)
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, LeaveOneOut
from sklearn.preprocessing import StandardScaler, LabelEncoder
from joblib import Parallel, delayed

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch as PyGBatch
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool
from torch.optim.lr_scheduler import CosineAnnealingLR

warnings.filterwarnings('ignore', category=RuntimeWarning)

# ── Repository-relative paths ────────────────────────────────
PROJECT_ROOT = Path(
    os.environ.get("NVU_PROJECT_ROOT", Path(__file__).resolve().parents[1])
).resolve()
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURE7_DATA_DIR = DATA_DIR / "figure7"

HIP_DIR = Path(os.environ.get("NVU_HIP_DIR", FIGURE7_DATA_DIR / "Hip"))
CTX_DIR = Path(os.environ.get("NVU_CTX_DIR", FIGURE7_DATA_DIR / "Cortex"))
GNN_DIR = Path(os.environ.get("NVU_GNN_DIR", FIGURE7_DATA_DIR))
PLOT_DIR = Path(os.environ.get("NVU_PLOT_DIR", RESULTS_DIR / "figure7"))
GNN_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB")

CTX_EXCLUDE    = set()
CELLTYPE_ORDER = ['Neuron','Astro','Micro','Endo','Pericyte','Oligo','OPC']
REGION_MAP     = {
    'CA1':0,'CA2':1,'CA3':2,'CA4':3,'DG':4,'FAS':5,'SLRM':6,
    'L1':7,'L23':8,'L456':9,'WM':10
}

DENSITY_SCALAR_NAMES = {'dist_mean', 'n_cells'}
DENSITY_SCALAR_PREFIXES = ('ratio__',)
DISEASE_COLORS = {'Control': '#4DBBD5', 'AD': '#E64B35'}
TISSUE_COLORS = {'hip': '#3C5488', 'ctx': '#F39B7F'}
REGION_FOCUS_ORDER = ['FAS', 'SLRM', 'CA1', 'CA2', 'CA3', 'CA4',
                      'DG', 'L1', 'L456', 'L23', 'WM']


def set_publication_plot_style():
    """
    Configure matplotlib PDF/SVG export for downstream editing in Illustrator or Inkscape.
    Call this before plotting; previously generated PDFs need to be regenerated.
    """
    import matplotlib as mpl
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42
    mpl.rcParams['svg.fonttype'] = 'none'
    mpl.rcParams['font.family'] = 'DejaVu Sans'
    mpl.rcParams['axes.unicode_minus'] = False


def save_publication_figure(fig, path_no_ext, dpi=300):
    """Save both PDF and SVG; SVG is usually best for post-hoc text editing."""
    set_publication_plot_style()
    fig.savefig(f'{path_no_ext}.pdf', dpi=dpi, bbox_inches='tight')
    fig.savefig(f'{path_no_ext}.svg', bbox_inches='tight')
    print(f"Saved: {path_no_ext}.pdf")
    print(f"Saved: {path_no_ext}.svg")

# ══════════════════════════════════════════════════════════════
# 1. Gene-module mapping
# ══════════════════════════════════════════════════════════════
def build_gene_module_map(hip_df, ctx_df, ab_df, tissue):
    """
    Three label classes:
      wgcna    : hdWGCNA non-grey module genes, including genes overlapping with Aβ, assigned to WGCNA labels
      abeta_only: shared Aβ genes that are not in WGCNA → 'Abeta_shared'
    """
    mod_df    = hip_df if tissue == 'hip' else ctx_df
    nongrey   = mod_df[mod_df['module'] != 'grey'].copy()
    wgcna_set = set(nongrey['gene_name'])
    ab_shared = set(
        ab_df[ab_df['intersection'] == 'CA1_SLRM_FAS']['gene'].tolist()
    )
    records = []
    for _, row in nongrey.iterrows():
        records.append({
            'gene_name': row['gene_name'],
            'module':    row['module'],
            'source':    'wgcna',
            'label':     row['module'],
        })
    for gene in sorted(ab_shared - wgcna_set):
        records.append({
            'gene_name': gene,
            'module':    'Abeta_shared',
            'source':    'abeta_only',
            'label':     'Abeta_shared',
        })
    result = pd.DataFrame(records)
    print(f"[{tissue}] labels={result['label'].nunique()}, "
          f"genes={len(result)}")
    return result


def get_full_scalar_names(gene_map_df, celltype_order, lr_names):
    """Full scalar feature names: module expression, LR intensity, cell composition, dist, and n_cells."""
    names = []
    for lbl in gene_map_df['label'].unique():
        names.append(f'{lbl}__all')
        for ct in celltype_order:
            names.append(f'{lbl}__{ct}')
    names.extend(lr_names)
    for ct in celltype_order:
        names.append(f'ratio__{ct}')
    names.extend(['dist_mean', 'n_cells'])
    return names


# ══════════════════════════════════════════════════════════════
# 2. LR feature construction
# ══════════════════════════════════════════════════════════════
def select_top_lr_pairs(lr_df, top_n=20, min_samples=3):
    stats = lr_df.groupby('pathway').agg(
        mean_val  = ('value', 'mean'),
        n_samples = ('sample', 'nunique'),
    ).reset_index()
    stats = stats[stats['n_samples'] >= min_samples]
    top   = stats.nlargest(top_n, 'mean_val')['pathway'].tolist()
    print(f"  Top LR pairs: {len(top)}")
    return top


def build_lr_feature_matrix(lr_df, top_pairs):
    lr_f = lr_df[lr_df['pathway'].isin(top_pairs)].copy()
    feat_dict = {}
    for (sample, area), grp in lr_f.groupby(['sample', 'area']):
        feat = {}
        for pair in top_pairs:
            pg = grp[grp['pathway'] == pair]
            feat[f'LR__{pair}__mean'] = float(pg['value'].mean()) if len(pg) > 0 else 0.0
            feat[f'LR__{pair}__max']  = float(pg['value'].max())  if len(pg) > 0 else 0.0
        feat_dict[(sample, area)] = feat
    return feat_dict


# ══════════════════════════════════════════════════════════════
# 3. Vectorized NVU scalar feature computation
# ══════════════════════════════════════════════════════════════
def to_dense(mat):
    if hasattr(mat, 'toarray'):
        return mat.toarray()
    if hasattr(mat, 'todense'):
        return np.asarray(mat.todense())
    return np.asarray(mat)


def compute_module_scalar_fast(unit, gene_map_df, celltype_order,
                                all_genes_idx, lr_feat_dict,
                                lr_names, sample_id, region):
    """
    NVU scalar features:
      ① module expression (WGCNA module x cell type)
      ② LR intensity (region level)
      ③ cell-composition proportions
      ④ dist_mean, n_cells
    """
    n_cells = len(unit)
    X = unit.X
    if issparse(X):
        X = X.toarray()
    X = X.astype(np.float32)

    ct_masks = np.array([
        unit.obs['celltype_unit'].values == ct
        for ct in celltype_order
    ], dtype=np.float32).T  # (n_cells, n_ct)

    feat_vals = []

    # ① module expression
    for lbl, grp in gene_map_df.groupby('label', sort=False):
        col_idx = [all_genes_idx[g] for g in grp['gene_name'].tolist()
                   if g in all_genes_idx]
        if not col_idx:
            feat_vals.extend([0.0] * (1 + len(celltype_order)))
            continue
        X_lbl = X[:, col_idx]
        feat_vals.append(float(X_lbl.mean()))
        with np.errstate(invalid='ignore', divide='ignore'):
            ct_sums   = ct_masks.T @ X_lbl
            ct_counts = ct_masks.sum(axis=0)
            ct_means  = np.where(
                ct_counts[:, None] > 0,
                ct_sums / ct_counts[:, None], 0.0
            ).mean(axis=1)
        feat_vals.extend(ct_means.tolist())

    # ② LR intensity
    lr_key = (sample_id, region)
    lr_vec = np.array(
        [lr_feat_dict[lr_key].get(n, 0.0) for n in lr_names]
        if lr_key in lr_feat_dict
        else [0.0] * len(lr_names),
        dtype=np.float32
    )
    feat_vals.extend(lr_vec.tolist())

    # ③ cell composition
    for ct_mask in ct_masks.T:
        feat_vals.append(float(ct_mask.mean()))

    # ④ dist, n_cells
    feat_vals.append(float(unit.obs['dist'].mean()))
    feat_vals.append(float(np.log1p(n_cells)))

    return np.array(feat_vals, dtype=np.float32), lr_vec


# ══════════════════════════════════════════════════════════════
# 4. Single-file processing
# ══════════════════════════════════════════════════════════════
def process_one_file(fpath, tissue, gene_map, scalar_names,
                     lr_feat_dict, lr_names,
                     celltype_order, pixel_size=0.5):
    try:
        import scanpy as sc
        adata = sc.read(str(fpath))
        feat_genes = gene_map['gene_name'].unique().tolist()
        keep = [g for g in feat_genes if g in adata.var_names]
        if not keep:
            return None
        adata = adata[:, keep].copy()

        gene_to_idx = {g: i for i, g in enumerate(adata.var_names)}
        node_gene_idx = np.array(
            [gene_to_idx[g] if g in gene_to_idx else -1 for g in feat_genes],
            dtype=np.int64
        )
        obs       = adata.obs
        sample_id = obs['sample_id'].iloc[0]
        label     = 1 if obs['group'].iloc[0] == 'AD' else 0

        is_nvu = (
            obs['unit_id'].notna() &
            (obs['unit_id'].astype(str).str.strip() != '') &
            (obs['unit_id'].astype(str) != '0') &
            obs['dist'].notna() &
            (obs['dist'] != -2147483648)
        )
        if is_nvu.sum() == 0:
            return None

        nvu_adata = adata[is_nvu]
        nvu_graphs, nvu_coords, nvu_regions, nvu_scalars = [], [], [], []

        for uid in nvu_adata.obs['unit_id'].unique():
            unit_mask = nvu_adata.obs['unit_id'] == uid
            unit      = nvu_adata[unit_mask]
            n_cells   = len(unit)
            region    = unit.obs['area_m'].mode()[0]

            # Node features
            X_raw = unit.X
            if issparse(X_raw):
                X_raw = X_raw.toarray()
            X_raw = X_raw.astype(np.float32)
            X_unit = np.zeros((n_cells, len(feat_genes)), dtype=np.float32)
            present = node_gene_idx >= 0
            X_unit[:, present] = X_raw[:, node_gene_idx[present]]

            ct_onehot = np.zeros((n_cells, len(celltype_order)), np.float32)
            for ci, ct in enumerate(celltype_order):
                ct_onehot[:, ci] = (unit.obs['celltype_unit'].values == ct)

            dist_val = unit.obs['dist'].values.astype(np.float32).reshape(-1, 1)

            # LR broadcast
            lr_key = (sample_id, region)
            lr_vec = np.array(
                [lr_feat_dict[lr_key].get(n, 0.0) for n in lr_names]
                if lr_key in lr_feat_dict
                else [0.0] * len(lr_names), dtype=np.float32
            )
            lr_broadcast = np.tile(lr_vec, (n_cells, 1))

            node_feat = np.concatenate(
                [X_unit, ct_onehot, dist_val, lr_broadcast], axis=1
            )
            x = torch.FloatTensor(node_feat)

            # Edges (100 micrometers)
            coords = unit.obs[['x', 'y']].values.astype(float)
            r_px   = 100.0 / pixel_size
            if n_cells > 1:
                pairs = list(KDTree(coords).query_pairs(r_px))
                if pairs:
                    ei = torch.LongTensor(pairs).T
                    edge_index = torch.cat([ei, ei.flip(0)], dim=1)
                else:
                    idx = torch.arange(n_cells)
                    src = idx.repeat_interleave(n_cells)
                    dst = idx.repeat(n_cells)
                    m   = src != dst
                    edge_index = torch.stack([src[m], dst[m]])
            else:
                edge_index = torch.zeros(2, 0, dtype=torch.long)

            nvu_graphs.append(Data(x=x, edge_index=edge_index))
            nvu_coords.append(coords.mean(axis=0))
            nvu_regions.append(region)

            scalar, _ = compute_module_scalar_fast(
                unit, gene_map, celltype_order, gene_to_idx,
                lr_feat_dict, lr_names, sample_id, region
            )
            nvu_scalars.append(scalar)

        n_nvu = len(nvu_graphs)
        if n_nvu == 0:
            return None

        return {
            'nvu_graphs':       nvu_graphs,
            'nvu_scalar_feats': np.array(nvu_scalars, dtype=np.float32),
            'nvu_scalar_names': scalar_names,
            'nvu_coords':       np.array(nvu_coords, dtype=np.float32),
            'nvu_regions':      nvu_regions,
            'n_nvu':            n_nvu,
            'label':            label,
            'sample_id':        sample_id,
            'tissue':           tissue,
            'genes_ok':         keep,
            'node_gene_names':   feat_genes,
            'node_lr_names':     lr_names,
            'node_feat_dim':    node_feat.shape[1],
        }
    except Exception as e:
        import traceback
        print(f"✗ {fpath.name}: {e}")
        traceback.print_exc()
        return None


def generate_all_results(gene_map_hip, gene_map_ctx,
                          SCALAR_NAMES_HIP, SCALAR_NAMES_CTX,
                          lr_feat_hip, lr_feat_ctx,
                          LR_NAMES_HIP, LR_NAMES_CTX,
                          n_jobs=36):
    hip_files = [f for f in sorted(Path(HIP_DIR).glob('*.h5ad'))
                 if 'Untitled' not in f.name]
    ctx_files = [f for f in sorted(Path(CTX_DIR).glob('*.h5ad'))
                 if not any(e in f.name for e in CTX_EXCLUDE)]

    tasks = (
        [(f, 'hip', gene_map_hip, SCALAR_NAMES_HIP,
          lr_feat_hip, LR_NAMES_HIP, CELLTYPE_ORDER)
         for f in hip_files] +
        [(f, 'ctx', gene_map_ctx, SCALAR_NAMES_CTX,
          lr_feat_ctx, LR_NAMES_CTX, CELLTYPE_ORDER)
         for f in ctx_files]
    )

    print(f"Total tasks={len(tasks)}, parallel jobs={n_jobs}")
    results = Parallel(n_jobs=n_jobs, backend='loky', verbose=5)(
        delayed(process_one_file)(f, t, gm, sn, lf, ln, ct)
        for f, t, gm, sn, lf, ln, ct in tasks
    )
    all_results = [r for r in results if r is not None]
    ad_r   = [r for r in all_results if r['label'] == 1]
    ctrl_r = [r for r in all_results if r['label'] == 0]
    print(f"\nTotal samples={len(all_results)} (AD={len(ad_r)}, Ctrl={len(ctrl_r)})")
    print(f"Total NVUs={sum(r['n_nvu'] for r in all_results)}")

    out = f'{GNN_DIR}/all_results_v2.pkl'
    with open(out, 'wb') as fp:
        pickle.dump(all_results, fp)
    print(f"Saved: {out}")
    return all_results


# ══════════════════════════════════════════════════════════════
# 5. Model definitions
# ══════════════════════════════════════════════════════════════
class CellGNN(nn.Module):
    """Layer 1: intra-NVU cell graph → NVU representation.

    The first gene_dim node-feature columns are gene expression; subsequent columns are celltype, dist, LR, and other covariates.
    Gene expression is passed through a separate learnable gene-gated branch to reduce reliance on density features alone.
    """
    def __init__(self, node_dim, hidden=128, gene_dim=None):
        super().__init__()
        self.node_dim = node_dim
        self.gene_dim = int(gene_dim or 0)
        self.covar_dim = node_dim - self.gene_dim

        if self.gene_dim > 0:
            self.gene_gate = nn.Parameter(torch.zeros(self.gene_dim))
            self.gene_proj = nn.Sequential(
                nn.Linear(self.gene_dim, hidden), nn.LayerNorm(hidden), nn.ReLU()
            )
        else:
            self.gene_gate = None
            self.gene_proj = None

        if self.covar_dim > 0:
            self.covar_proj = nn.Sequential(
                nn.Linear(self.covar_dim, hidden), nn.LayerNorm(hidden), nn.ReLU()
            )
        else:
            self.covar_proj = None

        n_branches = int(self.gene_dim > 0) + int(self.covar_dim > 0)
        self.proj = nn.Sequential(
            nn.Linear(hidden * n_branches, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU()
        )
        self.conv1 = GATConv(hidden, hidden, heads=4, concat=False, dropout=0.1)
        self.conv2 = GATConv(hidden, hidden, heads=4, concat=False, dropout=0.1)
        self.conv3 = GATConv(hidden, hidden, heads=1, concat=False)
        self.out_dim = hidden * 2

    def forward(self, x, edge_index, batch):
        branches = []
        if self.gene_dim > 0:
            gene_x = x[:, :self.gene_dim]
            gene_w = 1.0 + torch.sigmoid(self.gene_gate)
            branches.append(self.gene_proj(gene_x * gene_w))
        if self.covar_dim > 0:
            covar_x = x[:, self.gene_dim:]
            branches.append(self.covar_proj(covar_x))

        x = self.proj(torch.cat(branches, dim=-1))
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        return torch.cat([
            global_mean_pool(x, batch),
            global_max_pool(x, batch),
        ], dim=-1)

    def gene_importance(self, gene_names=None):
        if self.gene_gate is None:
            return None
        weights = torch.sigmoid(self.gene_gate).detach().cpu().numpy()
        if gene_names is None:
            gene_names = [f'gene_{i}' for i in range(len(weights))]
        return pd.DataFrame({
            'gene': gene_names,
            'weight': weights,
        }).sort_values('weight', ascending=False)


class NVUSampleGNN(nn.Module):
    """Layer 2: NVU graph → sample classification + auxiliary NVU-level classification"""
    def __init__(self, nvu_gnn_dim, scalar_dim, n_regions=12, hidden=128):
        super().__init__()
        self.proj  = nn.Sequential(
            nn.Linear(nvu_gnn_dim + scalar_dim, hidden),
            nn.LayerNorm(hidden), nn.ReLU()
        )
        self.conv1 = GATConv(hidden, hidden, heads=4, concat=False, dropout=0.1)
        self.conv2 = GATConv(hidden, hidden, heads=1, concat=False)
        self.region_emb = nn.Embedding(n_regions + 1, 16)
        # Adjust dropout in NVUSampleGNN.__init__ from 0.4 to 0.6
        self.sample_clf = nn.Sequential(
            nn.Linear(hidden * 2 + 16 + 1, 64),
            nn.ReLU(), nn.Dropout(0.6),    # ← 0.4→0.6
            nn.Linear(64, 32),
            nn.ReLU(), nn.Dropout(0.4),    # additional layer
            nn.Linear(32, 1)
        )
        self.nvu_clf = nn.Sequential(
            nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1)
        )

    def encode_nodes(self, nvu_gnn_repr, nvu_scalar,
                     nvu_edge_index):
        x      = self.proj(torch.cat([nvu_gnn_repr, nvu_scalar], dim=-1))
        x      = F.relu(self.conv1(x, nvu_edge_index))
        x_node = F.relu(self.conv2(x, nvu_edge_index))
        return x_node

    def forward(self, nvu_gnn_repr, nvu_scalar,
                nvu_edge_index, region_ids, nvu_batch):
        x_node = self.encode_nodes(nvu_gnn_repr, nvu_scalar, nvu_edge_index)
        nvu_logit = self.nvu_clf(x_node)

        g_mean = global_mean_pool(x_node, nvu_batch)
        g_max  = global_max_pool(x_node, nvu_batch)
        r_emb  = global_mean_pool(self.region_emb(region_ids), nvu_batch)
        n_nvu_feat = torch.zeros(
            nvu_batch.max().item() + 1, device=nvu_batch.device
        ).scatter_add_(
            0, nvu_batch,
            torch.ones(len(nvu_batch), device=nvu_batch.device)
        ).unsqueeze(-1) / 100.0

        sample_logit = self.sample_clf(
            torch.cat([g_mean, g_max, r_emb, n_nvu_feat], dim=-1)
        )
        return sample_logit, nvu_logit


class TwoLevelGNN(nn.Module):
    def __init__(self, cell_dim, scalar_dim, n_regions=12,
                 cell_hidden=128, nvu_hidden=128, gene_dim=None):
        super().__init__()
        self.cell_gnn = CellGNN(cell_dim, cell_hidden, gene_dim=gene_dim)
        self.nvu_gnn  = NVUSampleGNN(
            self.cell_gnn.out_dim, scalar_dim, n_regions, nvu_hidden
        )

    def forward(self, cell_batch, nvu_scalar,
                nvu_edge_index, region_ids, nvu_batch):
        nvu_repr = self.cell_gnn(
            cell_batch.x, cell_batch.edge_index, cell_batch.batch
        )
        return self.nvu_gnn(
            nvu_repr, nvu_scalar, nvu_edge_index, region_ids, nvu_batch
        )

    def extract_nvu_latent(self, cell_batch, nvu_scalar,
                           nvu_edge_index, region_ids, nvu_batch):
        nvu_cell_repr = self.cell_gnn(
            cell_batch.x, cell_batch.edge_index, cell_batch.batch
        )
        nvu_latent = self.nvu_gnn.encode_nodes(
            nvu_cell_repr, nvu_scalar, nvu_edge_index
        )
        sample_logit, nvu_logit = self.nvu_gnn(
            nvu_cell_repr, nvu_scalar, nvu_edge_index, region_ids, nvu_batch
        )
        return nvu_cell_repr, nvu_latent, sample_logit, nvu_logit


# ══════════════════════════════════════════════════════════════
# 6. Data preparation, including NVU sampling and edge sparsification
# ══════════════════════════════════════════════════════════════
def build_sparse_graph(data_list, max_edges_per_nvu=200):
    """Randomly sample edges when there are too many, preventing kernel OOM."""
    sparse_list = []
    for g in data_list:
        if g.num_edges > max_edges_per_nvu:
            n_keep = max_edges_per_nvu // 2
            ei     = g.edge_index
            n_pairs = ei.shape[1] // 2
            perm    = torch.randperm(n_pairs)[:n_keep]
            keep_idx = torch.cat([perm, perm + n_pairs])
            keep_idx = keep_idx[keep_idx < ei.shape[1]]
            sparse_list.append(Data(x=g.x, edge_index=ei[:, keep_idx]))
        else:
            sparse_list.append(g)
    return sparse_list


def _cohens_d(x_ad, x_ctrl):
    x_ad = np.asarray(x_ad, dtype=float)
    x_ctrl = np.asarray(x_ctrl, dtype=float)
    n1, n0 = len(x_ad), len(x_ctrl)
    if n1 < 2 or n0 < 2:
        return 0.0
    v1 = np.nanvar(x_ad, ddof=1)
    v0 = np.nanvar(x_ctrl, ddof=1)
    pooled = np.sqrt(((n1 - 1) * v1 + (n0 - 1) * v0) / max(n1 + n0 - 2, 1))
    if pooled == 0 or np.isnan(pooled):
        return 0.0
    return float((np.nanmean(x_ad) - np.nanmean(x_ctrl)) / pooled)


def _safe_mwu_p(x_ad, x_ctrl):
    try:
        if len(np.unique(np.r_[x_ad, x_ctrl])) <= 1:
            return 1.0
        return float(mannwhitneyu(x_ad, x_ctrl, alternative='two-sided').pvalue)
    except Exception:
        return 1.0


def _feature_weight_from_effect(effect, strength=2.0, min_weight=1.0,
                                max_weight=5.0):
    effect = np.asarray(effect, dtype=float)
    scale = np.nanpercentile(np.abs(effect), 95)
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    weight = min_weight + strength * np.abs(effect) / scale
    return np.clip(weight, min_weight, max_weight)


def aggregate_gene_matrix(all_results, tissue):
    rows, labels, sample_ids, gene_names = [], [], [], None
    for r in all_results:
        if r['tissue'] != tissue:
            continue
        names = r.get('node_gene_names', r.get('genes_ok'))
        if names is None:
            continue
        gene_dim = len(names)
        sum_x = np.zeros(gene_dim, dtype=np.float64)
        n_cells = 0
        for g in r['nvu_graphs']:
            x = g.x[:, :gene_dim].detach().cpu().numpy()
            sum_x += x.sum(axis=0)
            n_cells += x.shape[0]
        if n_cells == 0:
            continue
        rows.append(sum_x / n_cells)
        labels.append(r['label'])
        sample_ids.append(r['sample_id'])
        gene_names = names
    return np.vstack(rows), np.array(labels), sample_ids, list(gene_names)


def aggregate_scalar_matrix(all_results, tissue, name_prefix=None):
    rows, labels, sample_ids, names = [], [], [], None
    for r in all_results:
        if r['tissue'] != tissue:
            continue
        scalar_names = np.array(r['nvu_scalar_names'])
        mask = np.ones(len(scalar_names), dtype=bool)
        if name_prefix is not None:
            mask = np.array([n.startswith(name_prefix) for n in scalar_names])
        vals = r['nvu_scalar_feats'][:, mask].mean(axis=0)
        rows.append(vals)
        labels.append(r['label'])
        sample_ids.append(r['sample_id'])
        names = scalar_names[mask].tolist()
    return np.vstack(rows), np.array(labels), sample_ids, names


def build_sensitivity_table(X, y, feature_names, feature_type):
    records = []
    y = np.asarray(y)
    for i, name in enumerate(feature_names):
        ad = X[y == 1, i]
        ctrl = X[y == 0, i]
        mean_ad = float(np.nanmean(ad)) if len(ad) else 0.0
        mean_ctrl = float(np.nanmean(ctrl)) if len(ctrl) else 0.0
        diff = mean_ad - mean_ctrl
        d = _cohens_d(ad, ctrl)
        p = _safe_mwu_p(ad, ctrl)
        records.append({
            'feature': name,
            'feature_type': feature_type,
            'mean_ad': mean_ad,
            'mean_ctrl': mean_ctrl,
            'diff_ad_minus_ctrl': diff,
            'abs_diff': abs(diff),
            'cohens_d': d,
            'abs_cohens_d': abs(d),
            'mw_pvalue': p,
        })
    df = pd.DataFrame(records)
    df['rank_score'] = (
        df['abs_cohens_d'].fillna(0) *
        (-np.log10(df['mw_pvalue'].clip(lower=1e-300)))
    )
    return df.sort_values(
        ['rank_score', 'abs_cohens_d', 'abs_diff'],
        ascending=False
    ).reset_index(drop=True)


def build_region_susceptibility_table(all_results, tissue, nvu_pred_df=None):
    records = []
    for r in all_results:
        if r['tissue'] != tissue:
            continue
        counts = Counter(r['nvu_regions'])
        total = max(sum(counts.values()), 1)
        for region, count in counts.items():
            records.append({
                'sample_id': r['sample_id'],
                'label': r['label'],
                'region': region,
                'n_nvu': count,
                'ratio': count / total,
            })
    df = pd.DataFrame(records)
    out = []
    for region, grp in df.groupby('region'):
        ad = grp.loc[grp['label'] == 1, 'ratio'].values
        ctrl = grp.loc[grp['label'] == 0, 'ratio'].values
        row = {
            'region': region,
            'mean_ratio_ad': float(np.mean(ad)) if len(ad) else 0.0,
            'mean_ratio_ctrl': float(np.mean(ctrl)) if len(ctrl) else 0.0,
            'ratio_diff_ad_minus_ctrl': (
                float(np.mean(ad) - np.mean(ctrl))
                if len(ad) and len(ctrl) else 0.0
            ),
            'ratio_cohens_d': _cohens_d(ad, ctrl),
            'ratio_mw_pvalue': _safe_mwu_p(ad, ctrl),
        }
        out.append(row)
    region_df = pd.DataFrame(out)
    if nvu_pred_df is not None and len(nvu_pred_df):
        pred_stats = nvu_pred_df.groupby('region').agg(
            mean_nvu_ad_prob=('nvu_pred_prob', 'mean'),
            n_nvu=('nvu_pred_prob', 'size'),
        ).reset_index()
        region_df = region_df.merge(pred_stats, on='region', how='left')
    region_df['susceptibility_score'] = (
        region_df['ratio_cohens_d'].abs().fillna(0) *
        (-np.log10(region_df['ratio_mw_pvalue'].clip(lower=1e-300)))
    )
    if 'mean_nvu_ad_prob' in region_df:
        region_df['susceptibility_score'] += (
            region_df['mean_nvu_ad_prob'].fillna(0) - 0.5
        ).abs()
    return region_df.sort_values(
        'susceptibility_score', ascending=False
    ).reset_index(drop=True)


def build_feature_weights(all_results, tissue, weight_strength=2.0,
                          max_weight=5.0):
    gene_X, y, _, gene_names = aggregate_gene_matrix(all_results, tissue)
    gene_df = build_sensitivity_table(gene_X, y, gene_names, 'gene')
    lr_X, y_lr, _, lr_names = aggregate_scalar_matrix(
        all_results, tissue, name_prefix='LR__'
    )
    lr_df = build_sensitivity_table(lr_X, y_lr, lr_names, 'lr_pair')

    gene_df['model_weight'] = _feature_weight_from_effect(
        gene_df['cohens_d'].values,
        strength=weight_strength,
        max_weight=max_weight
    )
    lr_df['model_weight'] = _feature_weight_from_effect(
        lr_df['cohens_d'].values,
        strength=weight_strength,
        max_weight=max_weight
    )

    gene_weights = dict(zip(gene_df['feature'], gene_df['model_weight']))
    lr_weights = dict(zip(lr_df['feature'], lr_df['model_weight']))
    return {
        'gene_table': gene_df,
        'lr_table': lr_df,
        'gene_weights': gene_weights,
        'lr_weights': lr_weights,
    }


def export_sensitivity_tables(all_results, tissue, out_dir=PLOT_DIR,
                              weight_strength=2.0, max_weight=5.0,
                              nvu_pred_path=None):
    weights = build_feature_weights(
        all_results, tissue,
        weight_strength=weight_strength,
        max_weight=max_weight
    )
    gene_path = f'{out_dir}/sensitive_genes_{tissue}.csv'
    lr_path = f'{out_dir}/sensitive_lr_pairs_{tissue}.csv'
    weights['gene_table'].to_csv(gene_path, index=False)
    weights['lr_table'].to_csv(lr_path, index=False)

    nvu_pred_df = None
    if nvu_pred_path and Path(nvu_pred_path).exists():
        nvu_pred_df = pd.read_csv(nvu_pred_path)
    region_df = build_region_susceptibility_table(
        all_results, tissue, nvu_pred_df=nvu_pred_df
    )
    region_path = f'{out_dir}/susceptible_regions_{tissue}.csv'
    region_df.to_csv(region_path, index=False)
    print(f"[{tissue}] Sensitive-gene table: {gene_path}")
    print(f"[{tissue}] Sensitive LR table: {lr_path}")
    print(f"[{tissue}] Susceptible-region table: {region_path}")
    return weights, region_df


def _normalize_sample_key(x):
    x = str(x).strip()
    x = x.replace('.NVU_subtype.h5ad', '')
    x = x.replace('.h5ad', '')
    return x


def get_default_chip_metadata():
    """Build default metadata from the user-provided chip/sample table; an external CSV can override it."""
    clinical = [
        ('AD1', 'AD', 'M', 73, 570, 8.30, 7.65, 'V',   5),
        ('AD2', 'AD', 'F', 84, 140, 6.55, 8.50, 'IV',  4),
        ('AD3', 'AD', 'F', 93, 225, 6.81, 6.45, 'V',   5),
        ('AD4', 'AD', 'M', 91, 277, 7.02, 6.41, 'III', 3),
        ('AD5', 'AD', 'F', 85, 161, 7.20, 7.44, 'V',   4),
        ('AD6', 'AD', 'F', 76, 397, 7.46, 7.99, 'IV',  3),
        ('AD7', 'AD', 'M', 91, 220, 6.76, 7.48, 'IV',  2),
        ('AD8', 'AD', 'F', 86, 240, 6.93, 7.57, 'IV',  1),
        ('Con1', 'Control', 'M', 76, 354, 6.74, 6.46, 'II', 3),
        ('Con2', 'Control', 'F', 65, 149, 6.67, 7.82, 'II', 2),
        ('Con3', 'Control', 'F', 94, 153, 6.34, 6.42, 'I',  0),
        ('Con4', 'Control', 'M', 97, 274, 7.11, 7.47, 'I',  2),
        ('Con5', 'Control', 'F', 86, 390, 6.34, 6.25, np.nan, 2),
        ('Con6', 'Control', 'M', 74, 703, 6.54, 7.89, np.nan, 0),
        ('Con7', 'Control', 'M', 88, 615, 6.64, 6.68, np.nan, 0),
        ('Con8', 'Control', 'M', 85, 229, 6.91, 6.72, 'I',  0),
    ]
    rows = []
    for subject, disease, sex, age, pmi, brain_ph, rin, braak, thal in clinical:
        for rep in [1, 2]:
            chip = f'{subject}.{rep}'
            rows.append({
                'sample_id': chip,
                'chip': chip,
                'subject_id': subject,
                'disease': disease,
                'sex': sex,
                'age_at_death': age,
                'postmortem_interval_min': pmi,
                'brain_ph': brain_ph,
                'rin': rin,
                'braak_stage': braak,
                'thal_stage': thal,
                'clinical_severity': np.nan,
            })

    hip_chip_id_map = {
        'AD1.1': 'D01574B6',
        'AD1.2': 'D01574C4',
        'AD2.1': 'AD_1',
        'AD2.2': 'AD_2',
        'AD3.1': 'C01840B1',
        'AD3.2': 'C01834C3',
        'AD4.1': 'B03421D4',
        'AD4.2': 'B03421F4',
        'AD5.1': 'D03556C4',
        'AD5.2': 'D03556D4',
        'AD6.1': 'D04303A6',
        'AD6.2': 'D04303D1',
        'AD7.1': 'D03556E4',
        'AD7.2': 'D03556E6',
        'AD8.1': 'D04305A2',
        'AD8.2': 'D04305C6',
        'Con1.1': 'D01574A6',
        'Con1.2': 'D01574B1',
        'Con2.1': 'control_1',
        'Con2.2': 'control_2',
        'Con3.1': 'D01574C2',
        'Con3.2': 'D01574C6',
        'Con4.1': 'B03421A5',
        'Con4.2': 'B03421A6',
        'Con5.1': 'D03556D6',
        'Con5.2': 'D03556E2',
        'Con6.1': 'D04305A4',
        'Con6.2': 'D04305A6',
        'Con7.1': 'D03556F4',
        'Con7.2': 'D03556F6',
        'Con8.1': 'C04595E2',
        'Con8.2': 'C04595F1',
    }
    chip_rows = []
    base_by_chip = {
        r['chip']: r for r in rows
        if str(r.get('chip', '')).startswith(('AD', 'Con'))
    }
    for chip, chip_id in hip_chip_id_map.items():
        if chip not in base_by_chip:
            continue
        row = base_by_chip[chip].copy()
        row['sample_id'] = chip_id
        row['chip'] = chip
        row['chip_id'] = chip_id
        row['tissue_hint'] = 'hip'
        chip_rows.append(row)
    rows.extend(chip_rows)

    ctx_map = [
        ('GSM8330060_B02009F6', 'AD1',  'AD',      'Moderate'),
        ('GSM8330061_B02008D2', 'AD2',  'AD',      'Moderate'),
        ('GSM8330062_C02248B5', 'AD3',  'AD',      'Moderate'),
        ('GSM8330063_A02092E1', 'AD4',  'AD',      'Severe'),
        ('GSM8330064_B02008C6', 'AD5',  'AD',      'Severe'),
        ('GSM8330065_B01806B6', 'Con1', 'Control', np.nan),
        ('GSM8330066_D02175A4', 'Con2', 'Control', np.nan),
        ('GSM8330067_B01809C2', 'AD6',  'AD',      'Moderate'),
        ('GSM8330068_B01809A4', 'Con3', 'Control', np.nan),
        ('GSM8330069_B01809A3', 'Con4', 'Control', np.nan),
        ('GSM8330070_D02175A6', 'Con5', 'Control', np.nan),
        ('GSM8330071_B01806B5', 'Con6', 'Control', np.nan),
    ]
    clinical_by_subject = {r[0]: r for r in clinical}
    for sample_id, subject, disease, severity in ctx_map:
        _, _, sex, age, pmi, brain_ph, rin, braak, thal = clinical_by_subject[subject]
        rows.append({
            'sample_id': sample_id,
            'chip': sample_id,
            'subject_id': subject,
            'disease': disease,
            'sex': sex,
            'age_at_death': age,
            'postmortem_interval_min': pmi,
            'brain_ph': brain_ph,
            'rin': rin,
            'braak_stage': braak,
            'thal_stage': thal,
            'clinical_severity': severity,
        })

    meta = pd.DataFrame(rows)
    meta['sample_key'] = meta['sample_id'].map(_normalize_sample_key)
    return meta


def merge_chip_metadata(df, metadata=None, metadata_path=None,
                        sample_col='sample_id'):
    """Merge clinical metadata into prediction tables or latent metadata by chip name or sample_id."""
    out = df.copy()
    if metadata_path is not None and Path(metadata_path).exists():
        meta = pd.read_csv(metadata_path)
    elif metadata is not None:
        meta = metadata.copy()
    else:
        meta = get_default_chip_metadata()
    if 'sample_key' not in meta.columns:
        key_col = 'sample_id' if 'sample_id' in meta.columns else 'chip'
        meta['sample_key'] = meta[key_col].map(_normalize_sample_key)
    out['sample_key'] = out[sample_col].map(_normalize_sample_key)
    out = out.merge(
        meta.drop_duplicates('sample_key'),
        on='sample_key',
        how='left',
        suffixes=('', '_meta')
    )
    if 'disease' in out.columns:
        fallback_group = pd.Series(
            np.where(out['label'].astype(int) == 1, 'AD', 'Control'),
            index=out.index
        )
        out['group'] = out['disease'].where(
            out['disease'].notna(),
            fallback_group
        )
    else:
        out['group'] = np.where(out['label'].astype(int) == 1, 'AD', 'Control')
    return out


def save_prediction_tables_with_metadata(plot_dir=PLOT_DIR,
                                         metadata_path=None):
    """Merge saved sample-level and NVU-level prediction tables with clinical metadata and save copies."""
    meta = get_default_chip_metadata()
    outputs = {}
    for kind in ['gnn_pred', 'nvu_pred', 'nvu_latent']:
        for tissue in ['hip', 'ctx']:
            if kind == 'nvu_latent':
                path = Path(GNN_DIR) / f'nvu_latent_{tissue}_meta.csv'
            else:
                path = Path(plot_dir) / f'{kind}_{tissue}.csv'
            if not path.exists():
                continue
            df = pd.read_csv(path)
            df = merge_chip_metadata(df, meta, metadata_path=metadata_path)
            out_path = path.with_name(path.stem + '_with_metadata.csv')
            df.to_csv(out_path, index=False)
            outputs[f'{kind}_{tissue}'] = str(out_path)
            print(f"Metadata-merged table saved: {out_path}")
    return outputs


def filter_all_results(all_results, exclude_sample_ids=None,
                       include_tissues=None, min_nvu=None, max_nvu=None):
    """Filter samples according to QC rules for full-cohort and QC-filtered cohort reporting."""
    exclude_sample_ids = set(exclude_sample_ids or [])
    include_tissues = set(include_tissues) if include_tissues else None
    kept, removed = [], []
    for r in all_results:
        reason = None
        if r['sample_id'] in exclude_sample_ids:
            reason = 'manual_exclude'
        elif include_tissues is not None and r['tissue'] not in include_tissues:
            reason = 'tissue_exclude'
        elif min_nvu is not None and r['n_nvu'] < min_nvu:
            reason = f'n_nvu<{min_nvu}'
        elif max_nvu is not None and r['n_nvu'] > max_nvu:
            reason = f'n_nvu>{max_nvu}'

        if reason is None:
            kept.append(r)
        else:
            removed.append({
                'sample_id': r['sample_id'],
                'tissue': r['tissue'],
                'label': r['label'],
                'n_nvu': r['n_nvu'],
                'reason': reason,
            })
    removed_df = pd.DataFrame(removed)
    print(f"QC filtering: kept={len(kept)}, excluded={len(removed)}")
    if len(removed_df):
        print(removed_df)
    return kept, removed_df


def build_sample_feature_table(all_results, top_genes_by_tissue=None,
                               top_lrs_by_tissue=None, top_n=50,
                               include_regions=True):
    """Build a sample-level feature table for UMAP/PCA visualization."""
    records = []
    region_list = sorted(set(reg for r in all_results for reg in r['nvu_regions']))
    for r in all_results:
        rec = {
            'sample_id': r['sample_id'],
            'tissue': r['tissue'],
            'label': r['label'],
            'group': 'AD' if r['label'] == 1 else 'Control',
            'n_nvu': r['n_nvu'],
        }

        gene_names = r.get('node_gene_names', r.get('genes_ok', []))
        gene_dim = len(gene_names)
        gene_top = []
        if top_genes_by_tissue and r['tissue'] in top_genes_by_tissue:
            gene_top = top_genes_by_tissue[r['tissue']][:top_n]
        if gene_top:
            gene_idx = {g: i for i, g in enumerate(gene_names)}
            gene_sum = np.zeros(gene_dim, dtype=np.float64)
            n_cells = 0
            for g in r['nvu_graphs']:
                x = g.x[:, :gene_dim].detach().cpu().numpy()
                gene_sum += x.sum(axis=0)
                n_cells += x.shape[0]
            gene_mean = gene_sum / max(n_cells, 1)
            for gene in gene_top:
                rec[f'gene__{gene}'] = (
                    gene_mean[gene_idx[gene]] if gene in gene_idx else 0.0
                )

        lr_top = []
        if top_lrs_by_tissue and r['tissue'] in top_lrs_by_tissue:
            lr_top = top_lrs_by_tissue[r['tissue']][:top_n]
        if lr_top:
            scalar_names = list(r['nvu_scalar_names'])
            scalar_idx = {n: i for i, n in enumerate(scalar_names)}
            scalar_mean = r['nvu_scalar_feats'].mean(axis=0)
            for lr in lr_top:
                rec[f'lr__{lr}'] = (
                    scalar_mean[scalar_idx[lr]] if lr in scalar_idx else 0.0
                )

        if include_regions:
            counts = Counter(r['nvu_regions'])
            total = max(sum(counts.values()), 1)
            for region in region_list:
                rec[f'region_ratio__{region}'] = counts.get(region, 0) / total

        records.append(rec)

    return pd.DataFrame(records).fillna(0)


def plot_classification_umap(all_results, hip_weights=None, ctx_weights=None,
                             out_path=None, top_n=50, random_state=42):
    """
    Visualize the classification task with UMAP: color = AD/Control and shape = hip/ctx.
    If umap-learn is not installed, automatically fall back to PCA.
    """
    top_genes, top_lrs = {}, {}
    if hip_weights is not None:
        top_genes['hip'] = hip_weights['gene_table']['feature'].head(top_n).tolist()
        top_lrs['hip'] = hip_weights['lr_table']['feature'].head(top_n).tolist()
    if ctx_weights is not None:
        top_genes['ctx'] = ctx_weights['gene_table']['feature'].head(top_n).tolist()
        top_lrs['ctx'] = ctx_weights['lr_table']['feature'].head(top_n).tolist()

    df = build_sample_feature_table(
        all_results, top_genes, top_lrs, top_n=top_n, include_regions=True
    )
    feat_cols = [c for c in df.columns
                 if c not in ['sample_id', 'tissue', 'label', 'group']]
    X = StandardScaler().fit_transform(df[feat_cols].values)
    method = 'UMAP'
    try:
        import umap
        emb = umap.UMAP(
            n_neighbors=min(8, max(2, len(df) - 1)),
            min_dist=0.25,
            random_state=random_state
        ).fit_transform(X)
    except Exception:
        method = 'PCA'
        emb = PCA(n_components=2, random_state=random_state).fit_transform(X)

    df['UMAP1'] = emb[:, 0]
    df['UMAP2'] = emb[:, 1]

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    colors = {'AD': DISEASE_COLORS['AD'], 'Control': DISEASE_COLORS['Control']}
    markers = {'hip': 'o', 'ctx': '^'}
    for tissue in sorted(df['tissue'].unique()):
        for group in ['AD', 'Control']:
            sub = df[(df['tissue'] == tissue) & (df['group'] == group)]
            if len(sub) == 0:
                continue
            ax.scatter(
                sub['UMAP1'], sub['UMAP2'],
                c=colors[group],
                marker=markers.get(tissue, 's'),
                edgecolor='black',
                linewidth=0.5,
                s=70,
                label=f'{group} / {tissue}'
            )
    ax.set_xlabel(f'{method} 1')
    ax.set_ylabel(f'{method} 2')
    ax.set_title('Sample-level classification embedding')
    ax.legend(frameon=False, fontsize=8)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    if out_path is None:
        out_path = f'{PLOT_DIR}/classification_umap_ad_control_tissue.pdf'
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    fig.savefig(str(out_path).replace('.pdf', '.svg'), bbox_inches='tight')
    df.to_csv(str(out_path).replace('.pdf', '.csv'), index=False)
    print(f"Classification UMAP saved: {out_path}")
    return df


def plot_auc_summary(pred_files, out_path=None):
    """Plot hippocampus/cortex ROC curves and AUC bar summaries in a consistent style."""
    set_publication_plot_style()
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.6))
    auc_records = []
    colors = ['#0072B2', '#D55E00', '#009E73', '#CC79A7']
    for i, (label, path) in enumerate(pred_files.items()):
        if not Path(path).exists():
            continue
        df = pd.read_csv(path)
        y = df['label'].astype(int).values
        p = df['pred_prob'].astype(float).values
        if len(np.unique(y)) < 2:
            continue
        auc_val = roc_auc_score(y, p)
        fpr, tpr, _ = roc_curve(y, p)
        axes[0].plot(
            fpr, tpr, lw=2, color=colors[i % len(colors)],
            label=f'{label} AUC={auc_val:.3f}'
        )
        auc_records.append({'task': label, 'auc': auc_val})
    axes[0].plot([0, 1], [0, 1], color='0.7', lw=1, ls='--')
    axes[0].set_xlabel('False positive rate')
    axes[0].set_ylabel('True positive rate')
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].spines[['top', 'right']].set_visible(False)

    auc_df = pd.DataFrame(auc_records)
    axes[1].bar(
        auc_df['task'], auc_df['auc'],
        color=colors[:len(auc_df)], edgecolor='black', linewidth=0.5
    )
    axes[1].axhline(0.5, color='0.7', ls='--', lw=1)
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel('AUC')
    axes[1].tick_params(axis='x', rotation=25)
    axes[1].spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    if out_path is None:
        out_path = f'{PLOT_DIR}/auc_summary_hip_ctx.pdf'
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    fig.savefig(str(out_path).replace('.pdf', '.svg'), bbox_inches='tight')
    auc_df.to_csv(str(out_path).replace('.pdf', '.csv'), index=False)
    print(f"AUC summary figure saved: {out_path}")
    return auc_df


def _bootstrap_auc_ci(y, p, n_boot=2000, ci=95, seed=42):
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)
    rng = np.random.default_rng(seed)
    aucs = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y[idx], p[idx]))
    if len(aucs) == 0:
        auc = roc_auc_score(y, p) if len(np.unique(y)) > 1 else np.nan
        return auc, np.nan, np.nan, np.nan
    aucs = np.asarray(aucs)
    auc = roc_auc_score(y, p) if len(np.unique(y)) > 1 else np.nan
    alpha = (100 - ci) / 2
    lo = np.percentile(aucs, alpha)
    hi = np.percentile(aucs, 100 - alpha)
    se = np.std(aucs, ddof=1)
    return auc, lo, hi, se


def plot_combined_region_auc_summary(pred_files=None, out_path=None,
                                     n_boot=2000, seed=42):
    """
    Plot ROC curves and AUC bars for HIP, CTX, and combined HIP+CTX.
    Bar-plot error bars show bootstrap 95% CIs.
    """
    pred_files = pred_files or {
        'HIP': f'{PLOT_DIR}/gnn_pred_hip.csv',
        'CTX': f'{PLOT_DIR}/gnn_pred_ctx.csv',
    }
    frames, records = [], []
    for task, path in pred_files.items():
        if not Path(path).exists():
            continue
        df = pd.read_csv(path).copy()
        df['task'] = task
        frames.append(df)
        y = df['label'].astype(int).values
        p = df['pred_prob'].astype(float).values
        if len(np.unique(y)) < 2:
            continue
        auc, lo, hi, se = _bootstrap_auc_ci(
            y, p, n_boot=n_boot, seed=seed
        )
        records.append({
            'task': task,
            'auc': auc,
            'ci_low': lo,
            'ci_high': hi,
            'bootstrap_se': se,
            'n': len(df),
            'n_ad': int(np.sum(y == 1)),
            'n_control': int(np.sum(y == 0)),
        })

    if not frames:
        raise FileNotFoundError('No usable gnn_pred_*.csv files found')

    combined = pd.concat(frames, ignore_index=True)
    y = combined['label'].astype(int).values
    p = combined['pred_prob'].astype(float).values
    if len(np.unique(y)) >= 2:
        auc, lo, hi, se = _bootstrap_auc_ci(
            y, p, n_boot=n_boot, seed=seed
        )
        records.append({
            'task': 'HIP+CTX',
            'auc': auc,
            'ci_low': lo,
            'ci_high': hi,
            'bootstrap_se': se,
            'n': len(combined),
            'n_ad': int(np.sum(y == 1)),
            'n_control': int(np.sum(y == 0)),
        })

    auc_df = pd.DataFrame(records)
    order = [t for t in ['HIP', 'CTX', 'HIP+CTX'] if t in set(auc_df['task'])]
    auc_df['task'] = pd.Categorical(auc_df['task'], categories=order, ordered=True)
    auc_df = auc_df.sort_values('task').reset_index(drop=True)

    set_publication_plot_style()
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))
    colors = {
        'HIP': '#4DBBD5',
        'CTX': '#E64B35',
        'HIP+CTX': '#00A087',
    }

    for _, row in auc_df.iterrows():
        if row['task'] == 'HIP+CTX':
            df = combined
        else:
            df = combined[combined['task'] == row['task']]
        y = df['label'].astype(int).values
        p = df['pred_prob'].astype(float).values
        fpr, tpr, _ = roc_curve(y, p)
        axes[0].plot(
            fpr, tpr, lw=2,
            color=colors[str(row['task'])],
            label=f"{row['task']} AUC={row['auc']:.3f}"
        )

    axes[0].plot([0, 1], [0, 1], color='0.7', lw=1, ls='--')
    axes[0].set_xlabel('False positive rate')
    axes[0].set_ylabel('True positive rate')
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].spines[['top', 'right']].set_visible(False)

    x = np.arange(len(auc_df))
    yerr = np.vstack([
        auc_df['auc'] - auc_df['ci_low'],
        auc_df['ci_high'] - auc_df['auc'],
    ])
    axes[1].bar(
        x,
        auc_df['auc'],
        yerr=yerr,
        capsize=4,
        color=[colors[str(t)] for t in auc_df['task']],
        edgecolor='black',
        linewidth=0.6,
        error_kw={'elinewidth': 1.2, 'capthick': 1.2}
    )
    axes[1].axhline(0.5, color='0.7', ls='--', lw=1)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel('AUC')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(auc_df['task'].astype(str), rotation=25, ha='right')
    for i, row in auc_df.iterrows():
        axes[1].text(
            i, min(row['ci_high'] + 0.04, 1.03),
            f"{row['auc']:.2f}",
            ha='center', va='bottom', fontsize=8
        )
    axes[1].spines[['top', 'right']].set_visible(False)
    fig.tight_layout()

    out_path = out_path or f'{PLOT_DIR}/auc_summary_hip_ctx_combined_ci.pdf'
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    fig.savefig(str(out_path).replace('.pdf', '.svg'), bbox_inches='tight')
    auc_df.to_csv(str(out_path).replace('.pdf', '.csv'), index=False)
    combined.to_csv(str(out_path).replace('.pdf', '_predictions.csv'), index=False)
    print(f"Combined AUC figure saved: {out_path}")
    print(f"AUC table saved: {str(out_path).replace('.pdf', '.csv')}")
    return auc_df


def _clip_prob_for_logit(p, eps=1e-4):
    p = np.asarray(p, dtype=float)
    return np.clip(p, eps, 1 - eps)


def plot_risk_embedding_from_csv(pred_hip_path=None, pred_ctx_path=None,
                                 nvu_hip_path=None, nvu_ctx_path=None,
                                 out_path=None):
    """
    Visualize the classification task in model risk space.

    x-axis: sample-level AD logit score
    y-axis: mean NVU-level AD logit score
    color: AD/Control
    shape: hip/ctx

    Compared with PCA/UMAP, this plot directly shows the disease-risk space learned by the model,
    making it better suited for the classification-results panel.
    """
    pred_hip_path = pred_hip_path or f'{PLOT_DIR}/gnn_pred_hip.csv'
    pred_ctx_path = pred_ctx_path or f'{PLOT_DIR}/gnn_pred_ctx.csv'
    nvu_hip_path = nvu_hip_path or f'{PLOT_DIR}/nvu_pred_hip.csv'
    nvu_ctx_path = nvu_ctx_path or f'{PLOT_DIR}/nvu_pred_ctx.csv'
    out_path = out_path or f'{PLOT_DIR}/classification_risk_embedding.pdf'

    pred_frames = []
    for tissue, path in [('hip', pred_hip_path), ('ctx', pred_ctx_path)]:
        if Path(path).exists():
            df = pd.read_csv(path)
            df['tissue'] = tissue
            pred_frames.append(df)
    if not pred_frames:
        raise FileNotFoundError('gnn_pred_hip.csv / gnn_pred_ctx.csv not found')
    pred = pd.concat(pred_frames, ignore_index=True)

    nvu_frames = []
    for tissue, path in [('hip', nvu_hip_path), ('ctx', nvu_ctx_path)]:
        if Path(path).exists():
            df = pd.read_csv(path)
            df['tissue'] = tissue
            nvu_frames.append(df)
    if not nvu_frames:
        raise FileNotFoundError('nvu_pred_hip.csv / nvu_pred_ctx.csv not found')
    nvu = pd.concat(nvu_frames, ignore_index=True)

    nvu['nvu_logit'] = np.log(
        _clip_prob_for_logit(nvu['nvu_pred_prob']) /
        (1 - _clip_prob_for_logit(nvu['nvu_pred_prob']))
    )
    nvu_stats = nvu.groupby(['sample_id', 'tissue']).agg(
        nvu_logit_mean=('nvu_logit', 'mean'),
        nvu_prob_high_frac=('nvu_pred_prob', lambda x: float(np.mean(x > 0.7))),
        n_nvu=('nvu_pred_prob', 'size'),
    ).reset_index()

    pred['sample_logit'] = np.log(
        _clip_prob_for_logit(pred['pred_prob']) /
        (1 - _clip_prob_for_logit(pred['pred_prob']))
    )
    pred['group'] = np.where(pred['label'].astype(int) == 1, 'AD', 'Control')
    plot_df = pred.merge(nvu_stats, on=['sample_id', 'tissue'], how='left')
    plot_df = plot_df.fillna(0)

    set_publication_plot_style()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.8, 5.0))
    colors = {'AD': DISEASE_COLORS['AD'], 'Control': DISEASE_COLORS['Control']}
    markers = {'hip': 'o', 'ctx': '^'}
    for tissue in ['hip', 'ctx']:
        for group in ['AD', 'Control']:
            sub = plot_df[
                (plot_df['tissue'] == tissue) &
                (plot_df['group'] == group)
            ]
            if len(sub) == 0:
                continue
            ax.scatter(
                sub['sample_logit'], sub['nvu_logit_mean'],
                c=colors[group],
                marker=markers.get(tissue, 's'),
                edgecolor='black',
                linewidth=0.6,
                s=85,
                label=f'{group} / {tissue}'
            )

    ax.axvline(0, color='0.7', ls='--', lw=1)
    ax.axhline(0, color='0.7', ls='--', lw=1)
    ax.set_xlabel('Sample-level AD risk score')
    ax.set_ylabel('Mean NVU-level AD risk score')
    ax.set_title('Model risk embedding')
    ax.legend(frameon=False, fontsize=8)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    fig.savefig(str(out_path).replace('.pdf', '.svg'), bbox_inches='tight')
    plot_df.to_csv(str(out_path).replace('.pdf', '.csv'), index=False)
    print(f"Model risk-space figure saved: {out_path}")
    return plot_df


def plot_tissue_split_prediction_strip(pred_hip_path=None, pred_ctx_path=None,
                                       out_path=None):
    """Show sample AD predicted probabilities by tissue, useful when PCA/UMAP separation is weak."""
    pred_hip_path = pred_hip_path or f'{PLOT_DIR}/gnn_pred_hip.csv'
    pred_ctx_path = pred_ctx_path or f'{PLOT_DIR}/gnn_pred_ctx.csv'
    out_path = out_path or f'{PLOT_DIR}/prediction_score_strip_hip_ctx.pdf'

    frames = []
    for tissue, path in [('hip', pred_hip_path), ('ctx', pred_ctx_path)]:
        if Path(path).exists():
            df = pd.read_csv(path)
            df['tissue'] = tissue
            df['group'] = np.where(df['label'].astype(int) == 1, 'AD', 'Control')
            frames.append(df)
    if not frames:
        raise FileNotFoundError('Prediction table not found')
    df = pd.concat(frames, ignore_index=True)

    set_publication_plot_style()
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.4), sharey=True)
    colors = {'AD': DISEASE_COLORS['AD'], 'Control': DISEASE_COLORS['Control']}
    rng = np.random.default_rng(42)
    for ax, tissue in zip(axes, ['hip', 'ctx']):
        sub_t = df[df['tissue'] == tissue]
        for xi, group in enumerate(['Control', 'AD']):
            sub = sub_t[sub_t['group'] == group]
            jitter = rng.normal(0, 0.045, len(sub))
            ax.scatter(
                np.full(len(sub), xi) + jitter,
                sub['pred_prob'],
                c=colors[group],
                edgecolor='black',
                linewidth=0.5,
                s=65,
                alpha=0.9,
            )
            if len(sub):
                ax.hlines(
                    sub['pred_prob'].median(), xi - 0.22, xi + 0.22,
                    color='black', lw=2
                )
        ax.axhline(0.5, color='0.7', ls='--', lw=1)
        ax.set_title(tissue.upper())
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Control', 'AD'])
        ax.set_ylim(-0.03, 1.03)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylabel('Predicted AD probability')
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    fig.savefig(str(out_path).replace('.pdf', '.svg'), bbox_inches='tight')
    df.to_csv(str(out_path).replace('.pdf', '.csv'), index=False)
    print(f"Predicted-probability distribution figure saved: {out_path}")
    return df


def aggregate_nvu_latent_to_sample(latent_paths=None, meta_paths=None,
                                   metadata_path=None):
    """Read NVU latent vectors and aggregate them into sample-level features for PCA/UMAP classification views."""
    latent_paths = latent_paths or {
        'hip': f'{GNN_DIR}/nvu_latent_hip.npy',
        'ctx': f'{GNN_DIR}/nvu_latent_ctx.npy',
    }
    meta_paths = meta_paths or {
        'hip': f'{GNN_DIR}/nvu_latent_hip_meta.csv',
        'ctx': f'{GNN_DIR}/nvu_latent_ctx_meta.csv',
    }
    feature_frames, meta_frames = [], []
    for tissue in ['hip', 'ctx']:
        if not Path(latent_paths[tissue]).exists() or not Path(meta_paths[tissue]).exists():
            continue
        X = np.load(latent_paths[tissue])
        meta = pd.read_csv(meta_paths[tissue])
        meta['tissue'] = tissue
        feat = pd.DataFrame(X, columns=[f'z{i}' for i in range(X.shape[1])])
        feat['sample_id'] = meta['sample_id'].values
        agg_mean = feat.groupby('sample_id').mean().add_prefix('mean_')
        agg_std = feat.groupby('sample_id').std().fillna(0).add_prefix('std_')
        sample_feat = pd.concat([agg_mean, agg_std], axis=1).reset_index()
        sample_meta = meta.groupby('sample_id').agg(
            tissue=('tissue', 'first'),
            label=('label', 'first'),
            group=('group', 'first'),
            sample_pred_prob=('sample_pred_prob', 'first'),
            nvu_pred_prob_mean=('nvu_pred_prob', 'mean'),
            nvu_pred_prob_std=('nvu_pred_prob', 'std'),
            n_nvu=('nvu_pred_prob', 'size'),
        ).reset_index().fillna(0)
        feature_frames.append(sample_feat)
        meta_frames.append(sample_meta)
    X_df = pd.concat(feature_frames, ignore_index=True).fillna(0)
    sample_meta = pd.concat(meta_frames, ignore_index=True)
    sample_df = sample_meta.merge(X_df, on='sample_id', how='left').fillna(0)
    return merge_chip_metadata(sample_df, metadata_path=metadata_path)


def plot_sample_latent_embedding_from_saved(latent_paths=None, meta_paths=None,
                                            metadata_path=None, out_prefix=None,
                                            method='pca', color_fields=None):
    """Use sample-level representations aggregated from NVU latents to plot classification and metadata panels."""
    sample_df = aggregate_nvu_latent_to_sample(
        latent_paths=latent_paths,
        meta_paths=meta_paths,
        metadata_path=metadata_path
    )
    feature_cols = [
        c for c in sample_df.columns
        if c.startswith('mean_z') or c.startswith('std_z')
    ]
    X = StandardScaler().fit_transform(sample_df[feature_cols].values)
    method_label = method.upper()
    if method.lower() == 'umap':
        try:
            import umap
            emb = umap.UMAP(
                n_neighbors=min(8, max(2, len(sample_df) - 1)),
                min_dist=0.25,
                random_state=42
            ).fit_transform(X)
        except Exception:
            method_label = 'PCA'
            emb = PCA(n_components=2, random_state=42).fit_transform(X)
    else:
        method_label = 'PCA'
        emb = PCA(n_components=2, random_state=42).fit_transform(X)

    sample_df[f'{method_label}1'] = emb[:, 0]
    sample_df[f'{method_label}2'] = emb[:, 1]
    xcol, ycol = f'{method_label}1', f'{method_label}2'

    if color_fields is None:
        color_fields = [
            'group', 'sex', 'age_at_death',
            'braak_stage', 'thal_stage', 'clinical_severity',
            'sample_pred_prob', 'nvu_pred_prob_mean'
        ]
    color_fields = [c for c in color_fields if c in sample_df.columns]

    import matplotlib.pyplot as plt
    markers = {'hip': 'o', 'ctx': '^'}
    ncols = 3
    nrows = int(np.ceil(len(color_fields) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.3 * ncols, 3.8 * nrows),
                             squeeze=False)

    for ax, field in zip(axes.ravel(), color_fields):
        vals = sample_df[field]
        if field == 'group':
            for tissue in ['hip', 'ctx']:
                for group in ['Control', 'AD']:
                    sub = sample_df[
                        (sample_df['tissue'] == tissue) &
                        (sample_df['group'] == group)
                    ]
                    if len(sub) == 0:
                        continue
                    ax.scatter(
                        sub[xcol], sub[ycol],
                        c=DISEASE_COLORS[group],
                        marker=markers.get(tissue, 's'),
                        edgecolor='black',
                        linewidth=0.6,
                        s=80,
                        label=f'{group}/{tissue}'
                    )
            ax.legend(frameon=False, fontsize=7)
        elif pd.api.types.is_numeric_dtype(vals):
            sc = None
            for tissue in ['hip', 'ctx']:
                sub = sample_df[sample_df['tissue'] == tissue]
                sc = ax.scatter(
                    sub[xcol], sub[ycol],
                    c=sub[field],
                    cmap='viridis',
                    marker=markers.get(tissue, 's'),
                    edgecolor='black',
                    linewidth=0.5,
                    s=75,
                )
            if sc is not None:
                fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        else:
            cats = [c for c in pd.unique(vals.dropna()) if str(c) != 'nan']
            cmap = plt.get_cmap('tab10')
            cat_colors = {c: cmap(i % 10) for i, c in enumerate(cats)}
            for tissue in ['hip', 'ctx']:
                for cat in cats:
                    sub = sample_df[
                        (sample_df['tissue'] == tissue) &
                        (sample_df[field] == cat)
                    ]
                    if len(sub) == 0:
                        continue
                    ax.scatter(
                        sub[xcol], sub[ycol],
                        c=[cat_colors[cat]],
                        marker=markers.get(tissue, 's'),
                        edgecolor='black',
                        linewidth=0.5,
                        s=75,
                        label=f'{cat}/{tissue}'
                    )
            ax.legend(frameon=False, fontsize=6)
        ax.set_title(field)
        ax.set_xlabel(f'{method_label}1')
        ax.set_ylabel(f'{method_label}2')
        ax.spines[['top', 'right']].set_visible(False)

    for ax in axes.ravel()[len(color_fields):]:
        ax.axis('off')
    fig.tight_layout()
    out_prefix = out_prefix or f'{PLOT_DIR}/sample_latent_{method_label.lower()}'
    fig_path = f'{out_prefix}_metadata_panels.pdf'
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    sample_df.to_csv(f'{out_prefix}_metadata_embedding.csv', index=False)
    print(f"Sample latent metadata panel saved: {fig_path}")
    print(f"Sample latent embedding table saved: {out_prefix}_metadata_embedding.csv")
    return sample_df


def plot_nvu_latent_embedding_from_saved(latent_paths=None, meta_paths=None,
                                         metadata_path=None, out_path=None,
                                         method='pca'):
    """NVU-level latent PCA/UMAP without downsampling; each point is one NVU."""
    latent_paths = latent_paths or {
        'hip': f'{GNN_DIR}/nvu_latent_hip.npy',
        'ctx': f'{GNN_DIR}/nvu_latent_ctx.npy',
    }
    meta_paths = meta_paths or {
        'hip': f'{GNN_DIR}/nvu_latent_hip_meta.csv',
        'ctx': f'{GNN_DIR}/nvu_latent_ctx_meta.csv',
    }
    Xs, metas = [], []
    for tissue in ['hip', 'ctx']:
        if not Path(latent_paths[tissue]).exists() or not Path(meta_paths[tissue]).exists():
            continue
        X = np.load(latent_paths[tissue])
        meta = pd.read_csv(meta_paths[tissue])
        meta['tissue'] = tissue
        Xs.append(X)
        metas.append(meta)
    X = np.vstack(Xs)
    meta = pd.concat(metas, ignore_index=True)
    meta = merge_chip_metadata(meta, metadata_path=metadata_path)
    X_scaled = StandardScaler().fit_transform(X)
    method_label = method.upper()
    if method.lower() == 'umap':
        try:
            import umap
            emb = umap.UMAP(
                n_neighbors=30, min_dist=0.25,
                random_state=42
            ).fit_transform(X_scaled)
        except Exception:
            method_label = 'PCA'
            emb = PCA(n_components=2, random_state=42).fit_transform(X_scaled)
    else:
        method_label = 'PCA'
        emb = PCA(n_components=2, random_state=42).fit_transform(X_scaled)

    meta[f'{method_label}1'] = emb[:, 0]
    meta[f'{method_label}2'] = emb[:, 1]

    set_publication_plot_style()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    markers = {'hip': 'o', 'ctx': '^'}
    for tissue in ['hip', 'ctx']:
        for group in ['Control', 'AD']:
            sub = meta[(meta['tissue'] == tissue) & (meta['group'] == group)]
            if len(sub) == 0:
                continue
            ax.scatter(
                sub[f'{method_label}1'], sub[f'{method_label}2'],
                c=DISEASE_COLORS[group],
                marker=markers.get(tissue, 's'),
                s=6,
                alpha=0.22,
                linewidth=0,
                label=f'{group}/{tissue}'
            )
    ax.set_xlabel(f'{method_label}1')
    ax.set_ylabel(f'{method_label}2')
    ax.set_title(f'NVU latent {method_label}')
    ax.legend(frameon=False, fontsize=8, markerscale=2)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    out_path = out_path or f'{PLOT_DIR}/nvu_latent_{method_label.lower()}_all_points.pdf'
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    fig.savefig(str(out_path).replace('.pdf', '.svg'), bbox_inches='tight')
    meta.to_csv(str(out_path).replace('.pdf', '.csv'), index=False)
    print(f"NVU latent figure saved: {out_path}")
    return meta


def _read_tissue_table(path, tissue):
    df = pd.read_csv(path)
    df['tissue'] = tissue
    return df


def _signed_sensitivity(df):
    if 'cohens_d' in df.columns:
        return df['cohens_d'].astype(float)
    if 'diff_ad_minus_ctrl' in df.columns:
        return df['diff_ad_minus_ctrl'].astype(float)
    if 'ratio_cohens_d' in df.columns:
        return df['ratio_cohens_d'].astype(float)
    return pd.Series(np.zeros(len(df)), index=df.index)


def plot_sensitivity_analysis_panels(plot_dir=PLOT_DIR, out_prefix=None,
                                     top_n_genes=20, top_n_lr=20,
                                     top_n_regions=12):
    """
    Plot the final sensitivity-analysis figure:
      1) sensitive genes: hippocampus/cortex comparison
      2) sensitive ligand-receptor pairs: hippocampus/cortex comparison
      3) susceptible regions: susceptibility_score x NVU count, hippocampus/cortex comparison

    Also save CSV files for figure layout.
    """
    set_publication_plot_style()
    plot_dir = Path(plot_dir)
    out_prefix = out_prefix or str(plot_dir / 'figure6_sensitivity')

    gene = pd.concat([
        _read_tissue_table(plot_dir / 'sensitive_genes_hip.csv', 'hip'),
        _read_tissue_table(plot_dir / 'sensitive_genes_ctx.csv', 'ctx'),
    ], ignore_index=True)
    lr = pd.concat([
        _read_tissue_table(plot_dir / 'sensitive_lr_pairs_hip.csv', 'hip'),
        _read_tissue_table(plot_dir / 'sensitive_lr_pairs_ctx.csv', 'ctx'),
    ], ignore_index=True)
    region = pd.concat([
        _read_tissue_table(plot_dir / 'susceptible_regions_hip.csv', 'hip'),
        _read_tissue_table(plot_dir / 'susceptible_regions_ctx.csv', 'ctx'),
    ], ignore_index=True)

    gene['direction_score'] = _signed_sensitivity(gene)
    gene['plot_score'] = gene['rank_score'].astype(float).abs()
    gene['plot_label'] = gene['feature'].astype(str)

    lr = lr[lr['feature'].astype(str).str.endswith('__max')].copy()
    lr['direction_score'] = _signed_sensitivity(lr)
    lr['plot_score'] = lr['rank_score'].astype(float).abs()
    lr['plot_label'] = (
        lr['feature'].astype(str)
        .str.replace(r'^LR__', '', regex=True)
        .str.replace(r'__max$', '', regex=True)
    )

    region = region[region['region'].isin(REGION_FOCUS_ORDER)].copy()
    if 'n_nvu' not in region.columns:
        region['n_nvu'] = np.nan
    region['n_nvu'] = region['n_nvu'].fillna(0).astype(float)
    if 'susceptibility_score' not in region.columns:
        region['susceptibility_score'] = (
            region['ratio_cohens_d'].abs().fillna(0) *
            (-np.log10(region['ratio_mw_pvalue'].clip(lower=1e-300)))
        )
    region['nvu_weighted_score'] = (
        region['susceptibility_score'].astype(float) * region['n_nvu']
    )
    region['direction_score'] = region['ratio_diff_ad_minus_ctrl'].fillna(0)
    region['plot_score'] = np.log10(
        region['nvu_weighted_score'].clip(lower=0) + 1
    )
    region['plot_label'] = region['region'].astype(str)

    gene_top = pd.Index(
        gene.sort_values('rank_score', ascending=False)
            .groupby('tissue').head(top_n_genes)['feature']
    ).unique()
    lr_top = pd.Index(
        lr.sort_values('rank_score', ascending=False)
          .groupby('tissue').head(top_n_lr)['plot_label']
    ).unique()
    region_top = pd.Index(
        region.sort_values('nvu_weighted_score', ascending=False)
              .groupby('tissue').head(top_n_regions)['plot_label']
    ).unique()

    gene_plot = gene[gene['feature'].isin(gene_top)].copy()
    lr_plot = lr[lr['plot_label'].isin(lr_top)].copy()
    region_plot = region[region['plot_label'].isin(region_top)].copy()

    gene_order = (
        gene_plot.groupby('feature')['rank_score'].max()
        .sort_values(ascending=True).index.tolist()
    )
    lr_order = (
        lr_plot.groupby('plot_label')['rank_score'].max()
        .sort_values(ascending=True).index.tolist()
    )
    region_order = (
        region_plot.groupby('plot_label')['nvu_weighted_score'].max()
        .reindex(REGION_FOCUS_ORDER)
        .dropna()
        .sort_values(ascending=True).index.tolist()
    )

    import matplotlib.pyplot as plt
    tissue_colors = TISSUE_COLORS
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 7.2))

    def dot_panel(ax, df, name_col, order, x_col, title, xlabel,
                  size_col=None):
        y_map = {name: i for i, name in enumerate(order)}
        offsets = {'hip': -0.16, 'ctx': 0.16}
        for tissue in ['hip', 'ctx']:
            sub = df[df['tissue'] == tissue]
            y = sub[name_col].map(y_map).astype(float) + offsets[tissue]
            if size_col is not None and size_col in sub.columns:
                s_raw = sub[size_col].astype(float).fillna(0)
                if s_raw.max() > s_raw.min():
                    sizes = 30 + 120 * (s_raw - s_raw.min()) / (s_raw.max() - s_raw.min())
                else:
                    sizes = np.full(len(sub), 70)
            else:
                sizes = np.full(len(sub), 70)
            ax.scatter(
                sub[x_col], y,
                s=sizes,
                color=tissue_colors[tissue],
                edgecolor='black',
                linewidth=0.4,
                alpha=0.9,
                label=tissue.upper()
            )
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(order, fontsize=8)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.spines[['top', 'right']].set_visible(False)

    dot_panel(
        axes[0], gene_plot, 'feature', gene_order,
        'plot_score',
        'Sensitive genes',
        'Absolute sensitivity score\n(|Cohen d| × -log10 p)',
        size_col='model_weight'
    )
    dot_panel(
        axes[1], lr_plot, 'plot_label', lr_order,
        'plot_score',
        'Sensitive ligand-receptor pairs',
        'Absolute sensitivity score\n(|Cohen d| × -log10 p)',
        size_col='model_weight'
    )
    dot_panel(
        axes[2], region_plot, 'plot_label', region_order,
        'plot_score',
        'Susceptible regions',
        'log10(susceptibility × NVU count + 1)',
        size_col='n_nvu'
    )
    axes[2].legend(frameon=False, loc='lower right')
    fig.tight_layout()

    pdf_path = f'{out_prefix}_panels.pdf'
    svg_path = f'{out_prefix}_panels.svg'
    fig.savefig(pdf_path, dpi=300, bbox_inches='tight')
    fig.savefig(svg_path, bbox_inches='tight')

    gene_plot.to_csv(f'{out_prefix}_genes_plot.csv', index=False)
    lr_plot.to_csv(f'{out_prefix}_lr_pairs_plot.csv', index=False)
    region_plot.to_csv(f'{out_prefix}_regions_plot.csv', index=False)
    print(f"Sensitivity-analysis PDF saved: {pdf_path}")
    print(f"Sensitivity-analysis SVG saved: {svg_path}")
    return gene_plot, lr_plot, region_plot


def _infer_gene_category_lookup(tissue, gene_map=None):
    """Add module/category labels to sensitive genes; prefer the provided gene_map, otherwise read the default module file."""
    if gene_map is not None and len(gene_map):
        gm = gene_map.copy()
    else:
        module_path = (
            Path(GNN_DIR) / 'NVU.Module.csv'
            if tissue == 'hip'
            else Path(GNN_DIR) / 'Cortex_up_NVU.Module.csv'
        )
        if not module_path.exists():
            return {}
        gm = pd.read_csv(module_path)

    gene_col = 'gene_name' if 'gene_name' in gm.columns else 'gene'
    if gene_col not in gm.columns:
        return {}
    if 'label' in gm.columns:
        cat_col = 'label'
    elif 'module' in gm.columns:
        cat_col = 'module'
    elif 'source' in gm.columns:
        cat_col = 'source'
    else:
        return {}
    return dict(zip(gm[gene_col].astype(str), gm[cat_col].astype(str)))


def _top_abs_mean(df, score_col='rank_score', top_n=30):
    if df is None or len(df) == 0 or score_col not in df.columns:
        return 0.0
    vals = df[score_col].astype(float).abs().replace([np.inf, -np.inf], np.nan)
    vals = vals.dropna().sort_values(ascending=False).head(top_n)
    return float(vals.mean()) if len(vals) else 0.0


def _transform_factor_importance(values, method='log1p'):
    values = np.asarray(values, dtype=float)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    values = np.clip(values, 0, None)
    if method is None or method == 'none':
        return values
    if method == 'sqrt':
        return np.sqrt(values)
    if method == 'log1p':
        return np.log1p(values)
    raise ValueError("importance_transform must be one of: 'log1p', 'sqrt', 'none'")


def plot_model_factor_importance(plot_dir=PLOT_DIR, out_prefix=None,
                                 all_results=None,
                                 gene_map_hip=None, gene_map_ctx=None,
                                 top_n_genes=50, top_n_lr=50,
                                 normalize_within_tissue=True,
                                 importance_transform='log1p',
                                 post_normalize_power=0.5,
                                 fixed_point_size=72):
    """
    Summarize the contribution strength of different information sources to the model/classification task.

    This is an explanatory importance plot and does not retrain the model:
      - Model gene: sensitive_genes_*.csv absolute rank_score from
      - ligand-receptor: use only LR __max and aggregate it as overall LR intensity
      - region/NVU structure: susceptibility_score x n_nvu from susceptible_regions_*.csv
      - cell composition / density: if all_results is provided, additionally compute AD/Control sensitivity from scalar
        features ratio__, dist_mean, and n_cells

    importance_raw stores raw strength; importance_transformed is used for plotting/normalization.
    By default, log1p compresses the scale so NVU-count terms do not push gene/LR terms close to 0.
    Use importance_transform='sqrt' for milder compression.
    post_normalize_power defaults to 0.5, applying a square-root transform after normalization,
    which reduces visual differences in the final plot; set it to 1.0 to disable the second compression.
    fixed_point_size controls point size. The default fixed size avoids misreading n_features as importance.
    """
    set_publication_plot_style()
    import matplotlib.pyplot as plt

    plot_dir = Path(plot_dir)
    if out_prefix is None:
        out_prefix = str(plot_dir / 'figure6_factor_importance')

    gene_maps = {
        'hip': _infer_gene_category_lookup('hip', gene_map_hip),
        'ctx': _infer_gene_category_lookup('ctx', gene_map_ctx),
    }

    records = []
    for tissue in ['hip', 'ctx']:
        gene_path = plot_dir / f'sensitive_genes_{tissue}.csv'
        if gene_path.exists():
            gene = pd.read_csv(gene_path)
            records.append({
                'tissue': tissue,
                'factor_group': 'Model gene',
                'factor_class': 'Gene programs',
                'importance_raw': _top_abs_mean(
                    gene, 'rank_score', top_n=top_n_genes
                ),
                'n_features': int(len(gene)),
            })

        lr_path = plot_dir / f'sensitive_lr_pairs_{tissue}.csv'
        if lr_path.exists():
            lr = pd.read_csv(lr_path)
            lr = lr[lr['feature'].astype(str).str.endswith('__max')].copy()
            records.append({
                'tissue': tissue,
                'factor_group': 'Ligand-receptor pairs',
                'factor_class': 'Cell-cell signaling',
                'importance_raw': _top_abs_mean(
                    lr, 'rank_score', top_n=top_n_lr
                ),
                'n_features': int(len(lr)),
            })

        region_path = plot_dir / f'susceptible_regions_{tissue}.csv'
        if region_path.exists():
            region = pd.read_csv(region_path)
            region = region[region['region'].isin(REGION_FOCUS_ORDER)].copy()
            if 'n_nvu' not in region.columns:
                region['n_nvu'] = 0
            if 'susceptibility_score' not in region.columns:
                region['susceptibility_score'] = (
                    region['ratio_cohens_d'].astype(float).abs().fillna(0) *
                    (-np.log10(
                        region['ratio_mw_pvalue'].astype(float)
                        .clip(lower=1e-300)
                    ))
                )
            region['region_nvu_weighted'] = (
                region['susceptibility_score'].astype(float).abs() *
                region['n_nvu'].fillna(0).astype(float)
            )
            records.append({
                'tissue': tissue,
                'factor_group': 'Region / NVU structure',
                'factor_class': 'NVU architecture',
                'importance_raw': _top_abs_mean(
                    region, 'region_nvu_weighted',
                    top_n=len(REGION_FOCUS_ORDER)
                ),
                'n_features': int(len(region)),
            })

        if all_results is not None:
            try:
                scalar_X, scalar_y, _, scalar_names = aggregate_scalar_matrix(
                    all_results, tissue
                )
                scalar_sens = build_sensitivity_table(
                    scalar_X, scalar_y, scalar_names, 'scalar'
                )
                comp = scalar_sens[
                    scalar_sens['feature'].astype(str).str.startswith('ratio__')
                ]
                density = scalar_sens[
                    scalar_sens['feature'].astype(str)
                    .isin(['dist_mean', 'n_cells'])
                ]
                if len(comp):
                    records.append({
                        'tissue': tissue,
                        'factor_group': 'Cell-type composition',
                        'factor_class': 'NVU architecture',
                        'importance_raw': _top_abs_mean(
                            comp, 'rank_score', top_n=len(comp)
                        ),
                        'n_features': int(len(comp)),
                    })
                if len(density):
                    records.append({
                        'tissue': tissue,
                        'factor_group': 'Density / spacing',
                        'factor_class': 'NVU architecture',
                        'importance_raw': _top_abs_mean(
                            density, 'rank_score', top_n=len(density)
                        ),
                        'n_features': int(len(density)),
                    })
            except Exception as exc:
                print(f"[{tissue}] scalar factor importance skipped: {exc}")

    df = pd.DataFrame(records)
    if len(df) == 0:
        raise ValueError('No CSV files or all_results are available for factor-importance analysis.')

    df['importance_transformed'] = _transform_factor_importance(
        df['importance_raw'].values,
        method=importance_transform
    )

    if normalize_within_tissue:
        df['importance'] = df.groupby('tissue')['importance_transformed'].transform(
            lambda x: x / max(float(np.nanmax(x)), 1e-12)
        )
    else:
        max_val = max(float(np.nanmax(df['importance_transformed'])), 1e-12)
        df['importance'] = df['importance_transformed'] / max_val
    if post_normalize_power is not None and post_normalize_power != 1:
        df['importance'] = np.power(
            df['importance'].clip(lower=0),
            float(post_normalize_power)
        )

    order = (
        df.groupby('factor_group')['importance'].max()
        .sort_values(ascending=True).index.tolist()
    )
    y_map = {name: i for i, name in enumerate(order)}
    offsets = {'hip': -0.16, 'ctx': 0.16}

    fig_h = max(4.2, 0.34 * len(order) + 3.6)
    fig, ax = plt.subplots(figsize=(6.8, fig_h))
    for tissue in ['hip', 'ctx']:
        sub = df[df['tissue'] == tissue].copy()
        if len(sub) == 0:
            continue
        ax.scatter(
            sub['importance'],
            [y_map[x] + offsets[tissue] for x in sub['factor_group']],
            s=fixed_point_size,
            color=TISSUE_COLORS[tissue],
            edgecolor='black',
            linewidth=0.45,
            alpha=0.9,
            label=tissue.upper(),
        )

    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    ax.set_xlabel(
        f'Relative factor importance '
        f'({importance_transform}, power={post_normalize_power})'
    )
    ax.set_title('Model factor importance')
    ax.set_xlim(-0.03, 1.08)
    ax.grid(axis='x', color='0.9', lw=0.8)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc='lower right')
    fig.tight_layout()

    pdf_path = f'{out_prefix}.pdf'
    svg_path = f'{out_prefix}.svg'
    csv_path = f'{out_prefix}.csv'
    fig.savefig(pdf_path, dpi=300, bbox_inches='tight')
    fig.savefig(svg_path, bbox_inches='tight')
    df.sort_values(['factor_class', 'factor_group', 'tissue']).to_csv(
        csv_path, index=False
    )
    print(f"Factor-importance figure saved: {pdf_path}")
    print(f"Factor-importance SVG saved: {svg_path}")
    print(f"Factor-importance table saved: {csv_path}")
    return df


def _oriented_auc(y, score):
    """Return direction-corrected AUC, constrained to be >= 0.5."""
    if len(np.unique(y)) < 2:
        return np.nan
    auc = roc_auc_score(y, score)
    return max(auc, 1 - auc)


def _loo_linear_auc(X, y):
    """Use LOO to estimate PC-space linear separability for small sample-level datasets."""
    y = np.asarray(y).astype(int)
    if len(np.unique(y)) < 2 or len(y) < 4:
        return np.nan
    preds, trues = [], []
    loo = LeaveOneOut()
    for tr, te in loo.split(X):
        if len(np.unique(y[tr])) < 2:
            continue
        clf = LogisticRegression(
            solver='liblinear',
            class_weight='balanced',
            random_state=42
        )
        clf.fit(X[tr], y[tr])
        preds.append(clf.predict_proba(X[te])[:, 1][0])
        trues.append(y[te][0])
    if len(np.unique(trues)) < 2:
        return np.nan
    return roc_auc_score(trues, preds)


def _group_holdout_linear_auc(X, y, groups):
    """
    Evaluate NVU-level PC-space linear separability with sample holdout.
    Avoid overly optimistic results from putting NVUs from the same sample in both train and test sets.
    """
    y = np.asarray(y).astype(int)
    groups = np.asarray(groups)
    preds, trues = [], []
    for g in np.unique(groups):
        te = groups == g
        tr = ~te
        if len(np.unique(y[tr])) < 2:
            continue
        clf = LogisticRegression(
            solver='liblinear',
            class_weight='balanced',
            random_state=42
        )
        clf.fit(X[tr], y[tr])
        preds.extend(clf.predict_proba(X[te])[:, 1].tolist())
        trues.extend(y[te].tolist())
    if len(np.unique(trues)) < 2:
        return np.nan
    return roc_auc_score(trues, preds)


def quantify_pca_separation(X, meta, label_col='label',
                            group_col='sample_id', n_components=2,
                            level='sample'):
    """Quantify AD/Control separability in PCA space."""
    y = meta[label_col].astype(int).values
    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=n_components, random_state=42)
    emb = pca.fit_transform(X_scaled)

    pc_cols = [f'PC{i+1}' for i in range(n_components)]
    emb_df = meta.copy().reset_index(drop=True)
    for i, col in enumerate(pc_cols):
        emb_df[col] = emb[:, i]

    ad = emb[y == 1, :2]
    ctrl = emb[y == 0, :2]
    ad_centroid = ad.mean(axis=0)
    ctrl_centroid = ctrl.mean(axis=0)
    centroid_distance = float(np.linalg.norm(ad_centroid - ctrl_centroid))
    within = np.r_[
        np.linalg.norm(ad - ad_centroid, axis=1),
        np.linalg.norm(ctrl - ctrl_centroid, axis=1),
    ]
    within_mean = float(np.mean(within)) if len(within) else np.nan
    fisher_ratio = centroid_distance / within_mean if within_mean else np.nan

    try:
        sil_disease = float(silhouette_score(emb[:, :2], y))
    except Exception:
        sil_disease = np.nan

    tissue_silhouette = np.nan
    if 'tissue' in emb_df.columns and emb_df['tissue'].nunique() > 1:
        try:
            tissue_silhouette = float(
                silhouette_score(emb[:, :2], emb_df['tissue'].astype(str))
            )
        except Exception:
            tissue_silhouette = np.nan

    linear_auc_resub = _oriented_auc(
        y,
        LogisticRegression(
            solver='liblinear',
            class_weight='balanced',
            random_state=42
        ).fit(emb[:, :2], y).predict_proba(emb[:, :2])[:, 1]
    )
    if level == 'nvu':
        linear_auc_cv = _group_holdout_linear_auc(
            emb[:, :2], y, emb_df[group_col].values
        )
    else:
        linear_auc_cv = _loo_linear_auc(emb[:, :2], y)

    metrics = {
        'level': level,
        'n': len(emb_df),
        'n_ad': int(np.sum(y == 1)),
        'n_control': int(np.sum(y == 0)),
        'pc1_var': float(pca.explained_variance_ratio_[0]),
        'pc2_var': float(pca.explained_variance_ratio_[1]),
        'pc1_auc_oriented': _oriented_auc(y, emb[:, 0]),
        'pc2_auc_oriented': _oriented_auc(y, emb[:, 1]),
        'linear_auc_resub_pc12': linear_auc_resub,
        'linear_auc_cv_pc12': linear_auc_cv,
        'silhouette_disease_pc12': sil_disease,
        'silhouette_tissue_pc12': tissue_silhouette,
        'centroid_distance_pc12': centroid_distance,
        'within_group_distance_mean_pc12': within_mean,
        'fisher_ratio_pc12': fisher_ratio,
    }
    return emb_df, pd.DataFrame([metrics])


def quantify_saved_latent_pca_separation(metadata_path=None,
                                         out_prefix=None):
    """Quantify both sample-level and NVU-level latent PCA separability."""
    sample_df = aggregate_nvu_latent_to_sample(metadata_path=metadata_path)
    sample_feature_cols = [
        c for c in sample_df.columns
        if c.startswith('mean_z') or c.startswith('std_z')
    ]
    sample_emb, sample_metrics = quantify_pca_separation(
        sample_df[sample_feature_cols].values,
        sample_df,
        level='sample'
    )

    latent_paths = {
        'hip': f'{GNN_DIR}/nvu_latent_hip.npy',
        'ctx': f'{GNN_DIR}/nvu_latent_ctx.npy',
    }
    meta_paths = {
        'hip': f'{GNN_DIR}/nvu_latent_hip_meta.csv',
        'ctx': f'{GNN_DIR}/nvu_latent_ctx_meta.csv',
    }
    Xs, metas = [], []
    for tissue in ['hip', 'ctx']:
        X = np.load(latent_paths[tissue])
        meta = pd.read_csv(meta_paths[tissue])
        meta['tissue'] = tissue
        Xs.append(X)
        metas.append(meta)
    nvu_X = np.vstack(Xs)
    nvu_meta = pd.concat(metas, ignore_index=True)
    nvu_meta = merge_chip_metadata(nvu_meta, metadata_path=metadata_path)
    nvu_emb, nvu_metrics = quantify_pca_separation(
        nvu_X,
        nvu_meta,
        level='nvu'
    )

    metrics = pd.concat([sample_metrics, nvu_metrics], ignore_index=True)
    out_prefix = out_prefix or f'{PLOT_DIR}/latent_pca_separation'
    sample_emb.to_csv(f'{out_prefix}_sample_embedding.csv', index=False)
    nvu_emb.to_csv(f'{out_prefix}_nvu_embedding.csv', index=False)
    metrics.to_csv(f'{out_prefix}_metrics.csv', index=False)
    print(f"PCA separability metrics saved: {out_prefix}_metrics.csv")
    return sample_emb, nvu_emb, metrics


def make_scalar_mask(scalar_names, use_density_scalar=False):
    mask = []
    for name in scalar_names:
        is_density = (
            name in DENSITY_SCALAR_NAMES or
            any(name.startswith(p) for p in DENSITY_SCALAR_PREFIXES)
        )
        mask.append(use_density_scalar or not is_density)
    return np.array(mask, dtype=bool)


def fit_scalar_scaler(samples, scalar_mask):
    X = np.vstack([
        s['nvu_scalar'][:, scalar_mask].cpu().numpy()
        for s in samples
    ])
    scaler = StandardScaler()
    scaler.fit(X)
    return scaler


def transform_sample_scalar(sample, scaler, scalar_mask):
    out = dict(sample)
    x = sample['nvu_scalar'][:, scalar_mask].cpu().numpy()
    out['nvu_scalar'] = torch.FloatTensor(scaler.transform(x))
    return out


def move_sample_to_device(sample, device):
    out = dict(sample)
    out['cell_batch'] = sample['cell_batch'].to(device)
    for key in ['nvu_scalar', 'nvu_ei', 'region_ids',
                'nvu_batch', 'label', 'nvu_labels']:
        out[key] = sample[key].to(device)
    return out


def prepare_sample(result,
                   sample_radius_um=800,
                   pixel_size=0.5,
                   max_nvu=60,
                   max_edges_per_nvu=200,
                   node_feature_mode='gene_lr',
                   feature_weights=None,
                   gene_input_scale=1.0,
                   lr_input_scale=1.0,
                   structure_input_scale=1.0,
                   seed=42):
    """
    result → training dictionary
    includes stratified NVU sampling (max_nvu) and edge sparsification (max_edges_per_nvu)
    """
    coords = result['nvu_coords']
    n_nvu  = result['n_nvu']
    if n_nvu < 2:
        return None

    # Stratified NVU sampling
    if n_nvu > max_nvu:
        np.random.seed(seed)
        regions    = np.array(result['nvu_regions'])
        unique_reg = np.unique(regions)
        sampled_idx = []
        quota = max(1, max_nvu // len(unique_reg))
        for reg in unique_reg:
            reg_idx = np.where(regions == reg)[0]
            k = min(quota, len(reg_idx))
            sampled_idx.extend(
                np.random.choice(reg_idx, k, replace=False).tolist()
            )
        remaining = list(set(range(n_nvu)) - set(sampled_idx))
        if len(sampled_idx) < max_nvu and remaining:
            extra = np.random.choice(
                remaining,
                min(max_nvu - len(sampled_idx), len(remaining)),
                replace=False
            )
            sampled_idx.extend(extra.tolist())
        sampled_idx = sorted(sampled_idx[:max_nvu])
        selected_indices = sampled_idx
        graphs  = [result['nvu_graphs'][i]  for i in sampled_idx]
        coords  = result['nvu_coords'][sampled_idx]
        regions = [result['nvu_regions'][i] for i in sampled_idx]
        scalar  = result['nvu_scalar_feats'][sampled_idx]
        n_nvu   = len(sampled_idx)
    else:
        selected_indices = list(range(n_nvu))
        graphs  = result['nvu_graphs']
        regions = result['nvu_regions']
        scalar  = result['nvu_scalar_feats']

    # Edge sparsification
    graphs     = build_sparse_graph(graphs, max_edges_per_nvu)
    cell_batch = PyGBatch.from_data_list(graphs)

    gene_dim = len(result.get('node_gene_names', result['genes_ok']))
    lr_dim = len(result.get('node_lr_names', []))
    if node_feature_mode in {'gene_only', 'gene_lr'}:
        x = cell_batch.x.clone()
        covar_start = gene_dim
        covar_end = x.shape[1]
        if node_feature_mode == 'gene_only':
            x[:, covar_start:covar_end] = 0.0
        elif node_feature_mode == 'gene_lr':
            lr_start = max(covar_end - lr_dim, covar_start)
            x[:, covar_start:lr_start] = 0.0
        cell_batch.x = x

    if feature_weights is not None:
        x = cell_batch.x.clone()
        gene_names = result.get('node_gene_names', result['genes_ok'])
        gene_weights = feature_weights.get('gene_weights', {})
        if gene_weights:
            gw = torch.FloatTensor([
                gene_weights.get(g, 1.0) for g in gene_names
            ]).view(1, -1)
            x[:, :gene_dim] = x[:, :gene_dim] * gw
        lr_weights = feature_weights.get('lr_weights', {})
        lr_names = result.get('node_lr_names', [])
        if lr_weights and lr_names:
            lr_dim = len(lr_names)
            lr_start = x.shape[1] - lr_dim
            lw = torch.FloatTensor([
                lr_weights.get(n, 1.0) for n in lr_names
            ]).view(1, -1)
            x[:, lr_start:] = x[:, lr_start:] * lw
        cell_batch.x = x

    x = cell_batch.x.clone()
    lr_names = result.get('node_lr_names', [])
    lr_dim = len(lr_names)
    lr_start = x.shape[1] - lr_dim if lr_dim else x.shape[1]
    if gene_input_scale != 1.0:
        x[:, :gene_dim] *= float(gene_input_scale)
    if structure_input_scale != 1.0 and lr_start > gene_dim:
        x[:, gene_dim:lr_start] *= float(structure_input_scale)
    if lr_input_scale != 1.0 and lr_dim:
        x[:, lr_start:] *= float(lr_input_scale)
    cell_batch.x = x

    scalar_names = result['nvu_scalar_names']
    if feature_weights is not None and feature_weights.get('lr_weights'):
        scalar = scalar.copy()
        lr_weights = feature_weights['lr_weights']
        for i, name in enumerate(scalar_names):
            if name in lr_weights:
                scalar[:, i] *= lr_weights[name]

    # Inter-NVU edges
    r_px  = sample_radius_um / pixel_size
    pairs = list(KDTree(coords).query_pairs(r_px))
    if pairs:
        ei = torch.LongTensor(pairs).T
        nvu_ei = torch.cat([ei, ei.flip(0)], dim=1)
    else:
        nvu_ei = torch.zeros(2, 0, dtype=torch.long)

    region_ids = torch.LongTensor([REGION_MAP.get(r, 11) for r in regions])
    nvu_labels = torch.full((n_nvu,), result['label'], dtype=torch.float)

    return {
        'cell_batch':     cell_batch,
        'nvu_scalar':     torch.FloatTensor(scalar),
        'nvu_ei':         nvu_ei,
        'region_ids':     region_ids,
        'nvu_batch':      torch.zeros(n_nvu, dtype=torch.long),
        'label':          torch.tensor([result['label']], dtype=torch.float),
        'nvu_labels':     nvu_labels,
        'sample_id':      result['sample_id'],
        'regions':        regions,
        'nvu_indices':     selected_indices,
        'n_nvu_original': result['n_nvu'],
        'node_gene_dim':   gene_dim,
        'gene_names':      result.get('node_gene_names', result['genes_ok']),
        'scalar_names':    scalar_names,
    }


# ══════════════════════════════════════════════════════════════
# 7. Training functions
# ══════════════════════════════════════════════════════════════
def _train_one_fold(model, train_s, opt, sch, n_epochs, lambda_nvu,
                    gene_l1=1e-4, min_loss_threshold=None):
    """Train one fold."""
    model.train()
    t0 = time.time()
    for ep in range(n_epochs):
        ep_loss = 0
        for idx in np.random.permutation(len(train_s)):
            s = train_s[idx]
            opt.zero_grad()
            sl, nl = model(
                s['cell_batch'], s['nvu_scalar'],
                s['nvu_ei'], s['region_ids'], s['nvu_batch']
            )
            loss = (
                F.binary_cross_entropy_with_logits(
                    sl.squeeze(), s['label'].squeeze()
                ) +
                lambda_nvu * F.binary_cross_entropy_with_logits(
                    nl.squeeze(), s['nvu_labels']
                )
            )
            if model.cell_gnn.gene_gate is not None and gene_l1 > 0:
                loss = loss + gene_l1 * torch.sigmoid(
                    model.cell_gnn.gene_gate
                ).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item()
        sch.step()
        avg_loss = ep_loss / len(train_s)
        if (
            min_loss_threshold is not None and
            avg_loss < min_loss_threshold
        ):
            print(f"    Early stop ep{ep+1} loss={avg_loss:.4f} "
                  f"< {min_loss_threshold}")
            break
        if (ep + 1) % 20 == 0:
            elapsed = time.time() - t0
            eta     = elapsed / (ep + 1) * (n_epochs - ep - 1)
            print(f"    ep{ep+1}/{n_epochs} "
                  f"loss={avg_loss:.4f} "
                  f"{elapsed:.0f}s remaining≈{eta:.0f}s")


def train_gnn(all_results, tissue='hip',
              n_folds=5, n_epochs=100,
              lambda_nvu=0.3,
              max_nvu=60, max_edges_per_nvu=200,
              node_feature_mode='gene_lr',
              use_density_scalar=False,
              gene_l1=1e-4,
              weight_decay=1e-2,
              min_loss_threshold=None,
              feature_weights=None,
              gene_input_scale=1.0,
              lr_input_scale=1.0,
              structure_input_scale=1.0):
    """
    Use GroupKFold with 5 folds for hippocampus
    Use LOSO for cortex because the sample size is small
    """
    tissue_results = [r for r in all_results if r['tissue'] == tissue]
    print(f"\n[{tissue.upper()}] Preparing samples "
          f"(max_nvu={max_nvu}, max_edges={max_edges_per_nvu})...")

    if feature_weights == 'auto':
        labels_np = np.array([r['label'] for r in tissue_results])
        groups = np.arange(len(tissue_results))
        use_loso = (len(tissue_results) <= 12)
        if use_loso:
            print("Using LOSO validation with fold-specific sensitivity weights")
            iterator = [(
                [j for j in range(len(tissue_results)) if j != i],
                [i]
            ) for i in range(len(tissue_results))]
        else:
            print(f"Using {n_folds}-fold GroupKFold validation with fold-specific sensitivity weights")
            gkf = GroupKFold(n_splits=n_folds)
            iterator = list(gkf.split(groups, labels_np, groups))

        fold_aucs, all_preds, all_trues, all_ids = [], [], [], []
        nvu_pred_records = []
        for fold_i, (tr_idx, te_idx) in enumerate(iterator):
            print(f"\n  Fold/Sample {fold_i+1}/{len(iterator)}")
            train_results = [tissue_results[i] for i in tr_idx]
            test_results = [tissue_results[i] for i in te_idx]
            fold_weights = build_feature_weights(
                train_results, tissue,
                weight_strength=2.5,
                max_weight=6.0
            )
            train_s = [
                prepare_sample(
                    r, max_nvu=max_nvu,
                    max_edges_per_nvu=max_edges_per_nvu,
                    node_feature_mode=node_feature_mode,
                    feature_weights=fold_weights,
                    gene_input_scale=gene_input_scale,
                    lr_input_scale=lr_input_scale,
                    structure_input_scale=structure_input_scale
                )
                for r in train_results
            ]
            test_s = [
                prepare_sample(
                    r, max_nvu=max_nvu,
                    max_edges_per_nvu=max_edges_per_nvu,
                    node_feature_mode=node_feature_mode,
                    feature_weights=fold_weights,
                    gene_input_scale=gene_input_scale,
                    lr_input_scale=lr_input_scale,
                    structure_input_scale=structure_input_scale
                )
                for r in test_results
            ]
            train_s = [s for s in train_s if s is not None]
            test_s = [s for s in test_s if s is not None]
            if not train_s or not test_s:
                continue

            scalar_mask = make_scalar_mask(
                train_s[0]['scalar_names'],
                use_density_scalar=use_density_scalar
            )
            cell_dim = train_s[0]['cell_batch'].x.shape[1]
            scalar_dim = int(scalar_mask.sum())
            gene_dim = train_s[0]['node_gene_dim']
            if fold_i == 0:
                print(f"node feature mode={node_feature_mode}, "
                      f"kept scalar features={scalar_mask.sum()}, "
                      f"removed density scalar features={(~scalar_mask).sum()}")
                print(f"cell_dim={cell_dim}, gene_dim={gene_dim}, "
                      f"scalar_dim={scalar_dim}")

            scaler = fit_scalar_scaler(train_s, scalar_mask)
            train_s = [
                move_sample_to_device(
                    transform_sample_scalar(s, scaler, scalar_mask), DEVICE
                )
                for s in train_s
            ]
            test_s = [
                move_sample_to_device(
                    transform_sample_scalar(s, scaler, scalar_mask), DEVICE
                )
                for s in test_s
            ]

            model = TwoLevelGNN(
                cell_dim, scalar_dim, gene_dim=gene_dim
            ).to(DEVICE)
            opt = torch.optim.AdamW(
                model.parameters(), lr=5e-4, weight_decay=weight_decay
            )
            sch = CosineAnnealingLR(opt, T_max=n_epochs, eta_min=1e-5)
            _train_one_fold(
                model, train_s, opt, sch, n_epochs, lambda_nvu, gene_l1,
                min_loss_threshold=min_loss_threshold
            )

            model.eval()
            preds, trues = [], []
            with torch.no_grad():
                for s in test_s:
                    logit, nvu_logit = model(
                        s['cell_batch'], s['nvu_scalar'],
                        s['nvu_ei'], s['region_ids'], s['nvu_batch']
                    )
                    p = torch.sigmoid(logit).item()
                    t = s['label'].item()
                    preds.append(p)
                    trues.append(t)
                    nvu_probs = torch.sigmoid(
                        nvu_logit.squeeze()
                    ).detach().cpu().numpy()
                    if np.ndim(nvu_probs) == 0:
                        nvu_probs = np.array([float(nvu_probs)])
                    for local_i, nvu_prob in enumerate(nvu_probs):
                        nvu_pred_records.append({
                            'sample_id': s['sample_id'],
                            'label': t,
                            'tissue': tissue,
                            'fold': fold_i + 1,
                            'nvu_local_index': local_i,
                            'nvu_original_index': s['nvu_indices'][local_i],
                            'region': s['regions'][local_i],
                            'nvu_pred_prob': float(nvu_prob),
                            'sample_pred_prob': p,
                        })
                    print(f"    {s['sample_id']:25s} "
                          f"prob={p:.3f} "
                          f"true={'AD' if t==1 else 'Ctrl'} "
                          f"{'✓' if (p>0.5)==bool(t) else '✗'}")

            all_preds.extend(preds)
            all_trues.extend(trues)
            all_ids.extend([s['sample_id'] for s in test_s])
            if len(set(trues)) > 1:
                auc = roc_auc_score(trues, preds)
                fold_aucs.append(auc)
                print(f"    → Fold AUC={auc:.4f}")

            del model, train_s, test_s
            if DEVICE.type == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()

        overall_auc = roc_auc_score(all_trues, all_preds)
        mean_auc = np.mean(fold_aucs) if fold_aucs else 0.0
        std_auc = np.std(fold_aucs) if fold_aucs else 0.0
        print(f"\n{'='*55}")
        print(f"[{tissue.upper()}] Overall AUC = {overall_auc:.4f}")
        print(f"[{tissue.upper()}] Fold AUC    = {mean_auc:.4f} ± {std_auc:.4f}")

        pd.DataFrame({
            'sample_id': all_ids,
            'label': all_trues,
            'pred_prob': all_preds,
            'correct': [int((p > 0.5) == bool(t))
                        for p, t in zip(all_preds, all_trues)],
            'tissue': tissue,
        }).to_csv(f'{PLOT_DIR}/gnn_pred_{tissue}.csv', index=False)
        pd.DataFrame(nvu_pred_records).to_csv(
            f'{PLOT_DIR}/nvu_pred_{tissue}.csv', index=False
        )
        print(f"Predictions saved: {PLOT_DIR}/gnn_pred_{tissue}.csv")
        print(f"NVU-level predictions saved: {PLOT_DIR}/nvu_pred_{tissue}.csv")

        final_weights = build_feature_weights(
            tissue_results, tissue,
            weight_strength=2.5,
            max_weight=6.0
        )
        final_samples = [
            prepare_sample(
                r, max_nvu=max_nvu,
                max_edges_per_nvu=max_edges_per_nvu,
                node_feature_mode=node_feature_mode,
                feature_weights=final_weights,
                gene_input_scale=gene_input_scale,
                lr_input_scale=lr_input_scale,
                structure_input_scale=structure_input_scale
            )
            for r in tissue_results
        ]
        final_samples = [s for s in final_samples if s is not None]
        return overall_auc, mean_auc, fold_aucs, final_samples

    samples = []
    for r in tissue_results:
        s = prepare_sample(r, max_nvu=max_nvu,
                           max_edges_per_nvu=max_edges_per_nvu,
                           node_feature_mode=node_feature_mode,
                           feature_weights=feature_weights,
                           gene_input_scale=gene_input_scale,
                           lr_input_scale=lr_input_scale,
                           structure_input_scale=structure_input_scale)
        if s is not None:
            samples.append(s)

    ad_n   = sum(s['label'].item() == 1 for s in samples)
    ctrl_n = sum(s['label'].item() == 0 for s in samples)
    print(f"Valid samples={len(samples)} (AD={ad_n}, Ctrl={ctrl_n})")
    if not samples:
        raise ValueError(f"[{tissue}] has no trainable samples")

    scalar_mask = make_scalar_mask(
        samples[0]['scalar_names'],
        use_density_scalar=use_density_scalar
    )
    kept_scalar_names = np.array(samples[0]['scalar_names'])[scalar_mask]
    dropped_n = int((~scalar_mask).sum())
    print(f"node feature mode={node_feature_mode}, "
          f"kept scalar features={scalar_mask.sum()}, removed density scalar features={dropped_n}")

    cell_dim   = samples[0]['cell_batch'].x.shape[1]
    scalar_dim = int(scalar_mask.sum())
    gene_dim   = samples[0]['node_gene_dim']
    print(f"cell_dim={cell_dim}, gene_dim={gene_dim}, scalar_dim={scalar_dim}")

    groups    = np.arange(len(samples))
    labels_np = np.array([s['label'].item() for s in samples])

    # Choose the validation strategy based on sample size
    use_loso = (len(samples) <= 12)
    fold_aucs, all_preds, all_trues, all_ids = [], [], [], []
    nvu_pred_records = []

    if use_loso:
        print("Using LOSO validation")
        iterator = [(
            [j for j in range(len(samples)) if j != i],
            [i]
        ) for i in range(len(samples))]
    else:
        print(f"Using {n_folds}-fold GroupKFold validation")
        gkf      = GroupKFold(n_splits=n_folds)
        iterator = list(gkf.split(groups, labels_np, groups))

    for fold_i, (tr_idx, te_idx) in enumerate(iterator):
        train_s = [samples[i] for i in tr_idx]
        test_s  = [samples[i] for i in te_idx]
        print(f"\n  Fold/Sample {fold_i+1}/{len(iterator)}")

        scaler = fit_scalar_scaler(train_s, scalar_mask)
        train_s = [
            move_sample_to_device(
                transform_sample_scalar(s, scaler, scalar_mask), DEVICE
            )
            for s in train_s
        ]
        test_s = [
            move_sample_to_device(
                transform_sample_scalar(s, scaler, scalar_mask), DEVICE
            )
            for s in test_s
        ]

        model = TwoLevelGNN(cell_dim, scalar_dim, gene_dim=gene_dim).to(DEVICE)
        opt   = torch.optim.AdamW(
            model.parameters(), lr=5e-4, weight_decay=weight_decay
        )
        sch = CosineAnnealingLR(opt, T_max=n_epochs, eta_min=1e-5)

        _train_one_fold(
            model, train_s, opt, sch, n_epochs, lambda_nvu, gene_l1,
            min_loss_threshold=min_loss_threshold
        )

        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for s in test_s:
                logit, nvu_logit = model(
                    s['cell_batch'], s['nvu_scalar'],
                    s['nvu_ei'], s['region_ids'], s['nvu_batch']
                )
                p = torch.sigmoid(logit).item()
                t = s['label'].item()
                preds.append(p)
                trues.append(t)
                nvu_probs = torch.sigmoid(nvu_logit.squeeze()).detach().cpu().numpy()
                if np.ndim(nvu_probs) == 0:
                    nvu_probs = np.array([float(nvu_probs)])
                for local_i, nvu_prob in enumerate(nvu_probs):
                    nvu_pred_records.append({
                        'sample_id': s['sample_id'],
                        'label': t,
                        'tissue': tissue,
                        'fold': fold_i + 1,
                        'nvu_local_index': local_i,
                        'nvu_original_index': s['nvu_indices'][local_i],
                        'region': s['regions'][local_i],
                        'nvu_pred_prob': float(nvu_prob),
                        'sample_pred_prob': p,
                    })
                print(f"    {s['sample_id']:25s} "
                      f"prob={p:.3f} "
                      f"true={'AD' if t==1 else 'Ctrl'} "
                      f"{'✓' if (p>0.5)==bool(t) else '✗'}")

        all_preds.extend(preds)
        all_trues.extend(trues)
        all_ids.extend([s['sample_id'] for s in test_s])

        if len(set(trues)) > 1:
            auc = roc_auc_score(trues, preds)
            fold_aucs.append(auc)
            print(f"    → Fold AUC={auc:.4f}")

        del model, train_s, test_s
        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()

    # Overall results
    overall_auc = roc_auc_score(all_trues, all_preds)
    mean_auc    = np.mean(fold_aucs) if fold_aucs else 0.0
    std_auc     = np.std(fold_aucs)  if fold_aucs else 0.0

    print(f"\n{'='*55}")
    print(f"[{tissue.upper()}] Overall AUC = {overall_auc:.4f}")
    print(f"[{tissue.upper()}] Fold AUC    = {mean_auc:.4f} ± {std_auc:.4f}")

    # Save predictions
    df_pred = pd.DataFrame({
        'sample_id': all_ids,
        'label':     all_trues,
        'pred_prob': all_preds,
        'correct':   [int((p>0.5)==bool(t))
                      for p, t in zip(all_preds, all_trues)],
        'tissue':    tissue,
    })
    df_pred.to_csv(f'{PLOT_DIR}/gnn_pred_{tissue}.csv', index=False)
    print(f"Predictions saved: {PLOT_DIR}/gnn_pred_{tissue}.csv")

    df_nvu_pred = pd.DataFrame(nvu_pred_records)
    nvu_pred_path = f'{PLOT_DIR}/nvu_pred_{tissue}.csv'
    df_nvu_pred.to_csv(nvu_pred_path, index=False)
    print(f"NVU-level predictions saved: {nvu_pred_path}")

    pd.Series(kept_scalar_names).to_csv(
        f'{PLOT_DIR}/gnn_scalar_used_{tissue}.csv',
        index=False, header=['scalar_name']
    )

    return overall_auc, mean_auc, fold_aucs, samples


# ══════════════════════════════════════════════════════════════
# 8. NVU representation extraction for downstream region classification
# ══════════════════════════════════════════════════════════════
def extract_nvu_representations(model, samples):
    """
    Extract NVU-level representations for downstream region classification.
    Return: representation matrix plus metadata.
    """
    model.eval()
    all_repr, all_regions, all_labels, all_samples = [], [], [], []

    with torch.no_grad():
        for s in samples:
            s_dev = move_sample_to_device(s, next(model.parameters()).device)
            nvu_repr = model.cell_gnn(
                s_dev['cell_batch'].x,
                s_dev['cell_batch'].edge_index,
                s_dev['cell_batch'].batch
            )
            nvu_full = torch.cat(
                [nvu_repr.cpu(), s['nvu_scalar']], dim=-1
            ).numpy()

            all_repr.append(nvu_full)
            all_regions.extend(s['regions'])
            all_labels.extend([s['label'].item()] * nvu_repr.shape[0])
            all_samples.extend([s['sample_id']] * nvu_repr.shape[0])

    return {
        'repr':   np.vstack(all_repr),
        'region': np.array(all_regions),
        'label':  np.array(all_labels),
        'sample': np.array(all_samples),
    }


def export_nvu_latent_representations(model, samples, tissue='hip',
                                      out_dir=GNN_DIR,
                                      prefix='nvu_latent',
                                      save_cell_repr=False):
    """
    Export intermediate NVU representations for PCA/UMAP.

    Outputs:
      {prefix}_{tissue}.npy      : NVU sample-level graph latent, the level-2 GNN node representation
      {prefix}_{tissue}_meta.csv : sample, region, label, and predicted probability for each NVU
      optional {prefix}_{tissue}_cellrepr.npy : NVU representation output by the level-1 CellGNN
    """
    model.eval()
    device = next(model.parameters()).device
    latent_list, cell_repr_list, meta_records = [], [], []

    with torch.no_grad():
        for s in samples:
            s_dev = move_sample_to_device(s, device)
            nvu_cell_repr, nvu_latent, sample_logit, nvu_logit = (
                model.extract_nvu_latent(
                    s_dev['cell_batch'], s_dev['nvu_scalar'],
                    s_dev['nvu_ei'], s_dev['region_ids'], s_dev['nvu_batch']
                )
            )
            latent_np = nvu_latent.detach().cpu().numpy()
            latent_list.append(latent_np)
            if save_cell_repr:
                cell_repr_list.append(nvu_cell_repr.detach().cpu().numpy())

            sample_prob = float(torch.sigmoid(sample_logit).item())
            nvu_probs = torch.sigmoid(nvu_logit.squeeze()).detach().cpu().numpy()
            if np.ndim(nvu_probs) == 0:
                nvu_probs = np.array([float(nvu_probs)])

            for i in range(latent_np.shape[0]):
                meta_records.append({
                    'sample_id': s['sample_id'],
                    'tissue': tissue,
                    'label': int(s['label'].item()),
                    'group': 'AD' if int(s['label'].item()) == 1 else 'Control',
                    'region': s['regions'][i],
                    'nvu_local_index': i,
                    'nvu_original_index': s.get('nvu_indices', list(range(latent_np.shape[0])))[i],
                    'sample_pred_prob': sample_prob,
                    'nvu_pred_prob': float(nvu_probs[i]),
                    'n_nvu_original': s.get('n_nvu_original', latent_np.shape[0]),
                })

    latent = np.vstack(latent_list)
    meta = pd.DataFrame(meta_records)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    latent_path = out_dir / f'{prefix}_{tissue}.npy'
    meta_path = out_dir / f'{prefix}_{tissue}_meta.csv'
    np.save(latent_path, latent)
    meta.to_csv(meta_path, index=False)

    print(f"NVU latent saved: {latent_path}")
    print(f"NVU latent metadata saved: {meta_path}")

    outputs = {
        'latent': latent,
        'meta': meta,
        'latent_path': str(latent_path),
        'meta_path': str(meta_path),
    }
    if save_cell_repr:
        cell_repr = np.vstack(cell_repr_list)
        cell_repr_path = out_dir / f'{prefix}_{tissue}_cellrepr.npy'
        np.save(cell_repr_path, cell_repr)
        print(f"CellGNN NVU representation saved: {cell_repr_path}")
        outputs['cell_repr'] = cell_repr
        outputs['cell_repr_path'] = str(cell_repr_path)

    return outputs


def train_region_classifier(nvu_data, tissue='hip'):
    """Train a region classifier using NVU representations."""
    from sklearn.ensemble import RandomForestClassifier

    X      = nvu_data['repr']
    region = nvu_data['region']
    sample = nvu_data['sample']

    le  = LabelEncoder()
    y   = le.fit_transform(region)
    grp = LabelEncoder().fit_transform(sample)

    gkf = GroupKFold(n_splits=min(5, len(np.unique(grp))))
    all_true, all_pred = [], []
    for tr_idx, te_idx in gkf.split(X, y, grp):
        clf = RandomForestClassifier(
            n_estimators=200, max_depth=10, n_jobs=-1, random_state=42
        )
        clf.fit(X[tr_idx], y[tr_idx])
        all_true.extend(y[te_idx])
        all_pred.extend(clf.predict(X[te_idx]))

    print(f"\n[{tissue.upper()}] Region-classification report:")
    print(classification_report(all_true, all_pred,
                                 target_names=le.classes_))
    return clf, le


# ══════════════════════════════════════════════════════════════
# 9. Full-data model training for feature extraction
# ══════════════════════════════════════════════════════════════
def train_full_model(samples, tissue='hip',
                     n_epochs=150, lambda_nvu=0.3,
                     use_density_scalar=False,
                     gene_l1=1e-4):
    """Train one model on all available data for NVU representation extraction."""
    cell_dim   = samples[0]['cell_batch'].x.shape[1]
    scalar_mask = make_scalar_mask(
        samples[0]['scalar_names'],
        use_density_scalar=use_density_scalar
    )
    scaler = fit_scalar_scaler(samples, scalar_mask)
    samples_scaled = [
        move_sample_to_device(
            transform_sample_scalar(s, scaler, scalar_mask), DEVICE
        )
        for s in samples
    ]
    scalar_dim = int(scalar_mask.sum())
    gene_dim   = samples[0]['node_gene_dim']
    model = TwoLevelGNN(cell_dim, scalar_dim, gene_dim=gene_dim).to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-2)
    sch   = CosineAnnealingLR(opt, T_max=n_epochs, eta_min=1e-5)

    print(f"\n[{tissue.upper()}] Full-data model training ({n_epochs} epochs)...")
    _train_one_fold(model, samples_scaled, opt, sch, n_epochs,
                    lambda_nvu, gene_l1)

    torch.save(model.state_dict(),
               f'{GNN_DIR}/model_{tissue}_full.pt')
    print(f"Model saved: {GNN_DIR}/model_{tissue}_full.pt")

    gene_imp = model.cell_gnn.gene_importance(samples[0]['gene_names'])
    if gene_imp is not None:
        out_csv = f'{PLOT_DIR}/gene_weight_{tissue}.csv'
        gene_imp.to_csv(out_csv, index=False)
        print(f"Gene weights saved: {out_csv}")

    return_samples = [
        transform_sample_scalar(s, scaler, scalar_mask)
        for s in samples
    ]
    pd.to_pickle({'scaler': scaler, 'scalar_mask': scalar_mask},
                 f'{GNN_DIR}/scalar_scaler_{tissue}.pkl')
    return model, return_samples


# ══════════════════════════════════════════════════════════════
# 10. Main workflow
# ══════════════════════════════════════════════════════════════
if __name__ == '__main__':

    # ── Read gene lists ─────────────────────────────────────────
    modulegens_hip    = pd.read_csv(
        GNN_DIR / 'NVU.Module.csv'
    )
    modulegens_cortex = pd.read_csv(
        GNN_DIR / 'Cortex_up_NVU.Module.csv'
    )
    Abetagenes = pd.read_csv(
        GNN_DIR / 'Abeta_associated_genes_intersections.csv'
    )

    gene_map_hip = build_gene_module_map(
        modulegens_hip, modulegens_cortex, Abetagenes, 'hip'
    )
    gene_map_ctx = build_gene_module_map(
        modulegens_hip, modulegens_cortex, Abetagenes, 'ctx'
    )

    # ── Read LR data ───────────────────────────────────────────
    lr_hip = pd.read_csv(
        GNN_DIR / 'Hip_all_Stereosite.csv'
    )
    lr_ctx = pd.read_csv(
        GNN_DIR / 'Cortex_all_Stereosite.csv'
    )
    lr_ctx = lr_ctx[~lr_ctx['sample'].isin(CTX_EXCLUDE)].copy()

    print("Hippocampus LR:")
    TOP_LR_HIP = select_top_lr_pairs(lr_hip)
    print("Cortex LR:")
    TOP_LR_CTX = select_top_lr_pairs(lr_ctx)

    lr_feat_hip = build_lr_feature_matrix(lr_hip, TOP_LR_HIP)
    lr_feat_ctx = build_lr_feature_matrix(lr_ctx, TOP_LR_CTX)

    LR_NAMES_HIP = [f'LR__{p}__{s}'
                    for p in TOP_LR_HIP for s in ['mean', 'max']]
    LR_NAMES_CTX = [f'LR__{p}__{s}'
                    for p in TOP_LR_CTX for s in ['mean', 'max']]

    SCALAR_NAMES_HIP = get_full_scalar_names(
        gene_map_hip, CELLTYPE_ORDER, LR_NAMES_HIP
    )
    SCALAR_NAMES_CTX = get_full_scalar_names(
        gene_map_ctx, CELLTYPE_ORDER, LR_NAMES_CTX
    )

    # ── Generate/read all_results ────────────────────────────────
    pkl_path = f'{GNN_DIR}/all_results_v2.pkl'
    if Path(pkl_path).exists():
        print(f"\nReading cached data: {pkl_path}")
        with open(pkl_path, 'rb') as f:
            all_results = pickle.load(f)
        if not all(
            'node_gene_names' in r and 'node_lr_names' in r
            for r in all_results
        ):
            print("Detected an old cache without node gene/LR column metadata; regenerating all_results...")
            all_results = generate_all_results(
                gene_map_hip, gene_map_ctx,
                SCALAR_NAMES_HIP, SCALAR_NAMES_CTX,
                lr_feat_hip, lr_feat_ctx,
                LR_NAMES_HIP, LR_NAMES_CTX,
                n_jobs=36
            )
    else:
        all_results = generate_all_results(
            gene_map_hip, gene_map_ctx,
            SCALAR_NAMES_HIP, SCALAR_NAMES_CTX,
            lr_feat_hip, lr_feat_ctx,
            LR_NAMES_HIP, LR_NAMES_CTX,
            n_jobs=36
        )

    hip = [r for r in all_results if r['tissue'] == 'hip']
    ctx = [r for r in all_results if r['tissue'] == 'ctx']
    print(f"\nData overview: hippocampus={len(hip)}, cortex={len(ctx)}")

    # ── GNN training ──────────────────────────────────────────────
    print("\n" + "="*55)
    print("Start training - hippocampus")
    auc_hip_all, auc_hip_fold, folds_hip, hip_samples = train_gnn(
        all_results, tissue='hip',
        n_folds=5, n_epochs=100,
        max_nvu=60, max_edges_per_nvu=200,
        node_feature_mode='gene_lr',
        use_density_scalar=False
    )

    print("\n" + "="*55)
    print("Start training - cortex（LOSO）")
    auc_ctx_all, auc_ctx_fold, folds_ctx, ctx_samples = train_gnn(
        all_results, tissue='ctx',
        n_folds=5, n_epochs=150,
        max_nvu=60, max_edges_per_nvu=200,
        node_feature_mode='gene_lr',
        use_density_scalar=False
    )

    # ── Summary ─────────────────────────────────────────────────
    print("\n" + "="*55)
    print("Final result summary")
    print(f"  hippocampus Overall AUC = {auc_hip_all:.4f}")
    print(f"  hippocampus Fold AUC    = {auc_hip_fold:.4f} ± {np.std(folds_hip):.4f}")
    print(f"  cortex Overall AUC = {auc_ctx_all:.4f}")
    print(f"  cortex Fold AUC    = {auc_ctx_fold:.4f} ± {np.std(folds_ctx):.4f}")

    # ── Full-data model training and NVU representation extraction ──────────────────────────
    print("\nTraining full-data models for NVU representation extraction...")
    model_hip, hip_samples_scaled = train_full_model(
        hip_samples, tissue='hip', n_epochs=150,
        use_density_scalar=False
    )
    model_ctx, ctx_samples_scaled = train_full_model(
        ctx_samples, tissue='ctx', n_epochs=150,
        use_density_scalar=False
    )

    # Extract NVU representations
    nvu_data_hip = extract_nvu_representations(model_hip, hip_samples_scaled)
    nvu_data_ctx = extract_nvu_representations(model_ctx, ctx_samples_scaled)

    np.save(f'{GNN_DIR}/nvu_repr_hip.npy', nvu_data_hip['repr'])
    np.save(f'{GNN_DIR}/nvu_repr_ctx.npy', nvu_data_ctx['repr'])
    for tissue, nvu_data in [('hip', nvu_data_hip), ('ctx', nvu_data_ctx)]:
        pd.DataFrame({
            'region': nvu_data['region'],
            'label':  nvu_data['label'],
            'sample': nvu_data['sample'],
        }).to_csv(f'{GNN_DIR}/nvu_meta_{tissue}.csv', index=False)

    print(f"NVU representations saved: {GNN_DIR}/nvu_repr_*.npy")

    # Region classification
    clf_hip, le_hip = train_region_classifier(nvu_data_hip, 'hip')
    clf_ctx, le_ctx = train_region_classifier(nvu_data_ctx, 'ctx')
