"""
============================================================
Ficture血管反卷积 - 内皮中心版 (简化版)
============================================================
核心理念：内皮细胞是血管的本质特征，周细胞和SMC是支持/成熟标志

评分公式: vascular_score = endo_weight * endo + support_weight * support * endo
============================================================
"""

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import digamma, gammaln
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter
import os
import gzip
import warnings
warnings.filterwarnings('ignore')

try:
    from numba import jit, prange, njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    njit = jit
    prange = range

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(x, **kwargs): return x

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

try:
    import stereo as st
    HAS_STEREOPY = True
except ImportError:
    HAS_STEREOPY = False


# ============================================================
# 默认血管参考表达谱
# ============================================================
DEFAULT_VASCULAR_REFERENCE = {
    "Endothelial": {
        "CLDN5": 10, "PECAM1": 10, "VWF": 8, "CDH5": 8, "FLT1": 7,
        "KDR": 7, "TIE1": 6, "ESAM": 6, "ERG": 5, "CD34": 5, "ICAM2": 5,
        "VEGFR2": 5, "NOS3": 4, "THBD": 4, "SELE": 3
    },
    "Pericyte": {
        "PDGFRB": 10, "RGS5": 10, "KCNJ8": 8, "ABCC9": 7, 
        "NOTCH3": 7, "DES": 6, "ANPEP": 5, "CD248": 5,
        "NG2": 4, "CSPG4": 4
    },
    "SMC": {
        "ACTA2": 10, "MYH11": 10, "TAGLN": 9, "CNN1": 8, 
        "MYLK": 7, "MYOCD": 6, "SMTN": 5, "CALD1": 5
    },
}


# ============================================================
# Numba加速函数
# ============================================================
@njit(cache=True)
def _fast_hexagon_assign(x, y, hex_centers_x, hex_centers_y, hex_radius):
    """快速六边形分配"""
    n_points = len(x)
    n_hex = len(hex_centers_x)
    assignments = np.full(n_points, -1, dtype=np.int32)
    
    for i in prange(n_points):
        min_dist = hex_radius
        min_idx = -1
        for j in range(n_hex):
            dist = np.sqrt((x[i] - hex_centers_x[j])**2 + 
                          (y[i] - hex_centers_y[j])**2)
            if dist < min_dist:
                min_dist = dist
                min_idx = j
        assignments[i] = min_idx
    
    return assignments


@njit(cache=True)
def _fast_aggregate(assignments, gene_indices, counts, n_hex, n_genes):
    """快速聚合到六边形"""
    result = np.zeros((n_hex, n_genes), dtype=np.float64)
    hex_counts = np.zeros(n_hex, dtype=np.float64)
    
    for i in range(len(assignments)):
        hex_id = assignments[i]
        if hex_id >= 0:
            result[hex_id, gene_indices[i]] += counts[i]
            hex_counts[hex_id] += counts[i]
    
    return result, hex_counts


# ============================================================
# 变分推断LDA
# ============================================================
class VariationalLDA:
    """基于变分推断的LDA模型"""
    
    def __init__(self, n_topics, alpha=None, eta=None, 
                 learning_rate=0.7, batch_size=512,
                 max_iter=100, tol=1e-4, random_state=42):
        self.n_topics = n_topics
        self.alpha = alpha if alpha is not None else np.ones(n_topics) / n_topics
        self.eta = eta
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        np.random.seed(random_state)
        self.lambda_ = None
        self.n_words = None
    
    def _init_params(self, n_words):
        self.n_words = n_words
        if self.eta is not None:
            self.lambda_ = self.eta.copy() + np.random.gamma(100., 1./100., 
                                                              (self.n_topics, n_words))
        else:
            self.lambda_ = np.random.gamma(100., 1./100., (self.n_topics, n_words))
    
    def _dirichlet_expectation(self, alpha):
        if len(alpha.shape) == 1:
            return digamma(alpha) - digamma(alpha.sum())
        return digamma(alpha) - digamma(alpha.sum(axis=1, keepdims=True))
    
    def _e_step(self, doc_word_counts):
        n_docs = doc_word_counts.shape[0]
        gamma = np.random.gamma(100., 1./100., (n_docs, self.n_topics))
        Elogbeta = self._dirichlet_expectation(self.lambda_)
        
        for _ in range(20):
            Elogtheta = self._dirichlet_expectation(gamma)
            if sparse.issparse(doc_word_counts):
                gamma_new = self.alpha + doc_word_counts.dot(np.exp(Elogbeta).T) * np.exp(Elogtheta)
            else:
                gamma_new = self.alpha + np.exp(Elogtheta) * (doc_word_counts @ np.exp(Elogbeta).T)
            
            if np.mean(np.abs(gamma_new - gamma)) < 1e-4:
                break
            gamma = gamma_new
        
        return gamma
    
    def _m_step(self, doc_word_counts, gamma, rho):
        Elogtheta = self._dirichlet_expectation(gamma)
        if sparse.issparse(doc_word_counts):
            stats = np.exp(Elogtheta).T @ doc_word_counts
        else:
            stats = np.exp(Elogtheta).T @ doc_word_counts
        
        if self.eta is not None:
            lambda_new = self.eta + stats
        else:
            lambda_new = 1.0 + stats
        
        self.lambda_ = (1 - rho) * self.lambda_ + rho * lambda_new
    
    def fit(self, X, verbose=True):
        n_docs, n_words = X.shape
        self._init_params(n_words)
        n_batches = (n_docs + self.batch_size - 1) // self.batch_size
        
        for iteration in range(self.max_iter):
            indices = np.random.permutation(n_docs)
            total_change = 0
            
            for batch_idx in range(n_batches):
                start = batch_idx * self.batch_size
                end = min(start + self.batch_size, n_docs)
                batch_data = X[indices[start:end]]
                
                gamma = self._e_step(batch_data)
                rho = (iteration * n_batches + batch_idx + 1) ** (-self.learning_rate)
                old_lambda = self.lambda_.copy()
                self._m_step(batch_data, gamma, rho)
                total_change += np.mean(np.abs(self.lambda_ - old_lambda))
            
            avg_change = total_change / n_batches
            if verbose and (iteration + 1) % 10 == 0:
                print(f"  Iteration {iteration + 1}: avg_change = {avg_change:.6f}")
            if avg_change < self.tol:
                if verbose:
                    print(f"  收敛于 iteration {iteration + 1}")
                break
        
        return self
    
    def transform(self, X):
        gamma = self._e_step(X)
        return gamma / gamma.sum(axis=1, keepdims=True)
    
    def get_topic_words(self):
        return self.lambda_ / self.lambda_.sum(axis=1, keepdims=True)


# ============================================================
# 像素级解码器
# ============================================================
class PixelLevelDecoder:
    def __init__(self, anchor_resolution=10, neighbor_radius=30,
                 n_neighbors=6, spatial_sigma=None, n_jobs=-1):
        self.anchor_resolution = anchor_resolution
        self.neighbor_radius = neighbor_radius
        self.n_neighbors = n_neighbors
        self.spatial_sigma = spatial_sigma or anchor_resolution
        self.n_jobs = n_jobs
        
    def decode_pixels(self, pixel_coords, anchor_coords, anchor_factors):
        anchor_tree = cKDTree(anchor_coords)
        distances, indices = anchor_tree.query(pixel_coords, k=self.n_neighbors, workers=self.n_jobs)
        weights = np.exp(-distances**2 / (2 * self.spatial_sigma**2))
        weights = weights / weights.sum(axis=1, keepdims=True)
        
        n_factors = anchor_factors.shape[1]
        pixel_factors = np.zeros((len(pixel_coords), n_factors))
        for i in range(self.n_neighbors):
            pixel_factors += weights[:, i:i+1] * anchor_factors[indices[:, i]]
        
        return pixel_factors


# ============================================================
# 数据读取模块
# ============================================================
class SpatialDataReader:
    """空间转录组数据读取器"""
    
    @staticmethod
    def detect_format(file_path):
        """自动检测文件格式"""
        file_path_lower = file_path.lower()
        if file_path_lower.endswith('.gef'):
            return 'gef'
        elif file_path_lower.endswith('.gem.gz'):
            return 'gem_gz'
        elif file_path_lower.endswith('.gem'):
            return 'gem'
        elif file_path_lower.endswith('.h5') or file_path_lower.endswith('.h5ad'):
            return 'h5ad'
        else:
            try:
                with open(file_path, 'rb') as f:
                    header = f.read(8)
                    if header[:4] == b'\x89HDF':
                        return 'gef'
            except:
                pass
            return 'unknown'
    
    @staticmethod
    def read_gef(gef_path, verbose=True):
        """读取GEF格式"""
        if not HAS_H5PY:
            raise ImportError("需要安装 h5py: pip install h5py")
        
        if verbose:
            print(f"📖 读取GEF: {gef_path}")
        
        with h5py.File(gef_path, 'r') as f:
            grp = f['geneExp/bin1']
            
            gene_data = grp['gene'][:]
            if 'gene' in gene_data.dtype.names:
                gene_names = [g.decode('utf-8') if isinstance(g, bytes) else str(g) 
                             for g in gene_data['gene']]
            else:
                gene_names = [g.decode('utf-8') if isinstance(g, bytes) else str(g) 
                             for g in gene_data['geneName']]
            
            gene_names = [g.strip().rstrip('\x00') for g in gene_names]
            offsets = gene_data['offset']
            counts_per_gene = gene_data['count']
            
            exp_data = grp['expression'][:]
            x = exp_data['x'].astype(np.float32)
            y = exp_data['y'].astype(np.float32)
            count = exp_data['count'].astype(np.float32)
            
            gene_indices = np.zeros(len(exp_data), dtype=np.int32)
            for i, (offset, cnt) in enumerate(zip(offsets, counts_per_gene)):
                if cnt > 0:
                    gene_indices[offset:offset+cnt] = i
        
        if verbose:
            print(f"   记录数: {len(x)}, 基因数: {len(gene_names)}")
        
        return x, y, count, gene_indices, gene_names
    
    @staticmethod
    def read_gem_gz(gem_path, verbose=True):
        """读取GEMGEM.gz format"""
        if verbose:
            print(f"📖 读取GEM: {gem_path}")
        
        if gem_path.endswith('.gz'):
            open_func = gzip.open
            mode = 'rt'
        else:
            open_func = open
            mode = 'r'
        
        with open_func(gem_path, mode) as f:
            header_line = None
            for line in f:
                if not line.startswith('#'):
                    header_line = line.strip()
                    break
        
        if header_line is None:
            raise ValueError("无法找到GEM文件的表头")
        
        columns = header_line.split('\t')
        if verbose:
            print(f"   列名: {columns}")
        
        df = pd.read_csv(gem_path, sep='\t', comment='#', 
                         compression='gzip' if gem_path.endswith('.gz') else None)
        
        if verbose:
            print(f"   原始行数: {len(df)}")
        
        gene_col = x_col = y_col = count_col = None
        
        for col in df.columns:
            col_lower = col.lower()
            if col_lower in ['geneid', 'gene', 'genename', 'gene_name']:
                gene_col = col
            elif col_lower == 'x':
                x_col = col
            elif col_lower == 'y':
                y_col = col
            elif col_lower in ['midcount', 'umicount', 'count', 'midcounts', 'umicounts', 'counts']:
                count_col = col
        
        if gene_col is None or x_col is None or y_col is None or count_col is None:
            raise ValueError(f"无法识别GEM列名。找到的列: {list(df.columns)}")
        
        x = df[x_col].values.astype(np.float32)
        y = df[y_col].values.astype(np.float32)
        count = df[count_col].values.astype(np.float32)
        genes = df[gene_col].values
        
        unique_genes = list(pd.unique(genes))
        gene_to_idx = {g: i for i, g in enumerate(unique_genes)}
        gene_indices = np.array([gene_to_idx[g] for g in genes], dtype=np.int32)
        
        if verbose:
            print(f"   记录数: {len(x)}, 基因数: {len(unique_genes)}")
        
        return x, y, count, gene_indices, unique_genes
    
    @staticmethod
    def read_with_stereopy(file_path, verbose=True):
        """使用Stereopy读取"""
        if not HAS_STEREOPY:
            raise ImportError("需要安装 stereopy: pip install stereopy")
        
        if verbose:
            print(f"📖 使用Stereopy读取: {file_path}")
        
        data = st.io.read_gem(file_path)
        
        if hasattr(data, 'position'):
            coords = data.position
        elif hasattr(data, 'cells') and hasattr(data.cells, 'position'):
            coords = data.cells.position
        else:
            raise ValueError("无法从Stereopy对象中提取坐标")
        
        if hasattr(data, 'exp_matrix'):
            exp_matrix = data.exp_matrix
        else:
            exp_matrix = data.X
        
        if sparse.issparse(exp_matrix):
            exp_matrix = exp_matrix.tocoo()
            x_list, y_list, count_list, gene_idx_list = [], [], [], []
            
            for i, j, v in zip(exp_matrix.row, exp_matrix.col, exp_matrix.data):
                x_list.append(coords[i, 0])
                y_list.append(coords[i, 1])
                count_list.append(v)
                gene_idx_list.append(j)
            
            x = np.array(x_list, dtype=np.float32)
            y = np.array(y_list, dtype=np.float32)
            count = np.array(count_list, dtype=np.float32)
            gene_indices = np.array(gene_idx_list, dtype=np.int32)
        else:
            nonzero = np.nonzero(exp_matrix)
            x = coords[nonzero[0], 0].astype(np.float32)
            y = coords[nonzero[0], 1].astype(np.float32)
            count = exp_matrix[nonzero].astype(np.float32)
            gene_indices = nonzero[1].astype(np.int32)
        
        gene_names = list(data.gene_names)
        
        if verbose:
            print(f"   转换后记录数: {len(x)}")
        
        return x, y, count, gene_indices, gene_names
    
    @classmethod
    def read(cls, file_path, verbose=True, use_stereopy=False):
        """自动读取空间转录组数据"""
        if use_stereopy:
            return cls.read_with_stereopy(file_path, verbose)
        
        fmt = cls.detect_format(file_path)
        
        if fmt == 'gef':
            return cls.read_gef(file_path, verbose)
        elif fmt in ['gem', 'gem_gz']:
            return cls.read_gem_gz(file_path, verbose)
        else:
            if HAS_STEREOPY:
                if verbose:
                    print(f"⚠️ 未知格式，尝试使用Stereopy...")
                return cls.read_with_stereopy(file_path, verbose)
            else:
                raise ValueError(f"无法识别文件格式: {file_path}")


# ============================================================
# 主处理类 - 内皮中心版
# ============================================================
class FictureVascularDeconv:
    """
    Ficture风格血管因子反卷积 - 内皮中心版
    
    评分公式: vascular_score = endo_weight * endo + support_weight * support * endo
    """
    
    def __init__(self, 
                 hex_width=50,
                 anchor_resolution=10,
                 min_count=1,
                 n_factors=3,
                 reference_profiles=None,
                 prior_strength=1.0,
                 use_variational=True,
                 pixel_level=False,
                 include_all_genes=False,
                 min_gene_expression=0,
                 use_stereopy=False,
                 # 内皮中心评分参数
                 endo_weight=0.5,
                 support_weight=0.6,
                 n_jobs=-1,
                 verbose=True):
        """
        Parameters:
        -----------
        hex_width : float
            六边形网格宽度（微米）
        min_count : int
            六边形最小UMI阈值
        endo_weight : float
            内皮细胞权重 (默认0.7)
        support_weight : float
            支持细胞权重 (默认0.3)
        """
        self.hex_width = hex_width
        self.anchor_resolution = anchor_resolution
        self.min_count = min_count
        self.n_factors = n_factors
        self.reference_profiles = reference_profiles or DEFAULT_VASCULAR_REFERENCE
        self.prior_strength = prior_strength
        self.use_variational = use_variational
        self.pixel_level = pixel_level
        self.include_all_genes = include_all_genes
        self.min_gene_expression = min_gene_expression
        self.use_stereopy = use_stereopy
        self.endo_weight = endo_weight
        self.support_weight = support_weight
        self.n_jobs = n_jobs
        self.verbose = verbose
        
        self.lda_model = None
        self.pixel_decoder = None
        self.gene_list = None
        self.factor_names = None
        
    def _log(self, msg):
        if self.verbose:
            print(msg)
    
    def _build_prior_matrix(self, gene_list):
        """构建LDA先验矩阵"""
        self.factor_names = list(self.reference_profiles.keys())
        n_factors = len(self.factor_names)
        n_genes = len(gene_list)
        gene_to_idx = {g: i for i, g in enumerate(gene_list)}
        
        eta = np.ones((n_factors, n_genes)) * 0.01
        
        for f_idx, (factor, markers) in enumerate(self.reference_profiles.items()):
            for gene, weight in markers.items():
                if gene in gene_to_idx:
                    eta[f_idx, gene_to_idx[gene]] = weight * self.prior_strength
        
        return eta
    
    def _read_data(self, file_path):
        """读取数据"""
        return SpatialDataReader.read(
            file_path, 
            verbose=self.verbose, 
            use_stereopy=self.use_stereopy
        )
    
    def _filter_genes(self, gene_names, gene_indices, counts):
        """过滤基因"""
        if self.include_all_genes:
            self._log(f"📌 使用所有基因: {len(gene_names)}")
            mask = counts >= self.min_gene_expression
            return mask, gene_indices[mask], gene_names
        
        target_genes = set()
        for markers in self.reference_profiles.values():
            target_genes.update(markers.keys())
        
        available_genes = []
        gene_mapping = {}
        for i, g in enumerate(gene_names):
            if g in target_genes:
                gene_mapping[i] = len(available_genes)
                available_genes.append(g)
        
        self._log(f"📌 目标基因: {len(available_genes)}/{len(target_genes)}")
        if len(available_genes) < len(target_genes):
            missing = target_genes - set(available_genes)
            if self.verbose and len(missing) <= 10:
                self._log(f"   缺失基因: {missing}")
        
        mask = np.array([gene_indices[i] in gene_mapping for i in range(len(gene_indices))])
        new_gene_indices = np.array([gene_mapping[gene_indices[i]] 
                                     for i in np.where(mask)[0]], dtype=np.int32)
        
        return mask, new_gene_indices, available_genes
    
    def _calc_vascular_score(self, factor_props):
        """
        计算内皮中心血管分数
        
        公式: vascular_score = endo_weight * endo + support_weight * support * endo
        
        设计理念：
        - 内皮细胞是血管的本质特征
        - 没有内皮就不是真正的血管 (support * endo 确保这一点)
        - 周细胞和SMC是血管成熟度标志，起加成作用
        """
        # 获取各因子索引
        endo_idx = self.factor_names.index("Endothelial") if "Endothelial" in self.factor_names else None
        peri_idx = self.factor_names.index("Pericyte") if "Pericyte" in self.factor_names else None
        smc_idx = self.factor_names.index("SMC") if "SMC" in self.factor_names else None
        
        if endo_idx is None:
            # self._log("⚠️ 警告: 缺少Endothelial因子，使用均值作为血管分数")
            return factor_props.mean(axis=1)
        
        endo = factor_props[:, endo_idx]
        
        # 计算支持细胞分数
        support_scores = []
        if peri_idx is not None:
            support_scores.append(factor_props[:, peri_idx])
        if smc_idx is not None:
            support_scores.append(factor_props[:, smc_idx])
        
        if support_scores:
            support = np.mean(support_scores, axis=0)
        else:
            support = np.zeros_like(endo)
        
        # 内皮中心公式: support * endo 确保没有内皮时分数为0
        vascular_score = self.endo_weight * endo + self.support_weight * support * endo
        
        return vascular_score
    
    def process(self, file_path, output_dir=None):
        """
        主处理流程
        
        Returns:
        --------
        results : pd.DataFrame
            六边形级结果
        pixel_results : pd.DataFrame or None
            像素级结果
        """
        self._log("=" * 60)
        self._log("🩸 Ficture血管因子反卷积 - 内皮中心版")
        self._log("=" * 60)
        self._log(f"⚙️  参数设置:")
        self._log(f"   hex_width = {self.hex_width}μm")
        self._log(f"   min_count = {self.min_count}")
        self._log(f"   endo_weight = {self.endo_weight}")
        self._log(f"   support_weight = {self.support_weight}")
        self._log(f"   include_all_genes = {self.include_all_genes}")
        
        # 1. 读取数据
        x, y, count, gene_indices, gene_names = self._read_data(file_path)
        
        # 2. 过滤基因
        mask, filtered_gene_idx, self.gene_list = self._filter_genes(
            gene_names, gene_indices, count
        )
        
        x_filtered = x[mask]
        y_filtered = y[mask]
        count_filtered = count[mask]
        
        self._log(f"   过滤后记录数: {len(x_filtered)} ({len(x_filtered)/len(x)*100:.2f}%)")
        
        # 3. 创建六边形网格
        self._log(f"\n📐 六边形网格 (width={self.hex_width}μm)")
        
        x_min, x_max = x_filtered.min(), x_filtered.max()
        y_min, y_max = y_filtered.min(), y_filtered.max()
        
        hex_height = self.hex_width * np.sqrt(3) / 2
        x_spacing = self.hex_width * 0.75
        y_spacing = hex_height
        hex_radius = self.hex_width * 0.58
        
        hex_centers = []
        row = 0
        curr_y = y_min
        while curr_y <= y_max + hex_height:
            offset = (row % 2) * (x_spacing / 2)
            curr_x = x_min + offset
            while curr_x <= x_max + self.hex_width:
                hex_centers.append([curr_x, curr_y])
                curr_x += x_spacing
            curr_y += y_spacing
            row += 1
        
        hex_centers = np.array(hex_centers)
        self._log(f"   六边形总数: {len(hex_centers)}")
        
        # 4. 分配和聚合
        if HAS_NUMBA:
            assignments = _fast_hexagon_assign(
                x_filtered, y_filtered,
                hex_centers[:, 0], hex_centers[:, 1],
                hex_radius
            )
            hex_matrix, hex_counts = _fast_aggregate(
                assignments, filtered_gene_idx, count_filtered,
                len(hex_centers), len(self.gene_list)
            )
        else:
            tree = cKDTree(hex_centers)
            distances, assignments = tree.query(
                np.column_stack([x_filtered, y_filtered]), k=1, workers=self.n_jobs
            )
            assignments[distances > hex_radius] = -1
            
            hex_matrix = np.zeros((len(hex_centers), len(self.gene_list)))
            hex_counts = np.zeros(len(hex_centers))
            
            for i, (a, g, c) in enumerate(zip(assignments, filtered_gene_idx, count_filtered)):
                if a >= 0:
                    hex_matrix[a, g] += c
                    hex_counts[a] += c
        
        # 过滤低表达六边形
        valid_hex = hex_counts >= self.min_count
        n_valid = valid_hex.sum()
        self._log(f"   有效六边形: {n_valid} ({n_valid/len(hex_centers)*100:.1f}%)")
        
        if n_valid < 100:
            self._log(f"\n⚠️  警告: 有效六边形过少 ({n_valid})!")
            self._log(f"   建议: 降低 min_count 或增大 hex_width")
        
        # 5. LDA因子学习
        self._log(f"\n🧬 LDA因子学习")
        
        eta = self._build_prior_matrix(self.gene_list)
        
        if self.use_variational:
            self.lda_model = VariationalLDA(
                n_topics=len(self.factor_names),
                eta=eta,
                max_iter=100
            )
        else:
            from sklearn.decomposition import LatentDirichletAllocation
            self.lda_model = LatentDirichletAllocation(
                n_components=len(self.factor_names),
                max_iter=100,
                learning_method='online',
                random_state=42,
                n_jobs=self.n_jobs
            )
        
        valid_matrix = hex_matrix[valid_hex]
        if sparse.issparse(valid_matrix):
            valid_matrix = valid_matrix.toarray()
        
        self.lda_model.fit(valid_matrix, verbose=self.verbose)
        
        # 推断因子
        factor_props = self.lda_model.transform(valid_matrix)
        
        # 6. 计算血管分数 (内皮中心法)
        self._log(f"\n🔬 计算血管分数 (内皮中心法)")
        vascular_score = self._calc_vascular_score(factor_props)
        
        # 7. 构建结果
        valid_indices = np.where(valid_hex)[0]
        results = pd.DataFrame({
            'hex_id': valid_indices,
            'x': hex_centers[valid_indices, 0],
            'y': hex_centers[valid_indices, 1],
            'total_count': hex_counts[valid_hex]
        })
        
        # 各因子分数
        for i, factor in enumerate(self.factor_names):
            results[f'{factor}_score'] = factor_props[:, i]
        
        # 血管综合分数
        results['Vascular_score'] = vascular_score
        
        # 归一化各分数到0-1
        score_cols = [c for c in results.columns if '_score' in c]
        for col in score_cols:
            min_v, max_v = results[col].min(), results[col].max()
            if max_v > min_v:
                results[f'{col}_norm'] = (results[col] - min_v) / (max_v - min_v)
            else:
                results[f'{col}_norm'] = 0.0
        
        # 8. 像素级解码
        pixel_results = None
        if self.pixel_level:
            self._log(f"\n🔬 像素级解码")
            pixel_results = self._pixel_decode(
                x_filtered, y_filtered, filtered_gene_idx, count_filtered,
                hex_centers[valid_indices], factor_props
            )
        
        # 9. 保存结果
        if output_dir:
            self._save_results(results, pixel_results, output_dir)
        
        # 10. 打印统计
        self._log(f"\n📊 统计摘要:")
        self._log(f"   总六边形数: {len(results)}")
        self._log(f"   血管分数均值: {results['Vascular_score'].mean():.4f}")
        self._log(f"   血管分数中位数: {results['Vascular_score'].median():.4f}")
        self._log(f"   血管分数标准差: {results['Vascular_score'].std():.4f}")
        
        self._log(f"\n✅ 完成!")
        
        return results, pixel_results
    
    def _pixel_decode(self, x, y, gene_idx, counts, anchor_coords, anchor_factors):
        """像素级解码"""
        self.pixel_decoder = PixelLevelDecoder(
            anchor_resolution=self.anchor_resolution,
            neighbor_radius=self.hex_width,
            spatial_sigma=self.hex_width / 2,
            n_jobs=self.n_jobs
        )
        
        pixel_coords = np.column_stack([x, y])
        pixel_factors = self.pixel_decoder.decode_pixels(
            pixel_coords, anchor_coords, anchor_factors
        )
        
        results = pd.DataFrame({
            'x': x, 'y': y,
            'gene_idx': gene_idx,
            'count': counts
        })
        
        for i, factor in enumerate(self.factor_names):
            results[f'{factor}_score'] = pixel_factors[:, i]
        
        # 计算像素级血管分数
        vascular_score = self._calc_vascular_score(pixel_factors)
        results['Vascular_score'] = vascular_score
        
        return results
    
    def _save_results(self, results, pixel_results, output_dir):
        """保存结果"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存六边形结果
        csv_path = os.path.join(output_dir, f'hex_vascular_{self.hex_width}um.csv')
        results.to_csv(csv_path, index=False)
        self._log(f"💾 保存: {csv_path}")
        
        # 保存像素级结果
        if pixel_results is not None:
            pixel_path = os.path.join(output_dir, 'pixel_vascular.csv.gz')
            pixel_results.to_csv(pixel_path, index=False, compression='gzip')
            self._log(f"💾 保存: {pixel_path}")
        
        # 绘图
        self._plot_results(results, output_dir)
    
    def _plot_results(self, results, output_dir):
        """绘制结果图"""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            self._log("⚠️ 无法导入matplotlib，跳过绘图")
            return
        
        # 各因子分数
        score_cols = [c for c in results.columns if '_score' in c and '_norm' not in c]
        n_cols = min(3, len(score_cols))
        n_rows = (len(score_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows),
                                 facecolor='black')
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        
        cmaps = ['Reds', 'Greens', 'Blues', 'hot', 'Purples', 'Oranges']
        
        for idx, col in enumerate(score_cols):
            ax = axes[idx // n_cols, idx % n_cols]
            ax.set_facecolor('black')
            
            scatter = ax.scatter(
                results['x'], results['y'],
                c=results[col], cmap=cmaps[idx % len(cmaps)],
                s=3, marker='h', alpha=0.8,
                vmin=0
            )
            plt.colorbar(scatter, ax=ax, fraction=0.046)
            
            title = col.replace('_score', '')
            ax.set_title(title, color='white', fontsize=12, fontweight='bold')
            ax.set_aspect('equal')
            ax.axis('off')
        
        for idx in range(len(score_cols), n_rows * n_cols):
            axes[idx // n_cols, idx % n_cols].set_visible(False)
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, 'vascular_factors.png')
        plt.savefig(plot_path, dpi=200, facecolor='black', bbox_inches='tight')
        plt.close()
        self._log(f"💾 图片: {plot_path}")


# ============================================================
# 便捷接口
# ============================================================
def run_vascular_deconv(file_path, output_dir=None, **kwargs):
    """
    运行血管因子反卷积
    
    Parameters:
    -----------
    file_path : str
        输入文件路径
    output_dir : str
        输出目录
    **kwargs : 
        其他参数传递给 FictureVascularDeconv
        
    Returns:
    --------
    results, pixel_results
    """
    processor = FictureVascularDeconv(**kwargs)
    return processor.process(file_path, output_dir)


# ============================================================
# 使用示例
# ============================================================
if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════╗
║       Ficture血管因子反卷积 - 内皮中心版 (简化版)                    ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                   ║
║  评分公式:                                                         ║
║    vascular_score = endo_weight * endo + support_weight * support * endo
║                                                                   ║
║  设计理念:                                                         ║
║    - 内皮细胞是血管的本质特征                                        ║
║    - 没有内皮就不是真正的血管 (support * endo 确保这一点)             ║
║    - 周细胞和SMC是血管成熟度标志，起加成作用                          ║
║                                                                   ║
║  使用示例:                                                         ║
║                                                                   ║
║    # 基本用法                                                      ║
║    processor = FictureVascularDeconv(                             ║
║        hex_width=50,                                              ║
║        min_count=1                                                ║
║    )                                                              ║
║    results, _ = processor.process('tissue.gef', 'output/')       ║
║                                                                   ║
║    # 自定义权重                                                    ║
║    processor = FictureVascularDeconv(                             ║
║        hex_width=50,                                              ║
║        endo_weight=0.8,    # 增加内皮权重                           ║
║        support_weight=0.2  # 减少支持细胞权重                        ║
║    )                                                              ║
║                                                                   ║
║    # 便捷接口                                                      ║
║    results, _ = run_vascular_deconv(                              ║
║        'tissue.gef',                                              ║
║        output_dir='output/',                                      ║
║        hex_width=50,                                              ║
║        min_count=1                                                ║
║    )                                                              ║
║                                                                   ║
╚══════════════════════════════════════════════════════════════════╝
    """)