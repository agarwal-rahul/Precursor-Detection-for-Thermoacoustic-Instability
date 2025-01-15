"""
Thermoacoustic Stability Analysis System

This code implements a comprehensive stability analysis framework combining:
1. The 0-1 test for chaos detection
2. Lyapunov exponent analysis
3. Segment-based stability classification
4. Combustion dynamics simulation

The system provides tools for:
- Detecting chaotic behavior in pressure signals
- Computing stability metrics
- Classifying signal segments
- Managing and storing analysis results
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import os

# ===== Data Classes for Analysis Results =====
@dataclass
class TranslationVariables:
    """
    Storage class for translation components in the 0-1 test.
    
    Attributes:
        p (np.ndarray): Translation variables in p-direction
        q (np.ndarray): Translation variables in q-direction
        c (float): Translation parameter value
    """
    p: np.ndarray
    q: np.ndarray
    c: float

@dataclass
class DisplacementResult:
    """
    Storage class for mean square displacement results.
    
    Attributes:
        D (np.ndarray): Mean square displacement values
        n_values (np.ndarray): Corresponding time indices
    """
    D: np.ndarray
    n_values: np.ndarray

@dataclass
class TestResults:
    """
    Comprehensive results from the 0-1 test analysis.
    
    Attributes:
        K (float): Final K-value indicating regularity (0) or chaos (1)
        K_values (List[float]): Individual K-values for different c parameters
        translations (Dict[float, TranslationVariables]): Translation results for each c
        displacements (Dict[float, DisplacementResult]): Displacement results for each c
        is_chaotic (bool): Binary classification of dynamics
    """
    K: float
    K_values: List[float]
    translations: Dict[float, TranslationVariables]
    displacements: Dict[float, DisplacementResult]
    is_chaotic: bool

@dataclass
class CombinedAnalysisResult:
    """
    Combined results from sliding window analysis including both
    0-1 test and Lyapunov exponents.
    
    Attributes:
        window_centers (np.ndarray): Center points of analysis windows
        K_values (np.ndarray): 0-1 test results for each window
        lyapunov_values (np.ndarray): Lyapunov exponents for each window
        time (np.ndarray): Time points of original signal
        signal (np.ndarray): Original signal values
    """
    window_centers: np.ndarray
    K_values: np.ndarray
    lyapunov_values: np.ndarray
    time: np.ndarray
    signal: np.ndarray

@dataclass
class LyapunovResult:
    """
    Results from Lyapunov exponent calculation.
    
    Attributes:
        lambda_max (float): Largest Lyapunov exponent
        convergence (List[float]): Convergence history of exponent
        embedding_dimension (int): Optimal embedding dimension used
        delay (int): Time delay used for phase space reconstruction
    """
    lambda_max: float
    convergence: List[float]
    embedding_dimension: int
    delay: int

class SegmentHandler:
    """
    Manages the storage and classification of pressure signal segments.
    
    Handles:
    1. Segment storage organization
    2. Stability classification based on combined metrics
    3. File naming and directory management
    """
    def __init__(self, base_path="pressure_segments"):
        """
        Initialize segment handler with storage directory.
        
        Args:
            base_path (str): Base directory for segment storage
        """
        self.base_path = base_path
        self.data_path = os.path.join(base_path, "data")
        os.makedirs(self.data_path, exist_ok=True)

    def determine_stability(self, K_value: float, lyap_value: float) -> int:
        """
        Determine stability based on combined K-value and Lyapunov exponent.
        
        Classification logic:
        - High K + Positive λ: Unstable (chaotic)
        - Low K + Positive λ: Unstable (non-chaotic divergence)
        - Low K + Negative λ: Stable (convergent)
        
        Args:
            K_value (float): K statistic from 0-1 test [0,1]
            lyap_value (float): Largest Lyapunov exponent
            
        Returns:
            int: 1 for unstable, 0 for stable
        """
        if K_value >= 0.5:
            return 1  # Clear unstable chaotic case
        elif K_value < 0.5 and lyap_value >= 0:
            return 1  # Non-chaotic but divergent
        elif K_value < 0.5 and lyap_value < 0:
            return 0  # Clear stable case
        
        #if K_value >= 0.5 and lyap_value >= 0:
        #    return 1  # Clear unstable chaotic case
        #elif K_value < 0.5 and lyap_value >= 0:
        #    return 1  # Non-chaotic but divergent
        #elif K_value <= 0 and lyap_value <= 0:
        #    return 0  # Clear stable case
        #else:
        #    return 1  # Default to unstable for ambiguous cases
            
    def save_segment(self, segment: np.ndarray, stability: int, 
                    U0: float, segment_idx: int):
        """
        Save signal segment with metadata in standardized format.
        
        File naming convention:
        U0_[flow velocity]_segment_[index]_stability_[0/1].npy
        
        Args:
            segment (np.ndarray): Signal segment to save
            stability (int): Stability classification (0/1)
            U0 (float): Flow velocity parameter
            segment_idx (int): Segment index in sequence
        """
        filename = f"U0_{U0:.2f}_segment_{segment_idx:04d}_stability_{stability}.npy"
        filepath = os.path.join(self.data_path, filename)
        np.save(filepath, segment)

class LyapunovExponent:
    """
    Implements Lyapunov exponent calculation using:
    1. Phase space reconstruction
    2. Nearest neighbor tracking
    3. Divergence rate estimation
    
    Features:
    - Automatic embedding dimension selection
    - Adaptive delay selection
    - Robust divergence rate estimation
    """
    def __init__(self, 
                 min_neighbors: int = 5,
                 min_embedding_dim: int = 3,
                 max_embedding_dim: int = 10,
                 delay: Optional[int] = None,
                 min_time_steps: int = 10,
                 max_time_steps: int = 50):
        """
        Initialize Lyapunov exponent calculator.
        
        Args:
            min_neighbors (int): Minimum neighbors for statistics
            min_embedding_dim (int): Minimum dimension to try
            max_embedding_dim (int): Maximum dimension to try
            delay (Optional[int]): Fixed delay or None for automatic
            min_time_steps (int): Minimum tracking duration
            max_time_steps (int): Maximum tracking duration
        """
        self.min_neighbors = min_neighbors
        self.min_embedding_dim = min_embedding_dim
        self.max_embedding_dim = max_embedding_dim
        self.delay = delay
        self.min_time_steps = min_time_steps
        self.max_time_steps = max_time_steps

    def find_embedding_delay(self, signal: np.ndarray) -> int:
        """
        Find optimal embedding delay using autocorrelation.
        
        Method:
        1. Compute autocorrelation function
        2. Find first zero crossing or minimum
        
        This ensures time-delayed coordinates are maximally independent
        while maintaining dynamical connection.
        
        Args:
            signal (np.ndarray): Input time series
            
        Returns:
            int: Optimal embedding delay
        """
        n_points = len(signal)
        max_delay = min(n_points // 3, 100)  # Limit maximum delay
        
        # Compute autocorrelation function
        autocorr = np.correlate(signal - np.mean(signal), 
                              signal - np.mean(signal), 
                              mode='full')[n_points-1:]
        autocorr = autocorr / autocorr[0]  # Normalize
        
        # Find optimal delay
        for delay in range(1, max_delay):
            # Check for zero crossing or minimum
            if autocorr[delay] <= 0 or (delay > 1 and 
               autocorr[delay] > autocorr[delay-1] and 
               autocorr[delay-1] < autocorr[delay-2]):
                return delay
        
        return max_delay // 4  # Default if no clear minimum
    
    def embed_time_series(self, signal: np.ndarray, 
                         embedding_dim: int, 
                         delay: int) -> np.ndarray:
        """
        Perform time-delay embedding for phase space reconstruction.
        
        Method (Takens' embedding theorem):
        1. Create delay vectors using time-delayed coordinates
        2. Build embedding matrix where each row is a phase space point
        
        The embedding theoretically preserves the dynamical properties
        of the original system when:
            embedding_dim > 2 * attractor_dimension
        
        Args:
            signal (np.ndarray): Input time series
            embedding_dim (int): Number of embedding dimensions
            delay (int): Time delay between coordinates
            
        Returns:
            np.ndarray: Embedded time series [n_points x embedding_dim]
        """
        n_points = len(signal) - (embedding_dim - 1) * delay
        embedded = np.zeros((n_points, embedding_dim))
        
        for i in range(embedding_dim):
            embedded[:, i] = signal[i*delay:i*delay + n_points]
            
        return embedded

    def find_neighbors(self, 
                      point: np.ndarray, 
                      embedded: np.ndarray, 
                      exclude_window: int) -> List[int]:
        """
        Find nearest neighbors in phase space excluding temporal neighbors.
        
        Algorithm:
        1. Compute distances to all points
        2. Exclude points within temporal window
        3. Select nearest neighbors
        
        The temporal exclusion window helps ensure neighbors are
        dynamically similar rather than temporally adjacent.
        
        Args:
            point (np.ndarray): Reference point in phase space
            embedded (np.ndarray): Full embedded time series
            exclude_window (int): Points to exclude around reference
            
        Returns:
            List[int]: Indices of nearest neighbors
        """
        n_points = len(embedded)
        point_idx = len(embedded) - len(point)
        
        # Compute Euclidean distances to all points
        distances = np.linalg.norm(embedded - point, axis=1)
        
        # Exclude temporal neighbors
        exclude_start = max(0, point_idx - exclude_window)
        exclude_end = min(n_points, point_idx + exclude_window)
        distances[exclude_start:exclude_end] = np.inf
        
        # Find nearest neighbors
        neighbor_indices = np.argsort(distances)[:self.min_neighbors]
        return [idx for idx in neighbor_indices if distances[idx] < np.inf]

    def track_divergence(self, 
                        embedded: np.ndarray, 
                        reference_idx: int, 
                        neighbor_indices: List[int],
                        dt: float) -> Tuple[List[float], List[float]]:
        """
        Track the divergence of nearby trajectories in phase space.
        
        Method:
        1. Follow reference trajectory and neighbor trajectories
        2. Compute logarithmic distances over time
        3. Average across neighbors
        
        This implements the core algorithm for estimating
        the largest Lyapunov exponent from time series.
        
        Args:
            embedded (np.ndarray): Embedded time series
            reference_idx (int): Index of reference trajectory
            neighbor_indices (List[int]): Indices of neighbor trajectories
            dt (float): Time step between samples
            
        Returns:
            Tuple[List[float], List[float]]: Time points and divergence values
        """
        reference_traj = embedded[reference_idx:]
        n_steps = min(len(reference_traj), self.max_time_steps)
        
        divergences = []  # Log distances for each time
        times = []       # Corresponding time points
        
        for step in range(1, n_steps):
            # Skip initial transient
            if step < self.min_time_steps:
                continue
                
            step_divergences = []
            for neighbor_idx in neighbor_indices:
                # Check if neighbor trajectory is long enough
                if neighbor_idx + step >= len(embedded):
                    continue
                    
                # Compute distance at current time
                distance = np.linalg.norm(
                    reference_traj[step] - embedded[neighbor_idx + step]
                )
                if distance > 0:
                    step_divergences.append(np.log(distance))
            
            # Store average divergence if we have data
            if step_divergences:
                divergences.append(np.mean(step_divergences))
                times.append(step * dt)
        
        return times, divergences
    
    def fit_divergence_rate(self, times: List[float], divergences: List[float]) -> float:
        """
        Estimate Lyapunov exponent from divergence data using linear regression.
        
        Method:
        1. Fit line to log-distance vs time
        2. Slope gives Lyapunov exponent
        
        The exponential divergence rate (λ) satisfies:
            d(t) ∝ exp(λt)
        So log(d) = λt + c
        
        Args:
            times (List[float]): Time points
            divergences (List[float]): Log-distance values
            
        Returns:
            float: Estimated largest Lyapunov exponent
        """
        if len(times) < 2:
            return float('-inf')
            
        t = np.array(times)
        d = np.array(divergences)
        
        # Linear least squares fit
        A = np.vstack([t, np.ones(len(t))]).T
        slope, _ = np.linalg.lstsq(A, d, rcond=None)[0]
        
        return slope

    def compute(self, signal: np.ndarray, dt: float) -> float:
        """
        Compute largest Lyapunov exponent using optimal embedding.
        
        Algorithm:
        1. Find optimal embedding parameters
        2. Try increasing embedding dimensions
        3. Track trajectory divergence rates
        4. Select best estimate
        
        Args:
            signal (np.ndarray): Input time series
            dt (float): Time step between samples
            
        Returns:
            float: Largest Lyapunov exponent
        """
        if self.delay is None:
            self.delay = self.find_embedding_delay(signal)
        
        best_lambda = float('-inf')
        
        # Try different embedding dimensions
        for dim in range(self.min_embedding_dim, self.max_embedding_dim + 1):
            embedded = self.embed_time_series(signal, dim, self.delay)
            
            dim_divergences = []
            
            # Use multiple reference trajectories
            n_refs = min(10, len(embedded) // 4)
            ref_indices = np.linspace(0, len(embedded)-1, n_refs, dtype=int)[:-1]
            
            for ref_idx in ref_indices:
                neighbors = self.find_neighbors(embedded[ref_idx], 
                                             embedded, 
                                             exclude_window=self.delay)
                
                if len(neighbors) < self.min_neighbors:
                    continue
                
                times, divergences = self.track_divergence(embedded, 
                                                         ref_idx, 
                                                         neighbors, 
                                                         dt)
                
                if divergences:
                    lambda_max = self.fit_divergence_rate(times, divergences)
                    if lambda_max > float('-inf'):
                        dim_divergences.append(lambda_max)
            
            # Update best estimate if we have valid results
            if dim_divergences:
                avg_lambda = np.mean(dim_divergences)
                if avg_lambda > best_lambda:
                    best_lambda = avg_lambda
        
        return best_lambda

class ZeroOneTest:
    """
    Implements the 0-1 test for chaos detection in dynamical systems.
    
    Method:
    1. Transform dynamics to translation variables
    2. Analyze asymptotic growth of mean square displacement
    3. Compute correlation coefficient K
    
    K ≈ 0 indicates regular dynamics
    K ≈ 1 indicates chaotic dynamics
    """
    
    def __init__(self, num_c: int = 100, 
                 c_bounds: Tuple[float, float] = (np.pi/5, 4*np.pi/5), 
                 threshold: float = 0.5):
        """
        Initialize 0-1 test parameters.
        
        Args:
            num_c (int): Number of random c values to use
            c_bounds (Tuple[float, float]): Range for c values
            threshold (float): K threshold for chaos classification
        """
        self.num_c = num_c
        self.c_bounds = c_bounds
        self.threshold = threshold
        self.results = None

    def run(self, timeseries: np.ndarray, n_cut: Optional[int] = None) -> TestResults:
        """
        Execute 0-1 test on input time series.
        
        Algorithm:
        1. Generate random c values
        2. Compute translation variables for each c
        3. Calculate mean square displacements
        4. Compute K values and determine chaos
        
        Args:
            timeseries (np.ndarray): Input signal
            n_cut (Optional[int]): Cutoff for displacement calculation
            
        Returns:
            TestResults: Complete test results including translations and K values
        """
        trans_computer = TranslationComputer(timeseries)
        disp_computer = DisplacementComputer(n_cut)
        c_values = np.random.uniform(self.c_bounds[0], self.c_bounds[1], self.num_c)
        translations, displacements, K_values = {}, {}, []
        
        for c in c_values:
            # Compute translation variables p, q
            trans = trans_computer.compute_translations(c)
            translations[c] = trans
            
            # Compute mean square displacement D
            disp = disp_computer.compute_displacement(trans, trans_computer.mean)
            displacements[c] = disp
            
            # Compute K value
            K_values.append(KComputer.compute_K(disp.D))
        
        # Take median K value for robustness
        final_K = float(np.median(K_values))
        
        self.results = TestResults(
            K=final_K,
            K_values=K_values,
            translations=translations,
            displacements=displacements,
            is_chaotic=final_K > self.threshold
        )
        return self.results

    def run_with_sliding_windows_and_lyapunov(self, 
                                            timeseries: np.ndarray, 
                                            window_size: int, 
                                            overlap: int, 
                                            delta_t: float) -> CombinedAnalysisResult:
        """
        Perform sliding window analysis combining 0-1 test and Lyapunov exponents.
        
        Method:
        1. Slide window through signal
        2. For each window:
           - Compute 0-1 test K value
           - Compute Lyapunov exponent
        3. Combine results for stability analysis
        
        Args:
            timeseries (np.ndarray): Input signal
            window_size (int): Size of analysis windows
            overlap (int): Overlap between windows
            delta_t (float): Time step between samples
            
        Returns:
            CombinedAnalysisResult: Results for all windows
        """
        step = window_size - overlap
        num_windows = (len(timeseries) - window_size) // step + 1
        window_centers = []
        K_values = []
        lyapunov_values = []

        # Generate c values for 0-1 test
        c_values = np.random.uniform(self.c_bounds[0], self.c_bounds[1], self.num_c)
        
        # Progress bar for long computations
        progress_bar = tqdm(total=num_windows * (len(c_values) + 1), 
                          desc="Sliding Window Analysis")

        for i in range(num_windows):
            # Extract window
            start = i * step
            end = start + window_size
            window = timeseries[start:end]

            # Compute 0-1 test K values
            window_K_values = []
            for c in c_values:
                trans_computer = TranslationComputer(window)
                disp_computer = DisplacementComputer(n_cut=int(window_size / 2))
                trans = trans_computer.compute_translations(c)
                disp = disp_computer.compute_displacement(trans, trans_computer.mean)
                window_K_values.append(KComputer.compute_K(disp.D))
                progress_bar.update(1)

            # Compute Lyapunov exponent
            analyzer = LyapunovExponent()
            lyap_result = analyzer.compute(window, delta_t)
            lyapunov_values.append(lyap_result)
            progress_bar.update(1)

            K_values.append(float(np.median(window_K_values)))
            window_centers.append(start + window_size // 2)

        progress_bar.close()
        
        return CombinedAnalysisResult(
            window_centers=np.array(window_centers),
            K_values=np.array(K_values),
            lyapunov_values=np.array(lyapunov_values),
            time=None,
            signal=timeseries
        )
    
class TranslationComputer:
    """
    Computes translation variables for the 0-1 test.
    
    The translation variables (p,q) are computed by:
    p(n) = Σ φ(j)cos(jc)
    q(n) = Σ φ(j)sin(jc)
    where φ is the input signal and c is a parameter.
    """
    
    def __init__(self, timeseries: np.ndarray):
        """
        Initialize with input time series.
        
        Args:
            timeseries (np.ndarray): Input signal for analysis
        """
        self.timeseries = timeseries
        self.N = len(timeseries)
        self.mean = np.mean(timeseries)

    def compute_translations(self, c: float) -> TranslationVariables:
        """
        Compute translation variables p(n) and q(n).
        
        Method:
        1. For each time n, sum over previous times j:
           - Multiply signal by cos(jc) for p
           - Multiply signal by sin(jc) for q
        2. These act as coordinates in a translation plane
        
        Args:
            c (float): Translation parameter
            
        Returns:
            TranslationVariables: p and q coordinates with parameter c
        """
        p = np.zeros(self.N)  # p-coordinate array
        q = np.zeros(self.N)  # q-coordinate array
        
        for n in range(1, self.N + 1):
            j_values = np.arange(n)
            # Compute cumulative sums with trigonometric weights
            p[n - 1] = np.sum(self.timeseries[:n] * np.cos(j_values * c))
            q[n - 1] = np.sum(self.timeseries[:n] * np.sin(j_values * c))
            
        return TranslationVariables(p=p, q=q, c=c)

class DisplacementComputer:
    """
    Computes mean square displacement in translation plane.
    
    The mean square displacement is used to distinguish between:
    - Bounded motion (regular dynamics)
    - Diffusive motion (chaotic dynamics)
    """
    
    def __init__(self, n_cut: Optional[int] = None):
        """
        Initialize displacement computer.
        
        Args:
            n_cut (Optional[int]): Maximum time difference to consider
        """
        self.n_cut = n_cut

    def compute_displacement(self, vars: TranslationVariables, timeseries_mean: float) -> DisplacementResult:
        """
        Compute mean square displacement in translation plane.
        
        Method:
        1. For each time difference n:
           - Compute squared distances between points n steps apart
           - Average over all available pairs
        
        The growth rate of D(n) distinguishes between:
        - Regular dynamics: D(n) bounded
        - Chaotic dynamics: D(n) grows linearly
        
        Args:
            vars (TranslationVariables): Translation coordinates
            timeseries_mean (float): Mean of original signal
            
        Returns:
            DisplacementResult: Displacement values and time differences
        """
        N = len(vars.p)
        n_cut = self.n_cut or N // 10
        D = np.zeros(n_cut)
        
        for n in range(1, n_cut + 1):
            # Compute squared displacements for all pairs n steps apart
            squared_displacements = [
                (vars.p[j + n] - vars.p[j]) ** 2 + (vars.q[j + n] - vars.q[j]) ** 2
                for j in range(N - n)
            ]
            D[n - 1] = np.mean(squared_displacements)
            
        return DisplacementResult(D=D, n_values=np.arange(1, n_cut + 1))

class KComputer:
    """
    Computes correlation coefficient K for the 0-1 test.
    
    K measures the growth rate of mean square displacement:
    - K ≈ 0: bounded growth (regular)
    - K ≈ 1: linear growth (chaotic)
    """
    
    @staticmethod
    def compute_K(D: np.ndarray) -> float:
        """
        Compute correlation coefficient K.
        
        Method:
        1. Compute correlation between:
           - Mean square displacement D(n)
           - Linear growth n
        2. Normalize to range [0,1]
        
        Args:
            D (np.ndarray): Mean square displacement values
            
        Returns:
            float: K value in [0,1]
        """
        n = len(D)
        time_indices = np.arange(1, n + 1)
        
        # Compute means
        mean_D = np.mean(D)
        mean_t = np.mean(time_indices)
        
        # Center variables
        D_centered = D - mean_D
        t_centered = time_indices - mean_t
        
        # Compute correlation coefficient
        covariance = np.mean(D_centered * t_centered)
        var_D = np.mean(D_centered**2)
        var_t = np.mean(t_centered**2)
        
        # Return normalized correlation
        if var_D * var_t > 0:
            return min(covariance / np.sqrt(var_D * var_t), 1.0)
        return 0.0
    
class CombustionModel:
    """
    Implements a thermoacoustic combustion model with vortex dynamics.
    
    Features:
    - Modal decomposition of acoustic field
    - Vortex formation and tracking
    - Heat release coupling
    - Acoustic-flow interactions
    """
    
    def __init__(self):
        """
        Initialize combustion model parameters and state variables.
        
        Parameters:
        1. Thermodynamic
           - Specific heat ratio (gamma)
           - Speed of sound (c0)
           - Reference density (rho0)
        
        2. Geometric
           - Combustor length (L)
           - Flame location (Lc)
           - Step height (d)
        
        3. Flow Dynamics
           - Mean flow velocity (U0)
           - Strouhal number (St)
           - Convection parameters (alpha0, sigma_alpha)
        """
        # Physical parameters
        self.gamma = 1.4        # Specific heat ratio
        self.c0 = 700.0        # Speed of sound [m/s]
        self.L = 0.7           # Combustor length [m]
        self.Lc = 0.1         # Flame location [m]
        self.d = 0.025         # Step height [m]
        self.xi1 = 29.0        # Base damping rate [1/s]
        self.St = 0.35         # Strouhal number
        self.N = 10            # Number of acoustic modes
        self.beta = 6e03       # Heat release coefficient
        self.rho0 = 1.225      # Reference density [kg/m³]
        
        # Derived parameters
        self.p0 = self.rho0 * self.c0**2 / self.gamma  # Reference pressure
        self.U0 = 8.0          # Mean flow velocity [m/s]
        self.alpha0 = 0.2      # Mean convection ratio
        self.sigma_alpha = 0.02  # Convection variation
        
        # Heat release coupling
        self.c = -2 * (self.gamma - 1) * self.beta / (self.L * self.p0)
        
        # Modal basis
        self.k = np.array([(2 * n - 1) * np.pi / (2 * self.L) for n in range(1, self.N + 1)])
        self.omega = self.c0 * self.k  # Modal frequencies [rad/s]
        
        # State variables
        self.g = np.zeros(self.N)      # Modal amplitudes
        self.g[0] = 0.001              # Initial perturbation
        self.g_dot = np.zeros(self.N)  # Modal velocities
        self.vortices = []             # Active vortices
        self.circulation = 0.0         # Accumulated circulation

    def calculate_damping(self, n: int) -> float:
        """
        Calculate mode-dependent damping coefficient.
        
        Implements quadratic scaling with mode number:
        ξₙ = ξ₁(2n-1)²
        
        Args:
            n (int): Mode number (1-based)
            
        Returns:
            float: Damping coefficient [1/s]
        """
        return self.xi1 * (2 * n - 1) ** 2

    def acoustic_basis_functions(self, x: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluate acoustic basis functions at position x.
        
        Computes:
        - Pressure modes: cos(kₙx)
        - Velocity modes: sin(kₙx)
        
        Args:
            x (float): Axial position [m]
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Pressure and velocity mode shapes
        """
        cos_terms = np.cos(self.k * x)
        sin_terms = np.sin(self.k * x)
        return cos_terms, sin_terms

    def calculate_pressure(self, x: float) -> float:
        """
        Calculate acoustic pressure at position x.
        
        Uses modal expansion:
        p'(x,t) = -p₀ Σ (ġₙ/ωₙ)cos(kₙx)
        
        Args:
            x (float): Axial position [m]
            
        Returns:
            float: Acoustic pressure [Pa]
        """
        cos_terms, _ = self.acoustic_basis_functions(x)
        return -self.p0 * np.sum(self.g_dot * cos_terms / self.omega)
    
    def calculate_velocity(self, x: float) -> float:
        """
        Calculate acoustic velocity at position x.
        
        Uses modal expansion:
        u'(x,t) = (c₀/γ) Σ gₙ sin(kₙx)
        
        Args:
            x (float): Axial position [m]
            
        Returns:
            float: Acoustic velocity [m/s]
        """
        _, sin_terms = self.acoustic_basis_functions(x)
        return self.c0 / self.gamma * np.sum(self.g * sin_terms)

    def update_vortex_positions(self, dt: float) -> None:
        """
        Update positions of all active vortices.
        
        Includes:
        1. Mean convection
        2. Acoustic velocity effects
        3. Random turbulent fluctuations
        
        Args:
            dt (float): Time step [s]
        """
        for vortex in self.vortices:
            x, C = vortex  # Position and circulation
            
            # Add random turbulent fluctuation
            alpha = self.alpha0 + self.sigma_alpha * np.random.normal(0, 1)
            
            # Update position using total velocity
            dx = (alpha * self.U0 + self.calculate_velocity(x)) * dt
            vortex[0] += dx

    def update_circulation(self, dt: float) -> None:
        """
        Update circulation accumulation and check for vortex formation.
        
        Process:
        1. Accumulate circulation based on separation velocity
        2. Check against Strouhal criterion
        3. Form new vortex when threshold reached
        
        Args:
            dt (float): Time step [s]
        """
        # Total velocity at separation point
        u_sep = self.U0 + self.calculate_velocity(0)
        
        # Accumulate circulation
        self.circulation += 0.5 * u_sep**2 * dt
        
        # Critical circulation from Strouhal criterion
        C_crit = u_sep * self.d / (2 * self.St)
        
        # Check for vortex formation
        if self.circulation >= C_crit:
            self.vortices.append([0.0, self.circulation])
            self.circulation = 0.0

    def calculate_heat_release(self, C: float) -> np.ndarray:
        """
        Calculate heat release impact on modal velocities.
        
        Models heat release as impulsive forcing when
        vortices reach the flame holder.
        
        Args:
            C (float): Vortex circulation [m²/s]
            
        Returns:
            np.ndarray: Modal velocity impulses
        """
        cos_terms = np.cos(self.k * self.Lc)
        return self.c * C * self.omega * cos_terms

    def handle_vortex_impingement(self) -> None:
        """
        Process vortex-flame interactions.
        
        When vortices reach x = Lc:
        1. Calculate heat release impulse
        2. Apply impulse to modal velocities
        3. Remove the vortex
        """
        for i in range(len(self.vortices) - 1, -1, -1):
            if self.vortices[i][0] >= self.Lc:
                # Extract vortex circulation
                C = self.vortices[i][1]
                
                # Apply heat release impulse
                self.g_dot += self.calculate_heat_release(C)
                
                # Remove vortex
                self.vortices.pop(i)

    def update_modal_amplitudes(self, dt: float) -> None:
        """
        Update modal amplitudes using RK4 integration.
        
        Each mode follows:
        d²gₙ/dt² + 2ξₙdgₙ/dt + ωₙ²gₙ = 0
        
        Args:
            dt (float): Time step [s]
        """
        for n in range(self.N):
            # Get mode-specific damping
            xi_n = self.calculate_damping(n + 1)
            
            # Current acceleration
            g_ddot = -xi_n * self.g_dot[n] - self.omega[n]**2 * self.g[n]
            
            # RK4 coefficients

    def update_modal_amplitudes(self, dt: float) -> None:
        """
        Update modal amplitudes using RK4 integration.
        
        Implements 4th order Runge-Kutta method for the system:
        dg/dt = v
        dv/dt = -2ξv - ω²g
        
        Args:
            dt (float): Time step [s]
        """
        for n in range(self.N):
            # Get mode-specific damping
            xi_n = self.calculate_damping(n + 1)
            
            # Current acceleration
            g_ddot = -xi_n * self.g_dot[n] - self.omega[n]**2 * self.g[n]
            
            # RK4 coefficients for position (k) and velocity (l)
            k1 = dt * self.g_dot[n]
            l1 = dt * g_ddot
            
            k2 = dt * (self.g_dot[n] + 0.5 * l1)
            l2 = dt * (-xi_n * (self.g_dot[n] + 0.5 * l1) - 
                      self.omega[n]**2 * (self.g[n] + 0.5 * k1))
            
            k3 = dt * (self.g_dot[n] + 0.5 * l2)
            l3 = dt * (-xi_n * (self.g_dot[n] + 0.5 * l2) - 
                      self.omega[n]**2 * (self.g[n] + 0.5 * k2))
            
            k4 = dt * (self.g_dot[n] + l3)
            l4 = dt * (-xi_n * (self.g_dot[n] + l3) - 
                      self.omega[n]**2 * (self.g[n] + k3))
            
            # Update using weighted average
            self.g[n] += (k1 + 2 * k2 + 2 * k3 + k4) / 6
            self.g_dot[n] += (l1 + 2 * l2 + 2 * l3 + l4) / 6

    def simulate(self, t_end: float, dt: float):
        """
        Run full time-domain simulation.
        
        Simulates the coupled system including:
        1. Acoustic field evolution
        2. Vortex dynamics
        3. Heat release coupling
        
        Args:
            t_end (float): End time for simulation [s]
            dt (float): Time step [s]
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Time and pressure arrays
        """
        # Create time array
        t = np.arange(0, t_end, dt)
        
        # Initialize pressure history
        pressure_history = np.zeros_like(t)
        
        # Main time-stepping loop
        for i, _ in enumerate(t):
            # Record pressure at monitoring point
            pressure_history[i] = self.calculate_pressure(0.09)
            
            # Update physical processes
            self.update_circulation(dt)         # Check for new vortices
            self.update_vortex_positions(dt)    # Move existing vortices
            self.handle_vortex_impingement()    # Process flame interactions
            self.update_modal_amplitudes(dt)    # Advance acoustic field
            
        return t, pressure_history

def process_signals():
    """
    Main processing function for stability analysis.
    
    Workflow:
    1. Generate pressure signals for different flow velocities
    2. Analyze stability using sliding windows
    3. Classify and save segments
    4. Combine 0-1 test and Lyapunov analysis
    """
    # Analysis parameters
    t_end = 0.2           # Simulation duration [s]
    dt = 5e-5            # Time step [s]
    window_size = 500    # Analysis window size
    overlap = 250        # Window overlap
    U0_range = np.linspace(7.0, 10.0, 30)  # Flow velocities to test
    
    # Initialize analysis tools
    handler = SegmentHandler()
    zero_one_test = ZeroOneTest(num_c=100, threshold=0.5)
    
    # Process each flow velocity
    for U0 in tqdm(U0_range, desc="Processing U0 values"):
        # Generate pressure signal
        model = CombustionModel()
        model.U0 = U0
        t, pressure = model.simulate(t_end, dt)
        
        # Analyze with sliding windows
        result = zero_one_test.run_with_sliding_windows_and_lyapunov(
            timeseries=pressure,
            window_size=window_size,
            overlap=overlap,
            delta_t=dt
        )
        
        # Process and save segments
        step = window_size - overlap
        for i, (center, K_value, lyap_value) in enumerate(zip(
            result.window_centers,
            result.K_values,
            result.lyapunov_values
        )):
            # Extract segment
            start = i * step
            end = start + window_size
            if end > len(pressure):
                break
                
            # Classify and save segment
            segment = pressure[start:end]
            stability = handler.determine_stability(K_value, lyap_value)
            handler.save_segment(segment, stability, U0, i)

    print("\nSegment Analysis Complete!")
    print("Segments have been saved in the pressure_segments/data directory")
    print("Each segment filename: U0_[value]_segment_[index]_stability_[0or1].npy")

if __name__ == "__main__":
    process_signals()