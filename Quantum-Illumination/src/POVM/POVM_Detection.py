"""
POVM (Positive Operator-Valued Measure) for Target Detection

This module implements a POVM that can distinguish between:
- Target Present: Entangled idler-signal state after beam splitter interaction
- Target Absent: Product state with no entanglement

The POVM uses the optimal Helstrom measurement for binary state discrimination.
"""

import numpy as np
from scipy.linalg import eigh
from Target_Present import monte_carlo_average
from Target_Absent_state import (
    partial_trace, thermal_rho_single, build_bell_rho_IS_mqt,
    simulate_unitary_swap_target_absent
)


def sparse_to_dense_rho(rho_sparse, dim):
    """
    Convert sparse density matrix representation to dense numpy array.
    
    Args:
        rho_sparse (dict): Sparse representation with keys ((i1, i2, ...), (j1, j2, ...))
        dim (int): Total dimension of the Hilbert space
    
    Returns:
        np.ndarray: Dense density matrix of shape (dim, dim)
    """
    rho = np.zeros((dim, dim), dtype=complex)
    
    for key, val in rho_sparse.items():
        bra_key, ket_key = key
        
        # Convert tuple keys to linear indices
        # Assuming the state space is structured as (idler, signal) with M modes each
        # We need to determine the structure from the keys
        if isinstance(bra_key, tuple) and isinstance(ket_key, tuple):
            # For now, we'll use a simpler approach: enumerate all possible states
            # This requires knowing the structure - let's make it more flexible
            pass
    
    return rho


def sparse_to_dense_rho_structured(rho_sparse, M, d):
    """
    Convert sparse density matrix to dense array for (idler, signal) structure.
    
    The idler is in a single-photon subspace (M dimensions: one photon in mode k).
    The signal is in full Fock space (d^M dimensions).
    
    Args:
        rho_sparse (dict): Keys are ((k_i, signal_tuple_i), (k_j, signal_tuple_j))
                          where k_i is idler mode index (0 to M-1)
                          and signal_tuple_i is tuple of M photon numbers
        M (int): Number of modes
        d (int): Dimension per mode (Nmax + 1)
    
    Returns:
        np.ndarray: Dense density matrix of shape (M * d^M, M * d^M)
    """
    dim_idler = M  # Single-photon subspace
    dim_signal = d ** M
    dim_total = dim_idler * dim_signal
    
    rho = np.zeros((dim_total, dim_total), dtype=complex)
    
    # Helper to convert (k, signal_tuple) to linear index
    def state_to_index(k, signal_tuple, M, d):
        """
        Convert (idler_mode, signal_tuple) to linear index.
        
        Args:
            k: Idler mode index (0 to M-1)
            signal_tuple: Tuple of M photon numbers
            M: Number of modes
            d: Dimension per mode
        """
        # Ensure signal_tuple has M elements
        if len(signal_tuple) != M:
            # Pad with zeros if needed
            signal_tuple = signal_tuple + (0,) * (M - len(signal_tuple))
        
        # Convert signal tuple to linear index in d^M space
        signal_idx = 0
        for i, n in enumerate(signal_tuple[:M]):
            signal_idx += n * (d ** i)
        
        # Total index: idler_idx * dim_signal + signal_idx
        # Idler is in single-photon subspace: k directly
        idler_idx = k
        return idler_idx * dim_signal + signal_idx
    
    for key, val in rho_sparse.items():
        try:
            (k_i, sig_i), (k_j, sig_j) = key
            
            # Ensure k_i and k_j are integers
            k_i = int(k_i)
            k_j = int(k_j)
            
            # Ensure signal tuples are tuples
            if not isinstance(sig_i, tuple):
                sig_i = tuple(sig_i) if hasattr(sig_i, '__iter__') else (sig_i,)
            if not isinstance(sig_j, tuple):
                sig_j = tuple(sig_j) if hasattr(sig_j, '__iter__') else (sig_j,)
            
            idx_i = state_to_index(k_i, sig_i, M, d)
            idx_j = state_to_index(k_j, sig_j, M, d)
            
            rho[idx_i, idx_j] = val
        except (ValueError, TypeError, IndexError) as e:
            # Skip invalid keys
            continue
    
    return rho


def compute_target_present_state(M, eta, nbar, Nmax, K=1000):
    """
    Compute the target present state as a dense density matrix.
    
    The state is in (idler, signal) space where:
    - Idler: single-photon subspace (M dimensions)
    - Signal: full Fock space (d^M dimensions)
    
    Args:
        M (int): Number of modes
        eta (float): Beam splitter transmissivity
        nbar (float): Mean photon number for thermal environment
        Nmax (int): Maximum photon number truncation
        K (int): Number of Monte Carlo samples
    
    Returns:
        np.ndarray: Dense density matrix rho_present of shape (M * d^M, M * d^M)
    """
    d = Nmax + 1
    
    # Get sparse representation
    rho_sparse = monte_carlo_average(M, eta, nbar, Nmax, K)
    
    # Convert to dense (idler in single-photon subspace, signal in full Fock space)
    rho_present = sparse_to_dense_rho_structured(rho_sparse, M, d)
    
    # Ensure it's properly normalized
    trace = np.trace(rho_present)
    if abs(trace) > 1e-10:
        rho_present = rho_present / trace
    
    return rho_present


def compute_target_absent_state(M, Nmax, Nbar):
    """
    Compute the target absent state as a dense density matrix.
    
    Uses the existing simulation function to get the target absent state,
    then converts it to match the structure of target present state.
    
    Args:
        M (int): Number of modes
        Nmax (int): Maximum photon number truncation
        Nbar (float): Mean photon number for thermal environment
    
    Returns:
        np.ndarray: Dense density matrix rho_absent
    """
    d = Nmax + 1
    dim_bank = d ** M
    
    # Use existing simulation function (eta=0 corresponds to target absent)
    data = simulate_unitary_swap_target_absent(M=M, Nmax=Nmax, Nbar=Nbar, include_phase=False)
    rho_absent_full = data['rho_out']  # This is (dim_bank^2, dim_bank^2)
    
    # The output from simulate_unitary_swap_target_absent is in full Fock space
    # We need to convert it to match the structure used by target present
    # (idler in single-photon subspace, signal in full Fock space)
    
    # Extract single-photon subspace for idler
    # In the full state, idler modes are the first M modes
    # Single-photon states: |1,0,...,0>, |0,1,0,...,0>, ..., |0,...,0,1>
    # These map to indices in d^M space: 1, d, d^2, ..., d^(M-1)
    
    # Reshape to (dim_bank, dim_bank, dim_bank, dim_bank) for (I, S, I, S)
    rho_reshaped = rho_absent_full.reshape((dim_bank, dim_bank, dim_bank, dim_bank))
    
    # Extract single-photon subspace indices
    single_photon_indices = [d ** k for k in range(M)]
    
    # Build reduced matrix in (M, d^M, M, d^M) space
    dim_idler = M
    dim_signal = dim_bank
    rho_absent = np.zeros((dim_idler * dim_signal, dim_idler * dim_signal), dtype=complex)
    
    for i, idx_i in enumerate(single_photon_indices):
        for j, idx_j in enumerate(single_photon_indices):
            # Extract block for this idler pair
            block = rho_reshaped[idx_i, :, idx_j, :]
            # Map to (i, signal_i, j, signal_j) indices
            for sig_i in range(dim_signal):
                for sig_j in range(dim_signal):
                    rho_absent[i * dim_signal + sig_i, j * dim_signal + sig_j] = block[sig_i, sig_j]
    
    # Normalize
    trace = np.trace(rho_absent)
    if abs(trace) > 1e-10:
        rho_absent = rho_absent / trace
    
    return rho_absent


def helstrom_povm(rho_0, rho_1, p0=0.5, p1=0.5):
    """
    Construct the optimal Helstrom POVM for binary state discrimination.
    
    The Helstrom measurement minimizes the error probability when distinguishing
    between two quantum states rho_0 and rho_1 with prior probabilities p0 and p1.
    
    Args:
        rho_0 (np.ndarray): Density matrix for hypothesis 0 (target absent)
        rho_1 (np.ndarray): Density matrix for hypothesis 1 (target present)
        p0 (float): Prior probability of hypothesis 0
        p1 (float): Prior probability of hypothesis 1
    
    Returns:
        tuple: (E_0, E_1) POVM elements
               E_0: POVM element for "target absent"
               E_1: POVM element for "target present"
    """
    # Ensure probabilities sum to 1
    total = p0 + p1
    p0 = p0 / total
    p1 = p1 / total
    
    # Compute the difference operator: p1*rho_1 - p0*rho_0
    diff_op = p1 * rho_1 - p0 * rho_0
    
    # Diagonalize to find positive and negative eigenspaces
    eigenvals, eigenvecs = eigh(diff_op)
    
    # E_1: projection onto positive eigenspace (target present)
    # E_0: projection onto non-positive eigenspace (target absent)
    positive_mask = eigenvals > 0
    
    # Build E_1 from positive eigenvectors
    if np.any(positive_mask):
        positive_vecs = eigenvecs[:, positive_mask]
        E_1 = positive_vecs @ positive_vecs.conj().T
    else:
        # No positive eigenvalues: always choose hypothesis 0
        E_1 = np.zeros_like(rho_0)
    
    # E_0 = I - E_1 (completeness relation)
    dim = rho_0.shape[0]
    E_0 = np.eye(dim, dtype=complex) - E_1
    
    return E_0, E_1


def measure_povm(rho, E_0, E_1):
    """
    Apply POVM measurement to a density matrix.
    
    Args:
        rho (np.ndarray): Input density matrix
        E_0 (np.ndarray): POVM element for outcome 0 (target absent)
        E_1 (np.ndarray): POVM element for outcome 1 (target present)
    
    Returns:
        dict: Measurement results with keys:
            'prob_absent': Probability of "target absent" outcome
            'prob_present': Probability of "target present" outcome
            'outcome': Most likely outcome (0 or 1)
    """
    # Probabilities: Tr(E_i * rho)
    prob_absent = np.real(np.trace(E_0 @ rho))
    prob_present = np.real(np.trace(E_1 @ rho))
    
    # Normalize (should sum to 1, but numerical errors might occur)
    total = prob_absent + prob_present
    if total > 1e-10:
        prob_absent = prob_absent / total
        prob_present = prob_present / total
    
    # Most likely outcome
    outcome = 1 if prob_present > prob_absent else 0
    
    return {
        'prob_absent': prob_absent,
        'prob_present': prob_present,
        'outcome': outcome
    }


def error_probability_helstrom(rho_0, rho_1, p0=0.5, p1=0.5):
    """
    Compute the optimal error probability using Helstrom bound.
    
    Args:
        rho_0 (np.ndarray): Density matrix for hypothesis 0
        rho_1 (np.ndarray): Density matrix for hypothesis 1
        p0 (float): Prior probability of hypothesis 0
        p1 (float): Prior probability of hypothesis 1
    
    Returns:
        float: Optimal error probability
    """
    # Helstrom bound: P_error = (1 - ||p1*rho_1 - p0*rho_0||_1) / 2
    diff_op = p1 * rho_1 - p0 * rho_0
    
    # Trace norm: sum of absolute values of eigenvalues
    eigenvals = np.linalg.eigvalsh(diff_op)
    trace_norm = np.sum(np.abs(eigenvals))
    
    P_error = (1 - trace_norm) / 2
    
    return max(0.0, min(1.0, P_error))  # Clamp to [0, 1]


def build_povm_detector(M, eta, nbar, Nmax, K=1000, p0=0.5, p1=0.5):
    """
    Build a complete POVM detector for target present/absent discrimination.
    
    Args:
        M (int): Number of modes
        eta (float): Beam splitter transmissivity (for target present case)
        nbar (float): Mean photon number for thermal environment
        Nmax (int): Maximum photon number truncation
        K (int): Number of Monte Carlo samples for target present state
        p0 (float): Prior probability of target absent
        p1 (float): Prior probability of target present
    
    Returns:
        dict: Detector with keys:
            'E_0': POVM element for target absent
            'E_1': POVM element for target present
            'rho_present': Target present state
            'rho_absent': Target absent state
            'error_prob': Optimal error probability
    """
    print(f"Building POVM detector: M={M}, eta={eta}, nbar={nbar}, Nmax={Nmax}")
    
    # Compute states
    print("Computing target present state...")
    rho_present = compute_target_present_state(M, eta, nbar, Nmax, K)
    
    print("Computing target absent state...")
    rho_absent = compute_target_absent_state(M, Nmax, nbar)
    
    # Ensure same dimensions
    if rho_present.shape != rho_absent.shape:
        min_dim = min(rho_present.shape[0], rho_absent.shape[0])
        rho_present = rho_present[:min_dim, :min_dim]
        rho_absent = rho_absent[:min_dim, :min_dim]
    
    # Build Helstrom POVM
    print("Constructing Helstrom POVM...")
    E_0, E_1 = helstrom_povm(rho_absent, rho_present, p0, p1)
    
    # Compute error probability
    error_prob = error_probability_helstrom(rho_absent, rho_present, p0, p1)
    
    print(f"Optimal error probability: {error_prob:.6f}")
    
    return {
        'E_0': E_0,
        'E_1': E_1,
        'rho_present': rho_present,
        'rho_absent': rho_absent,
        'error_prob': error_prob
    }


def test_povm_detector(M=2, eta=0.5, nbar=0.5, Nmax=3, K=500):
    """
    Test the POVM detector on both target present and absent states.
    
    Args:
        M (int): Number of modes
        eta (float): Beam splitter transmissivity
        nbar (float): Mean photon number
        Nmax (int): Maximum photon number
        K (int): Monte Carlo samples
    """
    print("=" * 60)
    print("Testing POVM Target Detector")
    print("=" * 60)
    
    # Build detector
    detector = build_povm_detector(M, eta, nbar, Nmax, K)
    
    E_0 = detector['E_0']
    E_1 = detector['E_1']
    rho_present = detector['rho_present']
    rho_absent = detector['rho_absent']
    
    # Test on target present state
    print("\n" + "-" * 60)
    print("Testing on TARGET PRESENT state:")
    print("-" * 60)
    result_present = measure_povm(rho_present, E_0, E_1)
    print(f"Probability of 'target absent': {result_present['prob_absent']:.6f}")
    print(f"Probability of 'target present': {result_present['prob_present']:.6f}")
    print(f"Detected outcome: {'TARGET PRESENT' if result_present['outcome'] == 1 else 'TARGET ABSENT'}")
    
    # Test on target absent state
    print("\n" + "-" * 60)
    print("Testing on TARGET ABSENT state:")
    print("-" * 60)
    result_absent = measure_povm(rho_absent, E_0, E_1)
    print(f"Probability of 'target absent': {result_absent['prob_absent']:.6f}")
    print(f"Probability of 'target present': {result_absent['prob_present']:.6f}")
    print(f"Detected outcome: {'TARGET PRESENT' if result_absent['outcome'] == 1 else 'TARGET ABSENT'}")
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    print(f"Optimal error probability: {detector['error_prob']:.6f}")
    print(f"Success rate (target present): {result_present['prob_present']:.6f}")
    print(f"Success rate (target absent): {result_absent['prob_absent']:.6f}")
    print("=" * 60)
    
    return detector, result_present, result_absent


if __name__ == "__main__":
    # Example usage
    test_povm_detector(M=2, eta=0.5, nbar=0.5, Nmax=3, K=500)
