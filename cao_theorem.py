import numpy as np
from scipy.spatial.distance import cdist
import os

class CaoMethodProcessor:
    def __init__(self, input_folder="pressure_segments/data", 
                 output_folder="pressure_segments/data_with_embed", 
                 max_dim=20, threshold=0.05):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.max_dim = max_dim
        self.threshold = threshold

        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
            print(f"Created output folder: {self.output_folder}")

    def cao_method(self, time_series, tau):
        """
        Compute E1 and E2 statistics using Cao's method with proper error handling.
        """
        time_series = np.array(time_series)
        N = len(time_series)
        E1 = np.zeros(self.max_dim)
        E2 = np.zeros(self.max_dim)

        for d in range(1, self.max_dim):
            # Ensure we have enough points for the embedding
            if d * tau >= N:
                continue

            # Create delay vectors with proper bounds checking
            max_index = N - (d * tau)
            if max_index <= 0:
                break

            # Build embedding vectors
            embedding_indices = np.array([np.arange(i, i + d * tau, tau) for i in range(max_index)])
            Y1 = time_series[embedding_indices]
            
            # For d+1 dimension
            if (d + 1) * tau < N:
                embedding_indices_next = np.array([np.arange(i, i + (d + 1) * tau, tau) 
                                                 for i in range(N - (d + 1) * tau)])
                Y2 = time_series[embedding_indices_next]
            else:
                continue

            N1 = Y1.shape[0]
            N2 = Y2.shape[0]

            if N1 == 0 or N2 == 0:
                continue

            # Initialize arrays for storing distances
            a_d = np.zeros(N1)
            a_d1 = np.zeros(N2)

            # Process each point
            for i in range(min(N1, N2)):
                if i >= Y1.shape[0] or i >= Y2.shape[0]:
                    continue

                # Compute distances for current dimension
                dist_d = cdist([Y1[i]], Y1).flatten()
                if len(dist_d) > 1:  # Ensure we have neighbors
                    nn_d = np.argsort(dist_d)[1:min(6, len(dist_d))]
                    if len(nn_d) > 0:
                        a_d[i] = np.max(dist_d[nn_d])
                    else:
                        a_d[i] = np.nan

                # Compute distances for next dimension
                if i < N2:
                    dist_d1 = cdist([Y2[i]], Y2).flatten()
                    if len(dist_d1) > 1:
                        nn_d1 = np.argsort(dist_d1)[1:min(6, len(dist_d1))]
                        if len(nn_d1) > 0:
                            a_d1[i] = np.max(dist_d1[nn_d1])
                        else:
                            a_d1[i] = np.nan

            # Remove any NaN values
            valid_indices = ~np.isnan(a_d[:N2]) & ~np.isnan(a_d1[:N2]) & (a_d[:N2] != 0)
            
            if np.any(valid_indices):
                ratio = a_d1[:N2][valid_indices] / a_d[:N2][valid_indices]
                E1[d] = np.mean(ratio[~np.isnan(ratio) & ~np.isinf(ratio)])
            else:
                E1[d] = np.nan

            # Compute E2 with bounds checking
            if (d + 1) * tau < N:
                diffs = np.abs(time_series[((d + 1) * tau):] - time_series[:-((d + 1) * tau)])
                E2[d] = np.mean(diffs) if len(diffs) > 0 else np.nan

        # Remove any remaining NaN values
        E1 = E1[~np.isnan(E1)]
        E2 = E2[~np.isnan(E2)]

        return E1[1:], E2[1:]

    def calculate_optimal_delay(self, signal):
        """Calculate optimal time delay using autocorrelation."""
        n_points = len(signal)
        max_delay = min(n_points // 3, 100)
        
        signal_normalized = signal - np.mean(signal)
        autocorr = np.correlate(signal_normalized, signal_normalized, mode='full')[n_points-1:]
        autocorr = autocorr / autocorr[0]
        
        for delay in range(1, max_delay):
            if autocorr[delay] <= 0 or (delay > 1 and 
               autocorr[delay] > autocorr[delay-1] and 
               autocorr[delay-1] < autocorr[delay-2]):
                return delay
        return max_delay // 4

    def find_embedding_dimension(self, E1):
        """Determine embedding dimension from E1 values stabilization."""
        if len(E1) < 2:
            return 2  # Default minimum if not enough data
            
        dE1 = np.diff(E1)
        for i, rate in enumerate(dE1):
            if abs(rate) < self.threshold:
                return i + 2
        return len(E1)

    def process_segments(self):
        """Process pressure segments to append embedding dimensions."""
        print("\nProcessing pressure segments to append embedding dimensions...")
        
        for file_name in sorted(os.listdir(self.input_folder)):
            if not file_name.endswith(".npy"):
                continue

            try:
                # Load the pressure segment
                file_path = os.path.join(self.input_folder, file_name)
                pressure_segment = np.load(file_path)
                
                # Calculate optimal delay and embedding dimension
                optimal_tau = self.calculate_optimal_delay(pressure_segment)
                E1, _ = self.cao_method(pressure_segment, optimal_tau)
                
                if len(E1) > 0:  # Check if we got valid E1 values
                    embedding_dim = self.find_embedding_dimension(E1)
                else:
                    embedding_dim = 2  # Default value if calculation fails
                
                # Create new filename with embedding dimension
                base_name = os.path.splitext(file_name)[0]
                new_filename = f"{base_name}_embed_{embedding_dim}.npy"
                output_path = os.path.join(self.output_folder, new_filename)
                
                # Save the segment with embedding dimension appended to filename
                np.save(output_path, pressure_segment)
                
                print(f"Processed {file_name} -> Embedding Dimension: {embedding_dim}")
                
            except Exception as e:
                print(f"Error processing {file_name}: {str(e)}")

    def print_summary(self):
        """Print summary of processed segments"""
        print("\nProcessed Files Summary:")
        n_files = len([f for f in os.listdir(self.output_folder) if f.endswith('.npy')])
        print(f"Total segments processed: {n_files}")
        print(f"Files saved in: {self.output_folder}")
        print("Format: U0_[value]_segment_[index]_stability_[0/1]_embed_[dim].npy")

if __name__ == "__main__":
    processor = CaoMethodProcessor()
    processor.process_segments()
    processor.print_summary()