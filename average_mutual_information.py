import numpy as np
import os
from scipy.signal import savgol_filter

class AMICalculator:
    """
    Class to calculate Average Mutual Information (AMI) for estimating the optimal time delay (tau).
    """
    def __init__(self, time_series, max_lag, bins, normalize=True, smooth=True):
        self.time_series = time_series
        self.max_lag = max_lag
        self.bins = bins
        self.normalize = normalize
        self.smooth = smooth
        self.epsilon = 1e-10

        if self.normalize:
            self.time_series = (self.time_series - np.min(self.time_series)) / (
                np.max(self.time_series) - np.min(self.time_series)
            )

    def compute_ami(self):
        """Compute the Average Mutual Information (AMI) for the input time series."""
        ami_values = []

        for tau in range(1, self.max_lag + 1):
            ts_original = self.time_series[:-tau]
            ts_delayed = self.time_series[tau:]

            joint_hist, _, _ = np.histogram2d(ts_original, ts_delayed, bins=self.bins)
            joint_probs = joint_hist / joint_hist.sum()
            marginal_probs1 = joint_probs.sum(axis=0)
            marginal_probs2 = joint_probs.sum(axis=1)

            joint_probs += self.epsilon
            marginal_probs1 += self.epsilon
            marginal_probs2 += self.epsilon

            ami = np.nansum(
                joint_probs * np.log(joint_probs / (marginal_probs1[None, :] * marginal_probs2[:, None]))
            )
            ami_values.append(ami)

        ami_values = np.array(ami_values)

        if self.smooth:
            ami_values = savgol_filter(ami_values, window_length=7, polyorder=3)

        optimal_tau = (np.diff(np.sign(np.diff(ami_values))) > 0).argmax() + 1
        
        # Add validation to ensure reasonable tau value
        if optimal_tau < 1 or np.isnan(optimal_tau):
            optimal_tau = 1
        elif optimal_tau > self.max_lag:
            optimal_tau = self.max_lag

        return optimal_tau

class SegmentedAMIAnalyzer:
    """
    A class to process segmented signals and append their optimal time delay while preserving embedding dimension.
    """
    def __init__(self, input_folder="pressure_segments/data_with_embed", 
                 output_folder="pressure_segments/data_with_embed_tau", 
                 max_lag=500, bins=30, normalize=True, smooth=True):
        """Initialize the analyzer with parameters."""
        self.input_folder = input_folder  # Now reading from folder with embedding dimensions
        self.output_folder = output_folder
        self.max_lag = max_lag
        self.bins = bins
        self.normalize = normalize
        self.smooth = smooth

        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
            print(f"Created output folder: {self.output_folder}")

    def process_segmented_signals(self):
        """Process all segments to append optimal time delay while keeping embedding dimension."""
        print("\nProcessing segments to append time delays...")
        
        for file_name in sorted(os.listdir(self.input_folder)):
            if not file_name.endswith(".npy"):
                continue

            try:
                # Load the pressure segment
                file_path = os.path.join(self.input_folder, file_name)
                pressure_segment = np.load(file_path)

                # Calculate optimal time delay
                calculator = AMICalculator(
                    pressure_segment, 
                    self.max_lag, 
                    self.bins, 
                    self.normalize, 
                    self.smooth
                )
                optimal_tau = calculator.compute_ami()

                # Parse existing filename to keep embedding dimension
                # Format: U0_[value]_segment_[index]_stability_[0/1]_embed_[dim].npy
                parts = os.path.splitext(file_name)[0].split('_')
                
                # Extract all components
                U0_value = parts[1]
                segment_idx = parts[3]
                stability = parts[5]
                embed_dim = parts[7]

                # Create new filename with both embedding dimension and tau
                new_filename = f"U0_{U0_value}_segment_{segment_idx}_stability_{stability}_embed_{embed_dim}_tau_{optimal_tau}.npy"
                output_path = os.path.join(self.output_folder, new_filename)

                # Save the original segment with new filename including both embed and tau
                np.save(output_path, pressure_segment)

                print(f"Processed {file_name} -> Time Delay (tau): {optimal_tau}")

            except Exception as e:
                print(f"Error processing {file_name}: {str(e)}")

    def print_summary(self):
        """Print summary of processed segments."""
        print("\nProcessed Files Summary:")
        n_files = len([f for f in os.listdir(self.output_folder) if f.endswith('.npy')])
        print(f"Total segments processed: {n_files}")
        print(f"Files saved in: {self.output_folder}")
        print("Format: U0_[value]_segment_[index]_stability_[0/1]_embed_[dim]_tau_[value].npy")

if __name__ == "__main__":
    # Initialize analyzer with default parameters
    analyzer = SegmentedAMIAnalyzer()
    
    # Process segments and append time delays
    analyzer.process_segmented_signals()
    
    # Print summary
    analyzer.print_summary()