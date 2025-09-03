from typing import NamedTuple
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt # Import matplotlib here for example_usage



def detect_r_peaks(ecg_signal, min_distance=10):
    """
    Detects R-peaks in the ECG signal.

    Args:
        ecg_signal (array_like): The ECG signal. Should be filtered.
        sampling_rate (int): The sampling rate of the ECG signal in Hz.
        min_distance (float, optional): Minimum distance between R-peaks in seconds. Defaults to 0.2.

    Returns:
        numpy.ndarray: Indices of the detected R-peaks. Returns None if detection fails.
    """
    # # Ensure minimum distance is valid
    # min_distance_samples = int(min_distance * sampling_rate)
    # if min_distance_samples < 1:
    #     min_distance_samples = 1 # Set a minimum distance of 1 sample

    # Find peaks using scipy.signal.find_peaks
    # Add a height threshold relative to the signal's max value to avoid noise peaks
    height_threshold = 0.5 * np.max(ecg_signal) if np.max(ecg_signal) > 0 else None
    
    peaks, _ = find_peaks(ecg_signal, distance=min_distance, height=height_threshold)
    return peaks

def _filter_r_peaks(ecg_signal, peaks, sampling_rate):
    """
    Helper function to filter R-peaks (Optional).
    This is an example of a filtering method, and you might need to adapt it.
    """
    # Ensure peaks is not None and has elements
    if peaks is None or len(peaks) == 0:
        return np.array([]) # Return empty array if no peaks

    filtered_peaks = []
    if len(peaks) > 2:
        for i in range(1, len(peaks) - 1):
            prev_peak_val = ecg_signal[peaks[i-1]]
            curr_peak_val = ecg_signal[peaks[i]]
            next_peak_val = ecg_signal[peaks[i+1]]
            # Example condition: Check if current peak is significantly higher than neighbors
            if curr_peak_val > prev_peak_val and curr_peak_val > next_peak_val:
                 # Add more sophisticated checks if needed
                filtered_peaks.append(peaks[i])
        # Handle the first and last peaks if necessary (e.g., add if they meet criteria)
        # This example only considers peaks with two neighbors.
    elif len(peaks) > 0: # Handle cases with 1 or 2 peaks
         filtered_peaks = list(peaks) # Keep the peaks if only 1 or 2 detected initially

    return np.array(filtered_peaks)

def detect_t_end(averaged_rr, r_peak_index, window_start_offset=50, window_end_offset=400, threshold_factor=0.3):
    """
    Detects the end of the T-wave in the averaged RR interval using a derivative-based method.

    Args:
        averaged_rr (numpy.ndarray): The averaged RR interval. Should not be None.
        sampling_rate (int): The sampling rate of the ECG signal in Hz.
        r_peak_index (int): The index of the R-peak in the averaged RR interval.
        window_start_offset (float, optional): Start of the search window relative to R-peak (seconds). Defaults to 0.05.
        window_end_offset (float, optional): End of the search window relative to R-peak (seconds). Defaults to 0.4.
        threshold_factor (float, optional): Factor for derivative threshold calculation. Defaults to 0.3.

    Returns:
        int: Index of the T-wave end relative to the start of averaged_rr. Returns None if T-end not found or input is invalid.
    """
    if averaged_rr is None or len(averaged_rr) == 0:
        print("Error: Cannot detect T-end on empty or None averaged RR interval.")
        # return None

    # Define the search window for the T-wave end based on offsets from R-peak
    search_start = r_peak_index + window_start_offset #* sampling_rate)
    search_end = r_peak_index + window_end_offset #* sampling_rate)

    # Ensure indices are within the bounds of the averaged_rr array
    search_start = max(0, search_start) # Ensure start is not negative
    search_end = min(len(averaged_rr), search_end) # Ensure end does not exceed array length

    # Check if the search window is valid
    if search_start >= search_end or search_end - search_start < 2: # Need at least 2 points for diff
        print("Warning: Invalid or too short search window for T-end detection.")
        # return None

    # Extract the segment for T-wave end detection
    signal_segment = averaged_rr[search_start:search_end]

    # Calculate the first derivative (velocity) of the signal segment
    derivative = np.diff(signal_segment)

    if len(derivative) == 0:
        print("Warning: Could not compute derivative for T-end detection (segment too short?).")
        # return None

    # Find the maximum absolute derivative value (related to T-peak slope)
    # max_abs_derivative = np.max(np.abs(derivative))

    # Alternative: Use tangent method (find max derivative, draw tangent, find intersection)
    # This example uses a simpler threshold crossing method.

    # Find T-peak index within the segment (relative to segment start)
    t_peak_index_relative = np.argmax(np.abs(signal_segment)) # Find peak of T-wave (can be positive or negative)

    # Search for T-end after the T-peak
    search_start_tend = t_peak_index_relative #+ int(0.02 * sampling_rate) # Start search slightly after T-peak
    search_start_tend = max(0, search_start_tend) # Ensure start is not negative
    
    if search_start_tend >= len(derivative):
        print("Warning: T-peak is too close to the end of the search window.")
            # return None

    # Find the point where the derivative returns close to zero after the T-peak
    # This is a simplified approach; tangent methods are more common in literature
    # Find the index of the minimum derivative *after* the T-peak
    min_derivative_after_peak_index = np.argmin(derivative[search_start_tend:])

    # Calculate T-end relative index
    t_end_index_relative = search_start_tend + min_derivative_after_peak_index

    # Convert relative index back to the original averaged_rr index
    t_end_index_absolute = search_start + t_end_index_relative

    # Basic validation: T-end should be after R-peak

    if t_end_index_absolute <= r_peak_index:
        print("Warning: Detected T-end is before or at the R-peak index.")
        # return None

    return t_end_index_absolute




def correct_qt_interval(qt_interval_ms, rr_interval_duration_s, method='bazett'):
    """
    Corrects the QT interval for heart rate using Bazett's or Fridericia's formula.

    Args:
        qt_interval_ms (float): The QT interval in milliseconds. Can be None.
        rr_interval_duration_s (float): The RR interval duration in seconds.
        method (str, optional): The correction method ('bazett' or 'fridericia'). Defaults to 'bazett'.

    Returns:
        float: The corrected QT interval (QTc) in milliseconds. Returns None if input is invalid.
    """
    if qt_interval_ms is None:
        print("Cannot correct QT interval: Input QT interval is None.")
        return None
    if rr_interval_duration_s is None or rr_interval_duration_s <= 0:
        print(f"Cannot correct QT interval: Invalid RR interval duration ({rr_interval_duration_s} s).")
        return None

    qt_interval_s = qt_interval_ms / 1000.0 # Convert QT to seconds for formula

    try:
        if method.lower() == 'bazett':
            # QTc = QT / sqrt(RR)
            qtc_s = qt_interval_s / np.sqrt(rr_interval_duration_s)
        elif method.lower() == 'fridericia':
            # QTc = QT / cbrt(RR)
            qtc_s = qt_interval_s / (rr_interval_duration_s**(1/3))
        else:
            print(f"Warning: Invalid QTc correction method '{method}'. Using Bazett's.")
            qtc_s = qt_interval_s / np.sqrt(rr_interval_duration_s) # Default to Bazett

        qtc_ms = qtc_s * 1000.0 # Convert back to milliseconds
        return qtc_ms
    except Exception as e:
        print(f"An error occurred during QTc calculation: {e}")
        return None
    
class QTIntervalResult(NamedTuple):
    qt_interval: float
    start_index: int
    end_index: int

def process_ecg_for_qt(t, ecg_signal, qtc_method='bazett'):
    """
    Processes the ECG signal to compute the corrected QT interval (QTc).

    Args:
        ecg_signal (array_like): The raw ECG signal.
        sampling_rate (int): The sampling rate of the ECG signal in Hz.
        qtc_method (str, optional): QTc correction formula ('bazett' or 'fridericia'). Defaults to 'bazett'.


    Returns:
        dict: A dictionary containing calculated parameters:
              'qt_interval_ms', 'qtc_ms', 'median_rr_interval_s',
              'r_peak_index_avg', 't_end_index_avg', 'r_peaks_indices'.
              Returns None if processing fails at any critical step.
    """

    # 1. Preprocess the ECG signal
    print("Step 1: Preprocessing ECG...")
    filtered_ecg = ecg_signal
    
    # 2. Detect R-peaks on the filtered signal
    print("Step 2: Detecting R-peaks...")
    r_peaks = detect_r_peaks(filtered_ecg)
    assert len(r_peaks) > 0, "No R-peaks detected. Check signal quality and detection parameters."
    r_peak_index = r_peaks[0]
    t_end_index = detect_t_end(ecg_signal, r_peak_index)
    # qt_interval = t_end_index - r_peak_index

    qt_interval = t[t_end_index] - t[r_peak_index]
 

    return QTIntervalResult(start_index=r_peak_index, end_index=t_end_index, qt_interval=qt_interval,)

def example_usage():
    """
    Generate a syntheic ECG signal for demonstration purposes.
    """
    print("--- Running Example Usage (Time in Milliseconds) ---")
    # ECG parameters
    sampling_rate_hz = 1000  # Samples per second
    duration_s = 10           # Duration in seconds
    heart_rate_bpm = 60      # Beats per minute

    # Convert time parameters to milliseconds
    duration_ms = duration_s * 1000
    rr_interval_s = 60.0 / heart_rate_bpm
    rr_interval_ms = rr_interval_s * 1000

    num_beats = int(duration_s / rr_interval_s) # Or duration_ms / rr_interval_ms

    # Time vector in milliseconds
    # Total number of samples = duration_s * sampling_rate_hz
    num_samples = int(duration_s * sampling_rate_hz)
    t_ms = np.linspace(0, duration_ms, num_samples, endpoint=False)
    
    ecg_signal = np.zeros_like(t_ms)

    # --- Waveform Parameters (all in milliseconds) ---
    # Relative to R-peak time
    q_offset_ms = 40  # ms (formerly 0.04s)
    s_offset_ms = 40  # ms (formerly 0.04s)
    t_peak_offset_ms = 200 # ms (formerly 0.20s)
    # t_end_offset_ms = 300 # ms (formerly 0.30s) # Not directly used in this Gaussian model

    # Gaussian widths (standard deviations) in milliseconds
    r_width_ms = 20  # ms (formerly 0.02s)
    q_width_ms = 20  # ms (formerly 0.02s)
    s_width_ms = 30  # ms (formerly 0.03s)
    t_width_ms = 60  # ms (formerly 0.06s)
    start_p_wave = 200 # ms

    # Create multiple beats
    for i in range(num_beats):
        # R-peak time for the current beat, in milliseconds
        r_peak_time_ms = (i + start_p_wave / 1000) * rr_interval_ms 
        
        # Calculate absolute times for other wave components for this beat
        q_time_ms = r_peak_time_ms - q_offset_ms
        s_time_ms = r_peak_time_ms + s_offset_ms
        t_peak_time_ms = r_peak_time_ms + t_peak_offset_ms
        # t_end_time_ms = r_peak_time_ms + t_end_offset_ms # If needed for other models

        # Add waves for the current beat
        # R peak
        ecg_signal += 1.0 * np.exp(-((t_ms - r_peak_time_ms) / r_width_ms)**2)
        # Q wave
        ecg_signal -= 0.2 * np.exp(-((t_ms - q_time_ms) / q_width_ms)**2)
        # S wave
        ecg_signal -= 0.3 * np.exp(-((t_ms - s_time_ms) / s_width_ms)**2)
        # T wave
        ecg_signal += 0.4 * np.exp(-((t_ms - t_peak_time_ms) / t_width_ms)**2)

    # Add some baseline noise
    noise_amplitude = 0.0 # Set to 0 for no noise, or a small value like 0.05
    if noise_amplitude > 0:
        ecg_signal += noise_amplitude * np.random.randn(len(t_ms))
    
    # Add some baseline wander (low frequency noise)
    # Original: 0.2 Hz. To use with t_ms, convert frequency: 0.2 cycles/sec = 0.0002 cycles/ms
    wander_freq_hz = 0.2
    wander_freq_per_ms = wander_freq_hz / 1000.0
    wander_amplitude = 0.1 # Amplitude of the wander
    
    ecg_signal += wander_amplitude * np.sin(2 * np.pi * wander_freq_per_ms * t_ms)
    
    return t_ms, ecg_signal

def main_example():

    t, ecg_signal = example_usage()


    results = process_ecg_for_qt(t[-1000:], ecg_signal[-1000:], qtc_method='bazett')

    fig, ax = plt.subplots(2, 1)
    ax[0].plot(t, ecg_signal)
    ax[1].plot(t[-1000:], ecg_signal[-1000:])
    ax[1].plot(t[-1000:][results.start_index], ecg_signal[-1000:][results.start_index], "ro")
    ax[1].plot(t[-1000:][results.end_index], ecg_signal[-1000:][results.end_index], "go")
    fig.savefig("ecg_qt_calc.png")

def main():
    import pandas as pd
    import json
    from utils import Case
    from pathlib import Path

    qt_intervals = []
    qtdir = Path("qt_intervals")
    qtdir.mkdir(exist_ok=True)
    sexes = ["male", "female"]
    cases = [c.name for c in Case]  
    # cases = [c.name for c in Case if "TdP" not in c.name]  # Filter cases to only those with "TdP"
    qts = np.zeros((len(sexes), len(cases)))
    for i, sex in enumerate(sexes):
        for j, case in enumerate(cases):

            if "TdP" in case:
                outdir = Path("results") / f"{sex}-{case}-CTRL-initial-states"
            else:
                outdir = Path("results") / f"{sex}-{case}"
          
            if not (outdir / "ecg.csv").exists():
                continue
            df = pd.read_csv(outdir / "ecg.csv")


            t = df["time"].values[-1000:]
            ecg_signal = df["I"].values[-1000:]
            results = process_ecg_for_qt(t, ecg_signal, qtc_method='bazett')
            qt_intervals.append({
                "sex": sex,
                "case": case,
                "qt_interval": float(results.qt_interval),
                "start_index": int(results.start_index),
                "end_index": int(results.end_index),
            })
            
            qts[i, j] = results.qt_interval

            (qtdir / "qt_intervals.json").write_text(json.dumps(qt_intervals, indent=4))
            print(f"QT interval for {outdir.name}: {results.qt_interval:.2f} ms")

            fig, ax = plt.subplots()
            ax.plot(t, ecg_signal)
            ax.grid()
            ax.plot(t[results.start_index], ecg_signal[results.start_index], "ro")
            ax.plot(t[results.end_index], ecg_signal[results.end_index], "go")
            fig.savefig(qtdir / f"{outdir.name}.png")
            plt.close(fig)

    fig, ax = plt.subplots(figsize=(4, 10))
    im = ax.imshow(qts.T, cmap="YlGnBu", aspect="auto")

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(range(len(sexes)), labels=sexes,
                rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(len(cases)), labels=cases)

    # Loop over data dimensions and create text annotations.
    for i in range(len(sexes)):
        for j in range(len(cases)):
            ax.text(i, j, f"{qts[i, j]:.1f}",
                        ha="center", va="center", color="r")

    ax.set_title("QT intervals [ms]")
    fig.tight_layout()
    fig.savefig(qtdir / "qt_intervals.png")


if __name__ == "__main__":
    main()