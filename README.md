# VNA-Time-Domain
This script uses a Chirp-Z transform library (https://github.com/garrettj403/CZT) and some basic signal processing techniques to replicate a VNA's time domain response for any linear S parameter frequency sweep.

Currently, the script uses a variable kaiser window to process the frequency data appropriately for the bandpass, lowpass impulse, and lowpass step cases.
Additionally, the user can choose to set the response in meters, feet, or seconds for the response along with the velocity factor of the cable being used.

