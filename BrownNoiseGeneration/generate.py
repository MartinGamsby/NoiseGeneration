import numpy as np
from scipy.io.wavfile import write
import soundfile as sf
from scipy import signal

import numpy as np


# --------------------------------------------------------------------------------------------------- 
def generate_white_noise(duration, sampling_rate):
  """
  Generates white noise with a specified duration and sampling rate.

  Args:
      duration: Duration of the noise in seconds (float).
      sampling_rate: Sampling rate of the noise in Hz (int).

  Returns:
      A NumPy array containing the white noise samples.
  """
  samples = int(duration * sampling_rate)
  return np.random.rand(samples)

  
# --------------------------------------------------------------------------------------------------- 
def filter_to_brown_noise3(white_noise, f_cutoff, sampling_rate):
  """
  Filters white noise to generate brown noise using pre-defined filter coefficients.

  Args:
      white_noise: A NumPy array containing the white noise samples.
      f_cutoff: Cutoff frequency for brown noise (Hz).
      sampling_rate: Sampling rate of the noise in Hz (int).

  Returns:
      A NumPy array containing the filtered brown noise samples.
  """
  from scipy.signal import firwin
  # Design a finite impulse response (FIR) filter
  nyquist_rate = sampling_rate / 2
  num_taps = 501  # Adjust for desired filter complexity (odd number recommended) # Was 51
  normalized_cutoff = f_cutoff# / nyquist_rate  # Normalize cutoff frequency
  taps = firwin(num_taps, normalized_cutoff, nyq=nyquist_rate, window='blackman')  # Use Hamming window 'hamming','blackman'

  # Apply filter using convolution
  brown_noise = np.convolve(white_noise, taps, mode='same')

  return brown_noise
  
# --------------------------------------------------------------------------------------------------- 
def plot(y):
    from matplotlib import pyplot as plt
    plt.title("Matplotlib demo") 
    plt.xlabel("x axis caption") 
    plt.ylabel("y axis caption") 
    plt.plot(np.arange(0,len(y)), y)
    plt.show()

# --------------------------------------------------------------------------------------------------- 
def generate_mp3(duration = 60*59+59,
    from_hz = 432*2,#960+32,#1024
    to_hz = 0.996,#0.8,#32,
    nb_steps = 480,
    do_plot=True,
    save_wav=False,
    initial_steps=[256,150]):#30#32#16#6
    
    step_hz = int((to_hz-from_hz)/nb_steps)#+1
    #print(f"step_hz:{step_hz}" + str((to_hz-from_hz)/nb_steps))
    #steps = [256,256,to_hz,to_hz]#TODO: If we change that, change nb_steps_total
    
    slowing_down_hz = [from_hz]
    for i in range(nb_steps-1):
        slowing_down_hz.append(int(slowing_down_hz[-1]*to_hz))
    
    
    steps = initial_steps#TODO: If we change that, change nb_steps_total
    steps.extend(slowing_down_hz)
    
    print(steps)
    #steps.extend(range(from_hz,to_hz+step_hz,step_hz))
    #nb_steps_total = nb_steps+4#TODO
    nb_steps_total = len(steps)

    # Generate white noise
      # Seconds
    #duration *= (nb_steps/(nb_steps_total+1))#4 is the number of steps we add at the beginning. # +2 for the Fade in and out
    sampling_rate = 44100  # Hz


    duration /= nb_steps_total#16/2# per segment (/2 including the fades)
    print(f"{duration}s per step")
    brown_noise =np.zeros(0).astype(np.float32)
    last_noise = np.zeros(int(duration * sampling_rate)).astype(np.float32)

    # Create cross-fade curve (e.g., linear or cosine ramp)

    #10ms
    # I kept increasing that to like 200ms, but the pop is AFTER the fade in!?
    small_fade_dur = 0.01*sampling_rate
    big_fade_dur = duration*sampling_rate-small_fade_dur*2

    #big_fade_curve = np.linspace(0, 1, int(big_fade_dur))  # Linear fade
    big_fade_curve = np.cos(np.linspace(np.pi/2, 0, int(big_fade_dur)))  # Linear fade
    small_fade_curve = np.linspace(0, 1, int(small_fade_dur))  # Linear fade
    #small_fade_curve = np.cos(np.linspace(np.pi/2, 0, int(small_fade_dur)))
    small_silence = np.zeros(int(small_fade_dur)).astype(np.float32)

    #fade_curve = np.cos(np.linspace(np.pi/2, 0, int(duration*sampling_rate)))
    #fade_curve_inv = np.array([1-xi for xi in fade_curve])#[1-x for x in fade_curve]
    fade_curve = np.concatenate((small_silence, big_fade_curve, (1-small_fade_curve)))
    fade_curve_inv = np.concatenate((small_fade_curve, (1-big_fade_curve), small_silence))

    #TODO: Do 2 steps down, one step up to do it gradually...
        

    print(fade_curve)
    print(fade_curve_inv)
    print(1-fade_curve)
    #for f_cutoff in range(16,256,16):#
    #for f_cutoff in range(10,1000,100):
    #for f_cutoff in range(1000,10,-200):
    for step_i, f_cutoff in enumerate(steps):
    #for f_cutoff in {1024,819,655,524,419,335,268,214,172,137,110,88,70,56}:#*0.8
        # New white noise every time
        white_noise = generate_white_noise(duration, sampling_rate)
        print(f"{int((step_i+1)*100/nb_steps_total)}%: {f_cutoff}Hz")#: ({step_i}/{nb_steps_total} steps)")
        # Filter to generate brown noise (adjust f_cutoff for desired noise characteristics)
        brown_noise_chunk = filter_to_brown_noise3(white_noise, f_cutoff, sampling_rate)

        # That doesn't change anything... ?
        ### 
        #brown_noise_chunk = brown_noise_chunk - np.mean(brown_noise_chunk)
        brown_noise_chunk = signal.detrend(brown_noise_chunk,0)# still not enough to remove the pop!?

        #Normalize ... ?
        #hz_min = max(to_hz, from_hz)
        #hz_max = min(to_hz, from_hz)
        weird_ratio = ((f_cutoff+150) / 850)#hz_min)
        weird_ratio = min(1.0, weird_ratio)
        ####print(weird_ratio)
        max_val = np.max(np.abs(brown_noise_chunk)) * 0.99 * weird_ratio#This is weird ... sorry, probably only works for 1024-64
        brown_noise_chunk = brown_noise_chunk / max_val#32767#0.8  # Avoid potential clipping

        
        #print(brown_noise_chunk)
        #print(len(brown_noise_chunk))
        for i, val in enumerate(brown_noise_chunk):
            if abs(val) < 0.01:
                #print(i, val)
                for j in range(0,i):
                    brown_noise_chunk[j] = 0.
                break
        for i,val in reversed(list(enumerate(brown_noise_chunk))):
            if abs(val) < 0.01:
                #print(i, val)
                for j in range(i,len(brown_noise_chunk)):
                    brown_noise_chunk[j] = 0.
                break
        #print(brown_noise_chunk)
        # TODO instead: something like that???

        # Apply cross-fade to overlapping segments
        #faded_noise = last_noise * (1 - fade_curve) + brown_noise_chunk * fade_curve
        faded_noise = last_noise * fade_curve_inv + brown_noise_chunk * fade_curve
        #plot(brown_noise_chunk * fade_curve)
        #plot(last_noise * fade_curve_inv)
        #plot(faded_noise)
        #faded_noise = faded_noise / max_val
        #brown_noise = np.concatenate((brown_noise, faded_noise, brown_noise_chunk))
        brown_noise = np.concatenate((brown_noise, faded_noise))
        last_noise = brown_noise_chunk
      
    f_cutoff = 0
    # Normalize the brown noise samples to the -1 to 1 range
    #max_val = np.max(np.abs(brown_noise)) * 0.99
    #print(max_val)
    #brown_noise = brown_noise / max_val#32767#0.8  # Avoid potential clipping
    #from scipy.signal import normalize
    #plot(brown_noise)

    # Add silence at the beginning and end (adjust duration as needed)
    silence_duration = 0.05  # Seconds
    silence_samples = np.zeros(int(silence_duration * sampling_rate)).astype(np.float32)
    brown_noise = np.concatenate((silence_samples, brown_noise, silence_samples))

    # Apply dithering (adjust noise_shaping parameter as needed)
    #brown_noise = dither(brown_noise, noise_shaping=None)  # Experiment with noise shaping

    # Save as WAV for testing (optional)
    #scaled = np.int16(brown_noise / np.max(np.abs(brown_noise)) * 32767)
    if save_wav:
        write(f"brown_noise_test{f_cutoff}.wav", sampling_rate, brown_noise)

    # Save as MP3 using soundfile
    mp3_name = f"brown_noise{f_cutoff}.mp3"
    sf.write(mp3_name, brown_noise.astype(np.float32), sampling_rate, format='MP3')

        
        ## Save as a WAV file (scipy.io.wavfile for simpler WAV saving)
        #write("brown_noise.wav", sampling_rate, brown_noise.astype(np.float32))
        #
        ## Use soundfile to convert WAV to MP3 (requires installing soundfile with `pip install soundfile`)
        #sf.write("brown_noise.mp3", brown_noise, sampling_rate, format='MP3')
        
    if do_plot:
        plot(brown_noise)
    return mp3_name, steps, duration
        