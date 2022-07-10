import pyaudio
import wave
import time

CHUNK       = 1024
FORMAT      = pyaudio.paInt16           # Resolution 16 bit
CHANNELS    = 1                         # Mono
RATE        = 20000                     # Bitrate 20000 kHz


def record(t=0):
    p       = pyaudio.PyAudio()
    stream  = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    
    print("--- RECORDING STARTED")
    frames = []
    
    t_start     = time.time()           # Get time of start to break recording after <COUNT> seconds
    
    try:
        for i in range(0, int(t * RATE / CHUNK)):
            data = stream.read(CHUNK)
            frames.append(data)
        print("--- RECORDING STOPPED AFTER " + str(t) + " SECONDS")
        
    except KeyboardInterrupt:
        print("--- RECORDING STOPPED")
    except Exception as e:
        print("!!! ERROR: ", str(e))
        
    sample_width = p.get_sample_size(FORMAT)
    
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    return sample_width, frames


def record_to_file(file_path, t):
    w = wave.open(file_path, "wb")
    w.setnchannels(CHANNELS)
    sample_width, frames = record(t)
    w.setsampwidth(sample_width)
    w.setframerate(RATE)
    w.writeframes(b''.join(frames))
    w.close()
    

def record_session():
    print("START NEW RECORDING SESSION")
    print("-----------------------------")
    t_gend = input("Gender and/or person no. (e.g. p1): ")
    t_word = input("Command word number (e.g. 1):            ")
    print("\nATTENTION: Now please specify at which recording number you want to start.\nIf there are already recordings for this word, files will be overwritten otherwise!")
    t_offs = int(input("Start at recording number:              "))
    print("\n\n")
    
    filename = t_gend + "_b" + t_word + "_a"
    filetype = ".wav"
    duration = 2
    
    for i in range(t_offs, 31):
        print(50*"-")
        status = input("Start recording with <ENTER> (end session with <0>):")
        if status == "0":
            break
            return 1
        
        print("NEW RECORD  (" + filename + str(i) + ") - START IN 3")
        time.sleep(1)
        print("NEUE AUFNAHME - START IN 2")
        time.sleep(1)
        print("NEUE AUFNAHME - START IN 1")
        time.sleep(1)
        
        record_to_file(str(filename + str(i) + filetype), duration)
        
    print(50*"=")
    print("\nAll 30 recordings for a command word of this person were made or the session was terminated.\nFor other person pder command word please start new session.")
    
    
    
    
    
#record_to_file("test.wav", 2)       # Test once
record_session()                    # Launch new session