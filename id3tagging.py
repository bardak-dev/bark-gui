from mutagen.wave import WAVE
from mutagen.id3._frames import *

def add_id3_tag(filename, text, speakername, seed):
    audio = WAVE(filename)
    if speakername == None:
        speakername = "Unconditional"

    # write id3 tag with text truncated to 60 chars, as a precaution...
    audio["TIT2"] = TIT2(encoding=3, text=text[:60])
    audio["TPE1"] = TPE1(encoding=3, text=f"Voice {speakername} using Seed={seed}")
    audio["TPUB"] = TPUB(encoding=3, text="Bark by Suno AI")
    audio["COMMENT"] = COMM(encoding=3, text="Generated with Bark GUI - Text-Prompted Generative Audio Model. Visit https://github.com/C0untFloyd/bark-gui")
    audio.save()
