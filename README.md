# Bark Web UI

This is a simple Web UI for an extended Bark Version using Gradio.

It actually is some kind of Frankenstein-Bark with the
original Code as base and various changes/improvements ripped from
these 2 repos:
- https://github.com/serp-ai/bark-with-voice-clone
- https://github.com/makawy7/bark-webui (inspired me to even start this)


### Additional Features

- Web GUI
- Creation of very large text passages in chunks, combining the parts into a final result
- Creation of new Voices possible
- Easy Selection of Small/Big Models
- Works with 8Gb GPU or force it to use CPU instead

### Showcase

Example Input:
"Hello, I am called BARK and am a new text to audio model made by SUNO! Let me read an excerpt from War of the Worlds to you. [clears throat]
We know NOW that in the early years of the twentieth century, this world was being watched closely by intelligences greater than man’s and yet as mortal as his own. We know NOW that as human beings busied themselves about their various concerns they were scrutinized and studied, perhaps almost as NARROWLY as a man with a Microscope might scrutinize the transient creatures that swarm and multiply in a drop of water. YET across an immense ethereal gulf, minds that to our minds as ours are to the beasts in the jungle, intellects vast, cool and unsympathetic, regarded this earth with envious eyes and slowly and surely drew their plans against us. [sighs] In the thirty-ninth year of the twentieth century came the great disillusionment."

[Play resulting audio](./docs/wow.mp3)

<p align="center">
<img src="./docs/tts.png" width="500"></img>
<img src="./docs/clonevoice.png" width="500"></img>
</p>

### Installation

- `git clone https://github.com/C0untFloyd/bark-gui`
- `pip install .`
- `pip install gradio`

- (optional for audio playback) `pip install soundfile` 
- (optional) Install CUDA for much faster generation 


### Usage

- Linux `python webui.py`
- Windows Use the `StartBark.bat`

Resulting WAVs can be played in the GUI and will additionally be saved
in the `bark-gui\outputs` Folder


Check out the original [Bark](https://github.com/suno-ai/bark) repository for more information.
