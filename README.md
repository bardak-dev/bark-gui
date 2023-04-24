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
