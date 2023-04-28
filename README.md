# Bark Web UI

This is a simple Web UI for an extended Bark Version using Gradio, meant to be run locally.

It actually is some kind of Frankenstein-Bark with the
original Code as base and various changes/improvements I liked, ripped and improved from
these 2 repos:
- https://github.com/serp-ai/bark-with-voice-clone
- https://github.com/makawy7/bark-webui (inspired me to even start this)


### Additional Features

- Web GUI
- Creation of very large text passages in chunks, combining the parts into a final result
- Creation of new Voices possible (so far very bad results, hopefully improving in the future)
- Easy Selection of Small/Big Models
- Works with 8Gb GPU or force it to use your CPU instead

Example Input:

> Hello, I am called BARK and am a new text to audio model made by SUNO! Let me read an excerpt from War of the Worlds to you. [clears throat]
We know NOW that in the early years of the twentieth century, this world was being watched closely by intelligences greater than man�s and yet as mortal as his own. We know NOW that as human beings busied themselves about their various concerns they were scrutinized and studied, perhaps almost as NARROWLY as a man with a Microscope might scrutinize the transient creatures that swarm and multiply in a drop of water. YET across an immense ethereal gulf, minds that to our minds as ours are to the beasts in the jungle, intellects vast, cool and unsympathetic, regarded this earth with envious eyes and slowly and surely drew their plans against us. [sighs] In the thirty-ninth year of the twentieth century came the great disillusionment.

[Play resulting audio](https://user-images.githubusercontent.com/131583554/234322414-c90330e4-bbe8-4047-bea6-1f783d49204f.webm)


### Installation

- `git clone https://github.com/C0untFloyd/bark-gui`
- `pip install .`
- `pip install gradio`

- (optional for audio playback) `pip install soundfile` 
- (optional) Install CUDA for much faster generation 



### Usage

- Linux `python webui.py (optional arguments)`
- Windows Use the `StartBark.bat`

#### Commandline Arguments:

- -autolaunch Automatically open Browser with Bark-Tab
- -smallmodels Use small models, for GPUs with less than 10Gb Vram or to speed up process
- -forcecpu Force processing on CPU, if your GPU isn't up to the task
- -enablemps Support for Apple MPS
- -offloadcpu Offload models to CPU

On Windows edit the StartBark.bat to your likings.

#### Text-to-Speech Tab

Input any text to let Bark create a Speech, use the Dropbox to choose any voice from
the prompts folder (also custom ones). Choose 'None' for undefined (usefull for MAN:/WOMAN: prompts).
The `Quick Generation` checkbox creates audio a lot faster but might be more unstable and perhaps not that subtle
as this doesn't use finetuning parameters.
Checking `Use last generation as history` saves voices of each audio chunk to the outputs folder. If you want
to use them for output, just copy them into the assets/prompts folder.
Contrary to the original Bark, you can input any text length. The result will be created in chunks
and merged into 1 audio-file at the end. This can be played from the UI and the WAV-File is saved
into the Outputs folder.
<p>
<img src="./docs/tts.png" width="600"></img>
</P>

#### Clone Voice Tab

It's possible to clone your voice, although results so far aren't good.
Input a WAV Audio File with your sampled voice at the top. Below that, input your spoken
words. The path and filename text can be adjusted as you like, the default is the path to
the folder Bark is using for its voices. By clicking "Create" the process is started.
Currently there is only user feedback in the console output.

<p>
<img src="./docs/clonevoice.png" width="600"></img>
</p>

### FAQ

**Q:** Why are there voice changes in the resulting audio files?

Because (from my limited understanding) this is a similar stochastic model as other GPT-Style Models, where each output is based on a previous one.
This has a pretty good entertainment value but is of course suboptimal for plain TTS. Over time there surely will be a solution for stable generation.
Also from my experience, using the special tags like [sighs] etc. at the start/stop of a sentence seem to confuse the model. This seems to
be especially true for languages other than english. If you're not satisfied with the result, just try again and hope for the best.

**Q:** Why are cloned voices so bad?

Probably because the code for cloning is a lot of guesswork so far and was implemented by the authors of the fork
I mentioned at the top. The original Bark authors don't condone voice cloning and I think it will take some time for the
community to find 'the secret sauce'. Best results so far seem to use a very short input clip of 2-4 seconds.

**Q:** Why did you hack this together, stealing from various sources? When will you implement feature xxx?

Although I'm an experienced programmer, this is my first step into python programming and ML and serves me basically as a learning project.
I've been excited by the Bark Release and as a windows-centric user wanted something simple to run locally, without
caring about colabs, notebooks and the likes. Basically what automatic1111 did for Stable Diffusion. The many repos I checked
had something cool, which the others were missing. So I'm doing this basically for myself but I'm glad if you enjoy my experiments too.
