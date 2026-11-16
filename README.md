# Jarvis

A fast, fully local companion and assistant. It runs on your own computer with
[Ollama](https://ollama.com): nothing you say leaves the machine.

The goal is a voice assistant that answers quickly, remembers you, and can do
things for you. It is being built in small, working steps.

## Status

| Step | What | State |
|---|---|---|
| 1 | Text chat with a fast local model and a companion personality | Done |
| 2 | Hearing: push-to-talk and speech recognition | Done |
| 3 | Voice: natural speech, streamed sentence by sentence | Done |
| 4 | Memory: remembers facts about you between conversations | Planned |
| 5 | Tasks: time, weather, apps, screenshots | Planned |
| 6 | Email and calendar | Planned |

## Requirements

- Python 3.11 or newer
- [Ollama](https://ollama.com) running locally, with a chat model pulled:

```
ollama pull llama3.2:3b
```

## Run it

```
python -m jarvis
```

No packages need to be installed for the text chat; it uses only the Python
standard library.

```
Jarvis is ready (model: llama3.2:3b). Commands: /reset forgets this conversation, /quit exits.

You: Tell me a quick joke.
Jarvis: Why did the programmer quit his job? Because he didn't get arrays.
```

On a laptop RTX 4050 the first word arrives in about 0.1 seconds once the
model is loaded. The very first reply after a break takes around 10 seconds
while Ollama loads the model into memory.

## Talk to it

Install the voice packages once, then start in voice mode:

```
pip install -e ".[voice]"
python -m jarvis --voice
```

Hold **Right Ctrl**, speak, and release. Your words are transcribed by
[faster-whisper](https://github.com/SYSTRAN/faster-whisper) and sent to the
model. The reply is printed and spoken aloud with
[Piper](https://github.com/OHF-Voice/piper1-gpl), one sentence at a time, so
it starts talking before the whole reply has been written. Press the talk key
while it is speaking to interrupt it.

```
Jarvis is listening (speech recognition on cuda, voice en_GB-alan-medium). Hold ctrl_r to talk, Ctrl+C to quit.

You: Hey Jarvis, remind me to call my friend tomorrow at five.   (0.3s)
Jarvis: ...
```

The first run downloads the speech recognition model (about 480 MB) and the
voice (about 60 MB); after that both load from disk and need no network. On a
laptop RTX 4050 a short sentence is transcribed in about 0.3 seconds and the
reply starts playing about a second after you stop talking. Without an NVIDIA
GPU, speech recognition falls back to the CPU.

No Right Ctrl key? Set `push_to_talk_key` to another key such as `f9`. To
change the voice, set `tts_voice` to any name from the
[Piper voice samples](https://rhasspy.github.io/piper-samples/); set
`speak = false` for text replies only.

The talk key is watched system-wide while voice mode runs, so it works even
when another window is in front. Only that one key is checked.

## Settings

Copy `config.example.toml` to `config.toml` and edit it. The most useful
setting is your name:

```toml
user_name = "Sam"
```

Any setting can also be given as an environment variable, for example
`JARVIS_MODEL=qwen2.5:7b`. `config.toml` is ignored by git, so personal
details stay on your machine.

## Development

```
pip install -e ".[dev]"
pytest
```

## How it is organised

| File | Purpose |
|---|---|
| `jarvis/llm.py` | Streaming client for Ollama |
| `jarvis/chat.py` | Conversation history and reply streaming |
| `jarvis/persona.py` | The assistant's personality |
| `jarvis/config.py` | Settings from `config.toml` and the environment |
| `jarvis/stt.py` | Speech to text (faster-whisper, GPU with CPU fallback) |
| `jarvis/audio.py` | Microphone recording |
| `jarvis/ptt.py` | Push-to-talk key handling |
| `jarvis/sentences.py` | Cuts the streamed reply into sentences |
| `jarvis/tts.py` | Text to speech (Piper) with background, interruptible playback |
| `jarvis/voice.py` | The voice loop |
| `jarvis/__main__.py` | Command line entry point and text chat |

## License

MIT. The optional voice packages have their own licenses; in particular Piper
(`piper-tts`) is GPL-3.0.
