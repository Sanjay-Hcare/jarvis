# Jarvis

A fast, fully local companion and assistant. It runs on your own computer with
[Ollama](https://ollama.com): nothing you say leaves the machine.

The goal is a voice assistant that answers quickly, remembers you, and can do
things for you. It is being built in small, working steps.

## Status

| Step | What | State |
|---|---|---|
| 1 | Text chat with a fast local model and a companion personality | Done |
| 2 | Hearing: push-to-talk and speech recognition | Planned |
| 3 | Voice: natural speech, streamed sentence by sentence | Planned |
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
| `jarvis/__main__.py` | The text chat you run |

## License

MIT
