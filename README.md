# Jarvis

A fast voice companion that runs entirely on your own computer. You hold a
key and talk; it answers out loud in about a second, remembers what you tell
it about yourself, and can do a few things for you.

It uses [Ollama](https://ollama.com) for the language model,
[faster-whisper](https://github.com/SYSTRAN/faster-whisper) to hear you and
[Piper](https://github.com/OHF-Voice/piper1-gpl) to speak. No account, no API
key, and no audio or conversation leaves the machine.

## What it looks like

A real session (typed here; in voice mode the same lines are spoken):

```
You: Hi Jarvis. I'm learning Python and my sister's name is Priya.
Jarvis: I'm happy to help you with Python, Sam. Now, about your sister Priya - that's a great name. Does Priya know you're learning Python?
[remembered: Sam is learning Python.]
[remembered: Sam's sister is named Priya.]

You: What time is it?
[get_time: It is 1:16 AM on Tuesday, 17 November 2026.]
Jarvis: It is 1:16 AM on Tuesday, 17 November 2026.

You: What's the weather like in Mumbai?
[get_weather: Mumbai, India: clear sky, 25°C (feels like 26°C), wind 3 km/h. Today: high 33°C, low 22°C, 0% chance of rain.]
Jarvis: Mumbai, India is currently clear with a temperature of 25°C, feeling like 26°C, and a gentle wind of 3 km/h. The forecast shows a high of 33°C and a low of 22°C today, with no chance of rain.

You: Send an email to my boss.
Jarvis: I can't send an email yet, Sam.
```

After closing it and starting it again:

```
You: What's my sister called?
Jarvis: Your sister's name is Priya.
```

Lines in square brackets are printed by the program, not written by the
model: they show what was really looked up, done or remembered.

## What it can do

| | |
|---|---|
| **Talk** | Push-to-talk; replies are spoken sentence by sentence as they are generated, and you can interrupt by pressing the key again |
| **Remember** | Learns lasting facts about you and uses them in later conversations |
| **Tell you things** | Time and date, weather for any city |
| **Do things** | Open apps and websites, change the volume, take a screenshot, lock the screen |
| **Check on your day** | Unread email and calendar events (read-only) |

Measured on a laptop with an RTX 4050 (6 GB): the first word of a reply
arrives about 0.1 seconds after the question, a sentence is transcribed in
about 0.3 seconds, and in voice mode the reply starts playing roughly one
second after you stop talking. Without an NVIDIA GPU everything still works,
with slower speech recognition.

## Quick start

You need Python 3.11 or newer and [Ollama](https://ollama.com).

```
ollama pull llama3.2:3b
git clone https://github.com/Sanjay-Hcare/jarvis
cd jarvis
python -m jarvis
```

That is the text chat, which needs no packages at all. For voice:

```
pip install -e ".[all]"
python -m jarvis --voice
```

Hold **Right Ctrl**, speak, release. The first run downloads the speech
recognition model (about 480 MB) and the voice (about 60 MB); after that they
load from disk and no network is needed.

On Windows you can also double-click `jarvis.bat`, which starts Ollama if
needed and opens voice mode (`jarvis --text` for typing).

Tell it your name by copying `config.example.toml` to `config.toml` and
setting `user_name`.

## Voice

- **Hands-free (new).** `python -m jarvis --wake`, or `wake_word = true` in
  `config.toml`: say "Hey Jarvis" and then your request, in one go or after
  the beep. Saying it while Jarvis is talking interrupts it. Two checks keep
  false wake-ups down: [openWakeWord](https://github.com/dscripka/openWakeWord)
  listens for something that sounds like "Hey Jarvis", then the words are
  transcribed and Jarvis only answers if its name was really said at the
  start ("Hey Travis" and "I think Jarvis is a great name" are ignored). The
  microphone stays open in this mode, but nothing is recorded or kept until
  the wake word is heard, and nothing leaves the computer.
- **Talk key.** `push_to_talk_key` in `config.toml`; use `f9` or `alt_r` if
  your keyboard has no Right Ctrl. The key is watched system-wide while voice
  mode runs, so it works when another window is in front. Only that one key
  is checked.
- **Voice.** `tts_voice` takes any name from the
  [Piper voice samples](https://rhasspy.github.io/piper-samples/);
  `tts_speed = 1.1` talks a little faster; `speak = false` gives text replies.
- **Accuracy.** `stt_model` can be `tiny`, `base`, `small` (default) or
  `medium`. Bigger is more accurate and slower. `stt_language = ""` lets it
  detect the language instead of assuming English.

## Memory

After each reply, a background call asks the model whether your message
stated a lasting fact about you. Two guards keep the model from writing its
own guesses into memory: questions and requests are never examined, and a
fact is kept only if its words actually appear in what you said.

Facts live in `data/memory.json` on your computer, as plain JSON you can read
and edit. They are added to the system prompt along with an instruction never
to claim knowledge about you beyond them, which is why it answers "I don't
know that yet" rather than inventing a dog's name.

In text mode, `/memory` lists what it knows, `/remember <fact>` adds a fact
and `/forget <words>` removes matching ones. `remember = false` turns memory
off.

## Tasks

Ask in your own words:

| Say something like | What happens |
|---|---|
| "What time is it?" / "What's the date?" | Reads your computer's clock |
| "What's the weather in Mumbai?" | Current conditions and today's forecast from [Open-Meteo](https://open-meteo.com) |
| "Open Chrome" / "Can you launch Spotify?" | Opens an app (see the list below) |
| "Open YouTube" / "Go to GitHub" | Opens a website in your browser |
| "Turn the volume up" / "Make it quieter" / "Set the volume to 30" / "Mute" | Steps the volume by 10%, sets a level, or mutes |
| "Take a screenshot" | Saves an image to `Pictures/Jarvis` |
| "Lock my computer" | Locks the screen |
| "Any new mail?" | Counts unread mail from the last two days and names the newest senders and subjects |
| "What's on my calendar today?" / "Am I free this week?" | Lists your events |

Apps, volume, screenshots and locking work on Windows. Apps come from a fixed
list, so nothing the model says is ever run as a command:

- **Always:** calculator, camera, clock, command prompt, file explorer,
  notepad, paint, photos, settings, task manager.
- **If installed:** Chrome, Edge, Firefox, Word, Excel, PowerPoint, Outlook,
  VS Code, VLC, Spotify, Teams. Jarvis checks at start-up and only offers the
  ones it finds.
- **Websites:** browser (Google), YouTube, Gmail, Google Calendar, GitHub,
  WhatsApp Web.

Actions only happen when you ask for them: the sentence has to start with
the request, optionally after "Jarvis", "please", "can you" or "I need you
to". "Open Chrome" opens Chrome; "my exam is going to open with a hard
question" is just conversation.

### Email and calendar

These are read-only: nothing can be sent, deleted or changed.

1. Add `email_address = "you@gmail.com"` to `config.toml`.
2. Run `python -m jarvis.setup_accounts`. It asks for an email app password
   and walks you through connecting Google Calendar with the small script in
   `docs/calendar_script.gs`.

Passwords and the calendar link go into your system's credential store
(Windows Credential Manager, macOS Keychain), never into a file here. The
inbox is opened read-only and only sender and subject headers are fetched, so
nothing is marked as read. For Gmail, only the Primary tab is counted.

## Making a small model dependable

A 3-billion-parameter model is fast enough for conversation but unreliable in
ways that matter for an assistant. Each of these was observed and then fixed
in code rather than by asking the model nicely:

| The model would... | So the program... |
|---|---|
| Call a tool for anything ("tell me a joke" checked the time) | Offers a tool only when the message contains a matching phrase; the model picks which tool and its arguments |
| Act on chat ("I start college tomorrow" opened Google Calendar) | Offers actions only when a sentence is phrased as a request |
| Jump the volume ("quieter" arrived as down *and* 50%) | Ignores a level sent with "up" or "down" |
| Answer differently to the same request | Chooses tools at temperature 0, and re-asks with the "nothing was done" note if it declines |
| Claim actions it never took ("Email sent to Arjun.") | Attaches a note to the message, when no tool ran, saying nothing was done |
| Change the subject after a lookup | Attaches the result to the question with an instruction to report it |
| Embellish confirmations ("the window is 600x400 pixels") | Confirms actions in fixed words, with no second model call |
| Turn a failure into good news (a mailbox error became "No emails.") | Reads errors out word for word |
| Store its own guesses as facts about you | Only examines statements, and requires a fact's words to appear in what you said |

These are strong nudges and hard rules where a hard rule was possible. They
are not a guarantee: the model can still add a detail that is not in a result,
so when it matters, trust the bracketed line.

## Privacy

Everything runs locally except two lookups, which happen only when you ask:
the weather sends the city name to Open-Meteo, and email and calendar connect
to your own mail and Google accounts. Your settings, memory, voices and
credentials are never part of the repository (`config.toml` and `data/` are
ignored by git).

## Settings

Every setting, with its default, is listed in `config.example.toml`. Any of
them can also be given as an environment variable, for example
`JARVIS_MODEL=qwen2.5:7b`.

## Development

```
pip install -e ".[dev]"
pytest
```

The tests use stand-ins for the model, microphone, speakers, mail server and
calendar, so they run anywhere in under a second.

| File | Purpose |
|---|---|
| `jarvis/__main__.py` | Command line entry point and text chat |
| `jarvis/chat.py` | Conversation history, tool use and reply streaming |
| `jarvis/llm.py` | Streaming client for Ollama |
| `jarvis/persona.py` | Personality, and the notes attached to messages |
| `jarvis/memory.py` | Fact store, fact extraction and background learning |
| `jarvis/tools.py` | Tool definitions and the rules for offering them |
| `jarvis/tasks.py` | The built-in tools |
| `jarvis/mail.py` | Read-only inbox summary over IMAP |
| `jarvis/gcalendar.py` | Calendar lookups through an Apps Script web app |
| `jarvis/setup_accounts.py` | Stores email and calendar credentials |
| `jarvis/voice.py` | The voice loop |
| `jarvis/stt.py` | Speech to text (GPU with CPU fallback) |
| `jarvis/audio.py` | Microphone recording |
| `jarvis/ptt.py` | Push-to-talk key handling |
| `jarvis/wake.py` | "Hey Jarvis" detection and recording the request that follows |
| `jarvis/sentences.py` | Cuts the streamed reply into sentences |
| `jarvis/tts.py` | Text to speech with background, interruptible playback |
| `jarvis/config.py` | Settings from `config.toml` and the environment |
| `jarvis/system.py` | Start-up warnings when on battery or low on memory |

## License

MIT. The optional packages have their own licenses; in particular Piper
(`piper-tts`) is GPL-3.0.
