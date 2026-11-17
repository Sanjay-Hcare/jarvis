import json
from datetime import datetime

import pytest

from jarvis.chat import Conversation
from jarvis.llm import iter_content
from jarvis.tasks import (
    WEBSITES,
    available_apps,
    default_tools,
    get_time,
    get_weather,
    lock_computer,
    open_app,
    set_volume,
    take_screenshot,
)
from jarvis.tools import Tool, ToolBox, ToolCall


def names(tools):
    return [tool.name for tool in tools]


@pytest.mark.parametrize(
    "message, expected",
    [
        ("What time is it?", ["get_time"]),
        ("what's the date today", ["get_time"]),
        ("What day is it?", ["get_time"]),
        ("What's the weather like in Bangalore?", ["get_weather"]),
        ("Is it going to rain in Mumbai?", ["get_weather"]),
        ("Open notepad for me.", ["open_app"]),
        ("launch the calculator", ["open_app"]),
        ("Take a screenshot.", ["take_screenshot"]),
        ("capture my screen please", ["take_screenshot"]),
        ("Lock my computer.", ["lock_computer"]),
        ("lock the screen", ["lock_computer"]),
    ],
)
def test_requests_offer_the_matching_tool(message, expected):
    assert names(ToolBox(default_tools()).relevant(message)) == expected


@pytest.mark.parametrize(
    "message",
    [
        "How are you doing today?",
        "Tell me a joke.",
        "What is the capital of Japan?",
        "I had a long day, I'm tired.",
        "Explain what a neural network is in one sentence.",
        "Thanks, that's all.",
        "Good morning Jarvis",
        "What's two plus two?",
        "I need to unlock my potential",
        "It was a good time",
        # Action words in ordinary sentences must not trigger actions.
        "I start college tomorrow.",
        "My exam is going to open with a hard question.",
        "The volume of work is crazy this week.",
        "I forgot to lock my computer yesterday.",
        "I took a screenshot of it earlier.",
        "The party was louder than I expected.",
        "Can you tell me how to open a bank account?",
    ],
)
def test_ordinary_chat_offers_no_tools(message):
    assert ToolBox(default_tools()).relevant(message) == []


def test_get_time_is_readable():
    assert get_time(lambda: datetime(2026, 3, 5, 9, 7)) == "It is 9:07 AM on Thursday, 5 March 2026."


def fake_weather_fetch(url):
    if "geocoding" in url:
        if "Nowhere" in url:
            return {}
        return {"results": [{"name": "Pune", "country": "India", "latitude": 18.5, "longitude": 73.8}]}
    return {
        "current": {"temperature_2m": 27.4, "apparent_temperature": 29.6, "weather_code": 2, "wind_speed_10m": 9.2},
        "daily": {
            "temperature_2m_max": [31.2],
            "temperature_2m_min": [19.8],
            "precipitation_probability_max": [20],
        },
    }


def test_weather_report():
    report = get_weather("Pune", fetch=fake_weather_fetch)

    assert report == (
        "Pune, India: partly cloudy, 27°C (feels like 30°C), wind 9 km/h. "
        "Today: high 31°C, low 20°C, 20% chance of rain."
    )


def test_weather_uses_home_city_and_reports_problems():
    assert get_weather("", home_city="Pune", fetch=fake_weather_fetch).startswith("Pune, India")
    assert get_weather("", fetch=fake_weather_fetch).startswith("Error: no city")
    assert get_weather("Nowhere", fetch=fake_weather_fetch) == "Error: no place called Nowhere was found."


SOME_APPS = {"notepad": ["notepad.exe"], "chrome": "chrome.exe", "youtube": "https://www.youtube.com"}


def test_open_app_only_launches_listed_apps():
    launched = []

    assert open_app("Notepad", SOME_APPS, launch=launched.append) == "Opened notepad."
    assert open_app("  Google   Chrome ", SOME_APPS, launch=launched.append) == "Opened chrome."
    assert open_app("YouTube", SOME_APPS, launch=launched.append) == "Opened youtube."
    assert open_app("format c:", SOME_APPS, launch=launched.append) == (
        "Error: format c: is not one of the apps I can open."
    )
    assert launched == [["notepad.exe"], "chrome.exe", "https://www.youtube.com"]


def test_only_installed_programs_are_offered(monkeypatch):
    monkeypatch.setattr("jarvis.tasks.os.name", "nt")

    apps = available_apps(is_installed=lambda exe: exe in ("chrome.exe", "code.exe"))

    assert "chrome" in apps and "vs code" in apps
    assert "firefox" not in apps and "word" not in apps
    assert {"notepad", "settings", "youtube", "browser"} <= set(apps)
    assert list(apps) == sorted(apps)


def test_websites_are_offered_everywhere(monkeypatch):
    monkeypatch.setattr("jarvis.tasks.os.name", "posix")

    apps = available_apps(is_installed=lambda exe: True)

    assert set(apps) == set(WEBSITES)


def test_the_model_is_only_offered_apps_that_exist():
    tool = next(t for t in default_tools(apps=SOME_APPS) if t.name == "open_app")

    assert tool.schema()["function"]["parameters"]["properties"]["app"]["enum"] == list(SOME_APPS)


class FakeMixer:
    def __init__(self, level=40, muted=False):
        self.level = level
        self.is_muted = muted

    def get(self):
        return self.level

    def set(self, percent):
        self.level = percent

    def muted(self):
        return self.is_muted

    def mute(self, on):
        self.is_muted = on


@pytest.mark.parametrize(
    "change, level, start, expected, final",
    [
        ("up", None, 40, "Volume is now 50%.", 50),
        ("louder", None, 95, "Volume is now 100%.", 100),
        ("down", None, 40, "Volume is now 30%.", 30),
        ("down", None, 5, "Volume is now 0%.", 0),
        ("set", 75, 40, "Volume is now 75%.", 75),
        ("", "20%", 40, "Volume is now 20%.", 20),
        ("set", 150, 40, "Volume is now 100%.", 100),
        # A level sent along with up or down is ignored: it is a step, not a jump.
        ("down", 50, 40, "Volume is now 30%.", 30),
        ("down", 0, 40, "Volume is now 30%.", 30),
        ("up", 90, 40, "Volume is now 50%.", 50),
    ],
)
def test_volume_changes(change, level, start, expected, final):
    mixer = FakeMixer(start)

    assert set_volume(change, level, mixer=mixer) == expected
    assert mixer.level == final


def test_mute_and_unmute():
    mixer = FakeMixer(40)

    assert set_volume("mute", mixer=mixer) == "Muted."
    assert mixer.is_muted and mixer.level == 40
    assert set_volume("unmute", mixer=mixer) == "Volume is now 40%."
    assert not mixer.is_muted


def test_turning_up_also_unmutes_but_turning_down_does_not():
    up, down = FakeMixer(40, muted=True), FakeMixer(40, muted=True)

    set_volume("up", mixer=up)
    set_volume("down", mixer=down)

    assert not up.is_muted
    assert down.is_muted


def test_unclear_volume_requests_are_errors():
    mixer = FakeMixer(40)

    assert set_volume("", mixer=mixer).startswith("Error: say whether")
    assert set_volume("set", mixer=mixer).startswith("Error: say whether")
    assert set_volume("set", "loud", mixer=mixer) == "Error: loud is not a volume level."
    assert mixer.level == 40


@pytest.mark.parametrize(
    "message",
    [
        "Turn the volume up",
        "make it louder",
        "mute",
        "unmute the sound",
        "set volume to 30",
        "turn the music down",
        "Volume up please",
        "Can you make it louder?",
        "turn it down",
        "Hey Jarvis, mute.",
    ],
)
def test_volume_requests_offer_the_volume_tool(message):
    assert names(ToolBox(default_tools(apps=SOME_APPS)).relevant(message)) == ["set_volume"]


@pytest.mark.parametrize(
    "message, expected",
    [
        ("Can you open chrome?", ["open_app"]),
        ("Jarvis, open YouTube.", ["open_app"]),
        ("Hey Jarvis. Please launch Spotify", ["open_app"]),
        ("I need you to open the calculator", ["open_app"]),
        ("Thanks! Now lock my laptop.", ["lock_computer"]),
        ("Could you please take a screenshot", ["take_screenshot"]),
        ("go ahead and lock it", ["lock_computer"]),
    ],
)
def test_requests_are_recognised_after_polite_openings(message, expected):
    assert names(ToolBox(default_tools(apps=SOME_APPS)).relevant(message)) == expected


def test_confirmation_can_quote_the_result():
    client = ScriptedClient([ToolCall("vol", {"change": "up"})])
    box = ToolBox(
        [
            Tool(
                "vol",
                "Volume.",
                lambda change="": "Volume is now 50%.",
                {"type": "object", "properties": {"change": {}}},
                triggers="volume",
                confirmation="{result}",
            )
        ]
    )

    assert "".join(Conversation(client, "sys", toolbox=box).ask("volume up")) == "Volume is now 50%."


def test_screenshot_is_saved_with_a_timestamp(tmp_path):
    class Image:
        def save(self, path):
            path.write_bytes(b"png")

    message = take_screenshot(str(tmp_path), grab=Image, now=lambda: datetime(2026, 3, 5, 9, 7, 30))

    saved = tmp_path / "screenshot-20260305-090730.png"
    assert saved.read_bytes() == b"png"
    assert str(saved) in message


def test_lock_calls_the_system_lock():
    calls = []

    assert lock_computer(lock=lambda: calls.append("locked")) == "The computer is locked."
    assert calls == ["locked"]


def test_toolbox_runs_tools_and_reports_failures():
    def boom():
        raise RuntimeError("no network")

    box = ToolBox(
        [
            Tool("echo", "", lambda word="": f"said {word}", {"type": "object", "properties": {"word": {}}}),
            Tool("boom", "", boom),
        ]
    )

    assert box.call(ToolCall("echo", {"word": "hi", "unexpected": 1})) == "said hi"
    assert box.call(ToolCall("boom")) == "Error: no network"
    assert box.call(ToolCall("missing")) == "Error: there is no tool called missing."


def test_stream_parsing_yields_tool_calls():
    lines = [
        json.dumps(
            {"message": {"content": "", "tool_calls": [{"function": {"name": "get_weather", "arguments": {"city": "Pune"}}}]}}
        ).encode(),
        json.dumps({"message": {"content": ""}, "done": True}).encode(),
    ]

    assert list(iter_content(lines)) == [ToolCall("get_weather", {"city": "Pune"})]


class ScriptedClient:
    """Returns one scripted response per call and records what it was sent."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def stream_chat(self, messages, temperature=0.7, tools=None):
        self.calls.append({"messages": messages, "tools": tools})
        yield from self.responses.pop(0)


def toolbox_with(run):
    return ToolBox([Tool("get_time", "Get the time.", run, triggers=r"\btime\b")])


def test_tool_is_run_and_its_result_given_to_the_model():
    client = ScriptedClient([ToolCall("get_time")], ["It is ", "nine."])
    announced = []
    chat = Conversation(
        client,
        "sys",
        toolbox=toolbox_with(lambda: "It is 9:00 AM."),
        on_tool=lambda call, result: announced.append((call.name, result)),
        tool_result_note="report it",
    )

    assert "".join(chat.ask("what time is it")) == "It is nine."
    assert announced == [("get_time", "It is 9:00 AM.")]
    first, second = client.calls
    assert [tool["function"]["name"] for tool in first["tools"]] == ["get_time"]
    assert second["tools"] is None
    # The result travels with the question, followed by the instruction to report it.
    assert second["messages"][-1] == {
        "role": "user",
        "content": "what time is it\n\n[Result of get_time: It is 9:00 AM.]\n\n[report it]",
    }
    assert [m["role"] for m in second["messages"]] == ["system", "user"]
    # Only the visible exchange is kept as history.
    assert chat.history == [
        {"role": "user", "content": "what time is it"},
        {"role": "assistant", "content": "It is nine."},
    ]


def test_no_tools_are_offered_for_ordinary_chat():
    client = ScriptedClient(["Hello."])
    chat = Conversation(client, "sys", toolbox=toolbox_with(lambda: "unused"))

    assert "".join(chat.ask("hello there")) == "Hello."
    assert client.calls[0]["tools"] is None


def test_declining_the_tool_means_asking_again_with_the_no_action_note():
    client = ScriptedClient(["Done, the time is set!"], ["About ", "time you asked."])
    chat = Conversation(client, "sys", toolbox=toolbox_with(lambda: "unused"), no_action_note="nothing was done")

    # The first answer, written without the note, is thrown away.
    assert "".join(chat.ask("it is time to talk")) == "About time you asked."
    first, second = client.calls
    assert first["tools"] and second["tools"] is None
    assert second["messages"][-1]["content"].endswith("[nothing was done]")


def test_tool_choice_is_made_at_temperature_zero():
    temperatures = []

    class Client(ScriptedClient):
        def stream_chat(self, messages, temperature=0.7, tools=None):
            temperatures.append((bool(tools), temperature))
            yield from super().stream_chat(messages, temperature, tools)

    chat = Conversation(Client([ToolCall("get_time")], ["Nine."]), "sys", temperature=0.7, toolbox=toolbox_with(lambda: "9"))
    list(chat.ask("what time is it"))

    assert temperatures == [(True, 0), (False, 0.7)]


def test_tool_call_written_as_text_is_retried_without_tools():
    client = ScriptedClient(['{"name": null, "parameters": {}}'], ["Sure, let's talk."])
    chat = Conversation(client, "sys", toolbox=toolbox_with(lambda: "unused"))

    assert "".join(chat.ask("do you have time")) == "Sure, let's talk."
    assert client.calls[1]["tools"] is None


def test_weather_prefers_the_largest_place_and_knows_old_city_names():
    searched = []

    def fetch(url):
        if "geocoding" in url:
            searched.append(url)
            return {
                "results": [
                    {"name": "Hyderabad", "country": "Pakistan", "latitude": 25.4, "longitude": 68.4, "population": 1921275},
                    {"name": "Hyderabad", "country": "India", "latitude": 17.4, "longitude": 78.5, "population": 6993262},
                    {"name": "Hyderabad", "country": "Pakistan", "latitude": 30.0, "longitude": 71.0, "population": None},
                ]
            }
        return fake_weather_fetch(url)

    assert get_weather("Hyderabad", fetch=fetch).startswith("Hyderabad, India")
    get_weather("Bangalore", fetch=fetch)
    assert "name=Bengaluru" in searched[-1]


def test_fire_up_offers_the_app_tool_but_going_for_a_run_does_not():
    box = ToolBox(default_tools())

    assert names(box.relevant("Fire up paint for me.")) == ["open_app"]
    assert box.relevant("I usually go for a run in the morning") == []


def test_no_action_note_is_sent_but_not_stored_or_used_with_tools():
    client = ScriptedClient(["I can't do that yet."], [ToolCall("get_time")], ["Nine."])
    chat = Conversation(client, "sys", toolbox=toolbox_with(lambda: "It is 9:00 AM."), no_action_note="nothing was done")

    assert "".join(chat.ask("send an email")) == "I can't do that yet."
    assert client.calls[0]["messages"][-1] == {"role": "user", "content": "send an email\n\n[nothing was done]"}
    assert chat.history[0] == {"role": "user", "content": "send an email"}

    list(chat.ask("what time is it"))
    # A turn where a tool ran carries no such note.
    assert all("nothing was done" not in str(m["content"]) for m in client.calls[2]["messages"][-3:])


def test_actions_are_confirmed_in_fixed_words_without_asking_the_model():
    client = ScriptedClient([ToolCall("lock_it")])
    box = ToolBox([Tool("lock_it", "Lock.", lambda: "The computer is locked.", triggers="lock", confirmation="Locked.")])
    chat = Conversation(client, "sys", toolbox=box)

    assert "".join(chat.ask("lock it")) == "Locked."
    assert len(client.calls) == 1


def test_failures_are_reported_word_for_word():
    client = ScriptedClient([ToolCall("check")])
    box = ToolBox([Tool("check", "Check.", lambda: "Error: could not check the mailbox (connection reset).", triggers="mail")])
    chat = Conversation(client, "sys", toolbox=box, tool_result_note="report it")

    assert "".join(chat.ask("any mail")) == "Sorry, that didn't work: could not check the mailbox (connection reset)."
    assert len(client.calls) == 1


def test_a_failed_action_is_not_confirmed():
    def fail():
        raise RuntimeError("access denied")

    client = ScriptedClient([ToolCall("lock_it")])
    box = ToolBox([Tool("lock_it", "Lock.", fail, triggers="lock", confirmation="Locked.")])

    assert "".join(Conversation(client, "sys", toolbox=box).ask("lock it")) == "Sorry, that didn't work: access denied"
