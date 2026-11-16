import json
from datetime import datetime

import pytest

from jarvis.chat import Conversation
from jarvis.llm import iter_content
from jarvis.tasks import APPS, default_tools, get_time, get_weather, lock_computer, open_app, take_screenshot
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


def test_open_app_only_launches_listed_apps():
    launched = []

    assert open_app("Notepad", launch=launched.append) in ("Opened notepad.", "Error: opening apps is only supported on Windows.")
    assert open_app("format c:", launch=launched.append).startswith("Error: I can only open")
    assert open_app("browser", launch=launched.append) == "Opened browser."
    assert all(target in APPS.values() for target in launched)


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
    )

    assert "".join(chat.ask("what time is it")) == "It is nine."
    assert announced == [("get_time", "It is 9:00 AM.")]
    first, second = client.calls
    assert [tool["function"]["name"] for tool in first["tools"]] == ["get_time"]
    assert second["tools"] is None
    assert second["messages"][-1] == {"role": "tool", "content": "It is 9:00 AM."}
    assert second["messages"][-2]["tool_calls"][0]["function"]["name"] == "get_time"
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


def test_plain_answer_when_the_model_declines_the_tool():
    client = ScriptedClient(["About ", "time you asked."])
    chat = Conversation(client, "sys", toolbox=toolbox_with(lambda: "unused"))

    assert "".join(chat.ask("it is time to talk")) == "About time you asked."
    assert len(client.calls) == 1


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
