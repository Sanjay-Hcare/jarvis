from jarvis.system import LOW_MEMORY_GB, current_warnings, free_memory_gb, on_battery, performance_warnings


def test_no_warnings_when_plugged_in_with_memory_to_spare():
    assert performance_warnings(battery=False, free_gb=8.0) == []


def test_battery_and_low_memory_are_both_explained():
    warnings = performance_warnings(battery=True, free_gb=1.8)

    assert len(warnings) == 2
    assert warnings[0].startswith("Running on battery") and "Plug in the charger" in warnings[0]
    assert warnings[1].startswith("Only 1.8 GB of memory is free")


def test_unknown_state_gives_no_warnings():
    assert performance_warnings(battery=None, free_gb=None) == []
    assert performance_warnings(battery=False, free_gb=LOW_MEMORY_GB) == []


def test_reading_the_real_state_does_not_fail():
    assert on_battery() in (True, False, None)
    free = free_memory_gb()
    assert free is None or free > 0
    assert isinstance(current_warnings(), list)
