"""Tests for Layer 5 state machine."""

from __future__ import annotations

import pytest

from src.layer5_postprocess.state_machine import ActionStateMachine, State


def test_state_machine_initial_state():
    sm = ActionStateMachine()
    state = sm.update("cam_1", "tid_1", "fight", 0.8)
    assert state == State.NONE  # not enough frames


def test_state_machine_progression_to_alert():
    """Score high enough for enough frames -> ALERT."""
    sm = ActionStateMachine(
        candidate_frames=2,
        confirmed_frames=2,
        alert_frames=2,
        candidate_score=0.5,
        confirmed_score=0.7,
        alert_score=0.8,
        reset_score=0.3,
        reset_frames=5,
    )
    # 2 frames -> candidate
    assert sm.update("c", "t", "fight", 0.9) == State.NONE
    assert sm.update("c", "t", "fight", 0.9) == State.CANDIDATE
    # 2 more -> confirmed
    assert sm.update("c", "t", "fight", 0.9) == State.CANDIDATE
    assert sm.update("c", "t", "fight", 0.9) == State.CONFIRMED
    # 2 more -> alert
    assert sm.update("c", "t", "fight", 0.9) == State.CONFIRMED
    assert sm.update("c", "t", "fight", 0.9) == State.ALERT
    # Stays at alert
    assert sm.update("c", "t", "fight", 0.9) == State.ALERT


def test_state_machine_low_score_resets():
    sm = ActionStateMachine(
        candidate_frames=2,
        confirmed_frames=2,
        alert_frames=2,
        reset_score=0.3,
        reset_frames=2,
    )
    # Get to candidate
    sm.update("c", "t", "fight", 0.9)
    sm.update("c", "t", "fight", 0.9)
    assert sm.get_state("c", "t", "fight") == State.CANDIDATE
    # Low score for 2 frames -> reset
    sm.update("c", "t", "fight", 0.1)
    sm.update("c", "t", "fight", 0.1)
    assert sm.get_state("c", "t", "fight") == State.NONE


def test_state_machine_skip_none_label():
    sm = ActionStateMachine()
    state = sm.update("c", "t", "none", 0.99)
    assert state == State.NONE


def test_state_machine_independent_per_track():
    sm = ActionStateMachine(candidate_frames=1, confirmed_frames=1, alert_frames=1)
    sm.update("c", "t1", "fight", 0.9)
    assert sm.get_state("c", "t1", "fight") == State.CANDIDATE
    # Different track should be NONE
    assert sm.get_state("c", "t2", "fight") == State.NONE


def test_state_machine_reset_clears_state():
    sm = ActionStateMachine(candidate_frames=1)
    sm.update("c", "t", "fight", 0.9)
    sm.reset("c", "t")
    assert sm.get_state("c", "t", "fight") == State.NONE
