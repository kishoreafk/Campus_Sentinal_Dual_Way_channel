"""State machine for action event confirmation.

Per (camera_id, track_key, action_label) the machine transitions:
    none -> candidate -> confirmed -> alert
        -> none (when score drops below reset threshold for N frames)

Transition rules (configurable):
  none -> candidate      : score >= candidate_score for candidate_frames
  candidate -> confirmed : score >= confirmed_score for confirmed_frames
  confirmed -> alert     : score >= alert_score for alert_frames
  any -> none            : score < reset_score for reset_frames
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Tuple

from src.common.logger import get_logger
from src.common.metrics import STATE_MACHINE_TRANSITIONS

logger = get_logger()


class State(str, Enum):
    NONE = "none"
    CANDIDATE = "candidate"
    CONFIRMED = "confirmed"
    ALERT = "alert"


@dataclass
class _Tracker:
    state: State = State.NONE
    consec_high: int = 0  # frames at or above current required score
    consec_low: int = 0   # frames below reset_score


class ActionStateMachine:
    """Per-(camera, track, action) state machine."""

    def __init__(
        self,
        candidate_frames: int = 5,
        confirmed_frames: int = 10,
        alert_frames: int = 15,
        candidate_score: float = 0.5,
        confirmed_score: float = 0.7,
        alert_score: float = 0.8,
        reset_score: float = 0.3,
        reset_frames: int = 5,
    ):
        self.candidate_frames = candidate_frames
        self.confirmed_frames = confirmed_frames
        self.alert_frames = alert_frames
        self.candidate_score = candidate_score
        self.confirmed_score = confirmed_score
        self.alert_score = alert_score
        self.reset_score = reset_score
        self.reset_frames = reset_frames
        # key: (camera_id, track_key, action_label)
        self._trackers: Dict[Tuple[str, str, str], _Tracker] = defaultdict(_Tracker)

    def update(
        self,
        camera_id: str,
        track_key: str,
        action_label: str,
        score: float,
    ) -> State:
        """Process a new (smoothed) score and return the resulting state."""
        # Skip "none" label — it's the absence of action
        if action_label == "none":
            return State.NONE

        key = (camera_id, track_key, action_label)
        trk = self._trackers[key]

        # Decay logic
        if score < self.reset_score:
            trk.consec_low += 1
            trk.consec_high = 0
            if trk.consec_low >= self.reset_frames:
                if trk.state != State.NONE:
                    self._transition(trk, State.NONE, key)
                trk.consec_low = 0
            return trk.state

        trk.consec_low = 0
        # Required score depends on current state
        if trk.state == State.NONE:
            required_score = self.candidate_score
            required_frames = self.candidate_frames
            next_state = State.CANDIDATE
        elif trk.state == State.CANDIDATE:
            required_score = self.confirmed_score
            required_frames = self.confirmed_frames
            next_state = State.CONFIRMED
        elif trk.state == State.CONFIRMED:
            required_score = self.alert_score
            required_frames = self.alert_frames
            next_state = State.ALERT
        else:  # ALERT
            return trk.state  # Stay in alert

        if score >= required_score:
            trk.consec_high += 1
            if trk.consec_high >= required_frames:
                self._transition(trk, next_state, key)
                trk.consec_high = 0
        else:
            trk.consec_high = 0

        return trk.state

    def _transition(self, trk: _Tracker, new_state: State, key) -> None:
        old = trk.state
        if old == new_state:
            return
        trk.state = new_state
        STATE_MACHINE_TRANSITIONS.labels(old.value, new_state.value).inc()
        logger.debug(
            f"[{key[0]}] {key[1]} action={key[2]} state: {old.value} -> {new_state.value}"
        )

    def get_state(self, camera_id: str, track_key: str, action_label: str) -> State:
        return self._trackers.get(
            (camera_id, track_key, action_label), _Tracker()
        ).state

    def reset(self, camera_id: str, track_key: str) -> None:
        keys_to_remove = [k for k in self._trackers
                          if k[0] == camera_id and k[1] == track_key]
        for k in keys_to_remove:
            del self._trackers[k]
