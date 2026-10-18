"""Bounded transport for wuhao21/sts2-cli's actual JSON-lines protocol.

This is a transport, not a simulator, legal-action oracle, or Gym environment.
The upstream protocol has no request IDs or state-query command. After an
uncertain command delivery we therefore kill the worker and discard the episode;
retrying could execute a mutating action twice.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import queue
import signal
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal, Mapping, NoReturn, Sequence, TypedDict, cast


DecisionPhase = Literal[
    "map_select", "combat_play", "card_reward", "event_choice", "rest_site",
    "shop", "game_over", "card_select", "bundle_select",
]


class NativeDecision(TypedDict, total=False):
    """Raw upstream payload; no fields or authority are invented by the adapter."""

    type: Literal["decision"]
    decision: DecisionPhase
    player: dict[str, Any]
    context: dict[str, Any]
    hand: list[dict[str, Any]]
    enemies: list[dict[str, Any]]
    cards: list[dict[str, Any]]
    choices: list[dict[str, Any]]
    options: list[dict[str, Any]]
    bundles: list[dict[str, Any]]
    energy: int
    round: int
    act: int
    floor: int
    min_select: int
    max_select: int
    can_skip: bool
    victory: bool


class AbortCategory(StrEnum):
    STARTUP_FAILURE = "startup_failure"
    STARTUP_TIMEOUT = "startup_timeout"
    TRANSPORT_TIMEOUT = "transport_timeout"
    PROCESS_EXIT = "process_exit"
    PROTOCOL_ERROR = "protocol_error"
    ENGINE_ERROR = "engine_error"
    UNSUPPORTED_STATE = "unsupported_state"
    STALE_ACTION = "stale_action"
    NO_ACTIVE_RUN = "no_active_run"
    SHUTDOWN_FAILURE = "shutdown_failure"


class NativeAbort(RuntimeError):
    """Infrastructure failure, never a game defeat or a training reward.

    ``delivery`` describes only what the transport knows. Even an engine error
    response can follow a partial mutation; it must not be retried in this run.
    """

    def __init__(
        self,
        category: AbortCategory,
        message: str,
        *,
        delivery: Literal["not_sent", "unknown", "response_received"] = "not_sent",
        diagnostics: Sequence[str] = (),
        payload: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.category = category
        self.delivery = delivery
        self.diagnostics = tuple(diagnostics)
        self.payload = copy.deepcopy(payload)

    def as_dict(self) -> dict[str, Any]:
        return {
            "category": self.category.value,
            "message": str(self),
            "delivery": self.delivery,
            "diagnostics": list(self.diagnostics),
            "payload": self.payload,
        }


@dataclass(frozen=True)
class NativeTransition:
    """Local reconciliation record; not an upstream delivery receipt."""

    action: str
    before_revision: int
    after_revision: int
    before_hash: str
    after_hash: str

    @property
    def changed(self) -> bool:
        return self.before_hash != self.after_hash


_PHASES = {
    "map_select", "combat_play", "card_reward", "event_choice", "rest_site",
    "shop", "game_over", "card_select", "bundle_select",
}


def decision_hash(payload: Mapping[str, Any]) -> str:
    """Fingerprint the complete reported state for external cycle detection.

    Equality is only reported-state equality, not equality of hidden RNG or engine
    internals. Different JSON object key order does not change the fingerprint.
    """
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class NativeRunner:
    """One command at a time, cold process per run, bounded pipe IO.

    ``command`` is an argument vector, never a shell string. Call ``reset`` to
    start an episode. Arguments use the native CLI conventions (e.g. target and
    card indices are valid only in the latest returned state).
    """

    def __init__(
        self,
        command: Sequence[str],
        *,
        cwd: str | Path | None = None,
        env: Mapping[str, str] | None = None,
        timeout_s: float = 30.0,
        startup_timeout_s: float = 30.0,
        close_timeout_s: float = 3.0,
        max_line_bytes: int = 2 * 1024 * 1024,
        max_command_bytes: int = 64 * 1024,
        expected_protocol_version: str | None = "0.2.0",
    ) -> None:
        if isinstance(command, str) or not command or not all(isinstance(x, str) and x for x in command):
            raise ValueError("command must be a nonempty sequence of nonempty strings")
        for name, value in (("timeout_s", timeout_s), ("startup_timeout_s", startup_timeout_s), ("close_timeout_s", close_timeout_s)):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 128 for value in (max_line_bytes, max_command_bytes)):
            raise ValueError("pipe byte limits must be at least 128")
        self.command = tuple(command)
        self.cwd = str(cwd) if cwd is not None else None
        self.env = dict(env or {})
        self.timeout_s = timeout_s
        self.startup_timeout_s = startup_timeout_s
        self.close_timeout_s = close_timeout_s
        self.max_line_bytes = max_line_bytes
        self.max_command_bytes = max_command_bytes
        self.expected_protocol_version = expected_protocol_version
        self._process: subprocess.Popen[bytes] | None = None
        self._events: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=64)
        self._overflow = threading.Event()
        self._diagnostics: deque[str] = deque(maxlen=64)
        self._lock = threading.RLock()
        self._current: NativeDecision | None = None
        self._revision = 0
        self.last_transition: NativeTransition | None = None
        self.protocol_version: str | None = None

    @property
    def current(self) -> NativeDecision | None:
        return copy.deepcopy(self._current)

    @property
    def revision(self) -> int:
        return self._revision

    @property
    def diagnostics(self) -> tuple[str, ...]:
        return tuple(self._diagnostics)

    @property
    def pid(self) -> int | None:
        return self._process.pid if self._process else None

    def reset(
        self, seed: str | int, *, character: str = "Ironclad", ascension: int = 0, lang: str = "en"
    ) -> NativeDecision:
        """Cold-start, read ready, then send the native ``start_run`` command."""
        if not isinstance(seed, (str, int)) or isinstance(seed, bool) or not str(seed):
            raise ValueError("seed must be a nonempty string or integer")
        if isinstance(ascension, bool) or not isinstance(ascension, int) or not 0 <= ascension <= 2**31 - 1:
            raise ValueError("ascension must be a nonnegative Int32")
        if not character or not lang:
            raise ValueError("character and lang must not be empty")
        with self._lock:
            self.close()
            self._start()
            payload = self._exchange({"cmd": "start_run", "character": character, "seed": str(seed), "ascension": ascension, "lang": lang})
            return self._accept_decision(payload)

    def act(
        self,
        action: str,
        args: Mapping[str, str | int | bool] | None = None,
        *,
        expected_revision: int | None = None,
    ) -> NativeDecision:
        """Execute once and reconcile with the next returned raw decision.

        A stale local revision is rejected before writing. There is no automatic
        retry, fallback action, or wait-loop issuing extra gameplay commands.
        """
        if not isinstance(action, str) or not action:
            raise ValueError("action must be a nonempty string")
        if args is not None and any(not isinstance(k, str) or not isinstance(v, (str, int, bool)) for k, v in args.items()):
            raise ValueError("native args accept only string, integer, and boolean values; selection indices are a comma-separated string")
        if args is not None and any(isinstance(v, int) and not isinstance(v, bool) and not -(2**31) <= v <= 2**31 - 1 for v in args.values()):
            raise ValueError("native integer args must fit Int32")
        with self._lock:
            if self._current is None or self._current.get("decision") == "game_over":
                raise NativeAbort(AbortCategory.NO_ACTIVE_RUN, "reset is required before acting")
            if expected_revision is not None and expected_revision != self._revision:
                raise NativeAbort(AbortCategory.STALE_ACTION, f"expected revision {expected_revision}, current revision {self._revision}")
            before_revision = self._revision
            before_hash = decision_hash(self._current)
            command: dict[str, Any] = {"cmd": "action", "action": action}
            if args:
                command["args"] = dict(args)
            state = self._accept_decision(self._exchange(command))
            self.last_transition = NativeTransition(action, before_revision, self._revision, before_hash, decision_hash(state))
            return state

    def _start(self) -> None:
        self._events = queue.Queue(maxsize=64)
        self._overflow = threading.Event()
        self._diagnostics.clear()
        self.last_transition = None
        self.protocol_version = None
        process_env = os.environ.copy()
        process_env.update(self.env)
        try:
            self._process = subprocess.Popen(
                self.command, cwd=self.cwd, env=process_env,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                bufsize=0, start_new_session=os.name != "nt",
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
            )
        except OSError as exc:
            raise NativeAbort(AbortCategory.STARTUP_FAILURE, f"cannot start native command: {exc}") from exc
        assert self._process.stdout is not None and self._process.stderr is not None
        threading.Thread(target=self._read_stdout, args=(self._process.stdout, self._events, self._overflow), daemon=True).start()
        threading.Thread(target=self._read_stderr, args=(self._process.stderr,), daemon=True).start()
        ready = self._receive(time.monotonic() + self.startup_timeout_s, startup=True, delivery="not_sent")
        if ready.get("type") != "ready" or not isinstance(ready.get("version"), str):
            self._fail(AbortCategory.PROTOCOL_ERROR, "expected native ready message with version", delivery="not_sent", payload=ready)
        version = ready["version"]
        if self.expected_protocol_version is not None and version != self.expected_protocol_version:
            self._fail(AbortCategory.PROTOCOL_ERROR, f"unsupported native protocol {version!r}; expected {self.expected_protocol_version!r}", payload=ready)
        self.protocol_version = version

    def _publish(self, events: queue.Queue[tuple[str, Any]], overflow: threading.Event, event: tuple[str, Any]) -> None:
        try:
            events.put_nowait(event)
        except queue.Full:
            overflow.set()

    def _read_stdout(self, stream: Any, events: queue.Queue[tuple[str, Any]], overflow: threading.Event) -> None:
        try:
            while True:
                line = stream.readline(self.max_line_bytes + 1)
                if not line:
                    self._publish(events, overflow, ("eof", None))
                    return
                if len(line) > self.max_line_bytes:
                    self._publish(events, overflow, ("error", "stdout line exceeds byte limit"))
                    return
                try:
                    text = line.decode("utf-8").strip()
                except UnicodeDecodeError:
                    self._publish(events, overflow, ("error", "stdout is not UTF-8"))
                    return
                if not text:
                    continue
                if not text.startswith(("{", "[")):
                    self._diagnostics.append("stdout: " + text[:2048])
                    continue
                try:
                    value = json.loads(text, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"nonfinite JSON {value}")))
                except (ValueError, json.JSONDecodeError) as exc:
                    self._publish(events, overflow, ("error", f"malformed JSON response: {exc}"))
                    return
                if not isinstance(value, dict) or not isinstance(value.get("type"), str):
                    self._publish(events, overflow, ("error", "response must be an object with a string type"))
                    return
                self._publish(events, overflow, ("response", value))
        except (OSError, ValueError) as exc:
            self._publish(events, overflow, ("error", f"stdout reader failed: {exc}"))

    def _read_stderr(self, stream: Any) -> None:
        try:
            while line := stream.readline(2049):
                self._diagnostics.append("stderr: " + line.decode("utf-8", errors="replace").strip()[:2048])
        except (OSError, ValueError):
            pass  # Closing/killing a worker can close a reader's pipe.

    def _exchange(self, command: Mapping[str, Any], *, timeout_s: float | None = None) -> dict[str, Any]:
        process = self._process
        if process is None or process.stdin is None:
            raise NativeAbort(AbortCategory.NO_ACTIVE_RUN, "native process is not running")
        data = (json.dumps(command, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
        if len(data) > self.max_command_bytes:
            raise ValueError("command exceeds native byte limit")
        if self._overflow.is_set() or not self._events.empty():
            self._fail(AbortCategory.PROTOCOL_ERROR, "unsolicited native response; command alignment is unknown")
        if process.poll() is not None:
            self._fail(AbortCategory.PROCESS_EXIT, f"native process exited with {process.returncode}")
        deadline = time.monotonic() + (timeout_s if timeout_s is not None else self.timeout_s)
        done = threading.Event()
        write_error: list[BaseException] = []

        def write() -> None:
            try:
                view = memoryview(data)
                while view:
                    count = process.stdin.write(view)
                    if count is None or count <= 0:
                        raise BrokenPipeError("native stdin made no write progress")
                    view = view[count:]
                process.stdin.flush()
            except (OSError, ValueError) as exc:
                write_error.append(exc)
            finally:
                done.set()

        threading.Thread(target=write, daemon=True).start()
        if not done.wait(max(0.0, deadline - time.monotonic())):
            self._fail(AbortCategory.TRANSPORT_TIMEOUT, "native stdin write timed out", delivery="unknown")
        if write_error:
            self._fail(AbortCategory.PROCESS_EXIT, f"native stdin write failed: {write_error[0]}", delivery="unknown")
        return self._receive(deadline, delivery="unknown")

    def _receive(self, deadline: float, *, startup: bool = False, delivery: Literal["not_sent", "unknown", "response_received"] = "unknown") -> dict[str, Any]:
        while True:
            if self._overflow.is_set():
                self._fail(AbortCategory.PROTOCOL_ERROR, "native response queue overflow", delivery=delivery)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                category = AbortCategory.STARTUP_TIMEOUT if startup else AbortCategory.TRANSPORT_TIMEOUT
                self._fail(category, "native response deadline exceeded", delivery=delivery)
            try:
                kind, value = self._events.get(timeout=min(remaining, 0.05))
            except queue.Empty:
                continue
            if kind == "eof":
                self._fail(AbortCategory.PROCESS_EXIT, "EOF from native process", delivery=delivery)
            if kind == "error":
                self._fail(AbortCategory.PROTOCOL_ERROR, str(value), delivery=delivery)
            if value.get("type") == "error":
                self._fail(AbortCategory.ENGINE_ERROR, str(value.get("message", "native engine error")), delivery="response_received", payload=value)
            return cast(dict[str, Any], value)

    def _accept_decision(self, payload: dict[str, Any]) -> NativeDecision:
        if payload.get("type") != "decision":
            self._fail(AbortCategory.PROTOCOL_ERROR, "expected a native decision response", delivery="response_received", payload=payload)
        if payload.get("decision") not in _PHASES:
            self._fail(AbortCategory.UNSUPPORTED_STATE, f"unsupported native decision: {payload.get('decision')!r}", delivery="response_received", payload=payload)
        if not isinstance(payload.get("player"), dict) or not isinstance(payload.get("context"), dict):
            self._fail(AbortCategory.PROTOCOL_ERROR, "native decision is missing player or context object", delivery="response_received", payload=payload)
        if payload["decision"] == "game_over" and not isinstance(payload.get("victory"), bool):
            self._fail(AbortCategory.PROTOCOL_ERROR, "game_over is missing explicit victory Boolean", delivery="response_received", payload=payload)
        self._current = cast(NativeDecision, copy.deepcopy(payload))
        self._revision += 1
        return copy.deepcopy(self._current)

    def _fail(self, category: AbortCategory, message: str, *, delivery: Literal["not_sent", "unknown", "response_received"] = "not_sent", payload: dict[str, Any] | None = None) -> NoReturn:
        self._kill()
        raise NativeAbort(category, message, delivery=delivery, diagnostics=self.diagnostics, payload=payload)

    def _kill(self) -> None:
        process, self._process = self._process, None
        self._current = None
        self.last_transition = None
        if process is None:
            return
        if os.name != "nt":
            # A launcher such as `dotnet run` can have a game child that inherits
            # its pipes. Reap the isolated group, not only the launcher PID.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        elif process.poll() is None:
            try:
                subprocess.run(
                    ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    timeout=self.close_timeout_s, check=False,
                )
            except (OSError, subprocess.TimeoutExpired):
                pass
            if process.poll() is None:
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
        try:
            process.wait(timeout=self.close_timeout_s)
        except subprocess.TimeoutExpired:
            pass
        for stream in (process.stdin, process.stdout, process.stderr):
            if stream is not None:
                stream.close()

    def close(self) -> None:
        """Send native quit, verify acknowledgement and exit; kill on failure."""
        with self._lock:
            process = self._process
            if process is None:
                self._current = None
                return
            try:
                result = self._exchange({"cmd": "quit"}, timeout_s=self.close_timeout_s)
                if result.get("type") != "quit_result" or result.get("success") is not True:
                    self._fail(AbortCategory.SHUTDOWN_FAILURE, "native quit was not acknowledged", delivery="response_received", payload=result)
                try:
                    returncode = process.wait(timeout=self.close_timeout_s)
                except subprocess.TimeoutExpired:
                    self._fail(AbortCategory.SHUTDOWN_FAILURE, "native process did not exit after quit", delivery="response_received")
                if returncode != 0:
                    self._fail(AbortCategory.SHUTDOWN_FAILURE, f"native quit exit code was {returncode}", delivery="response_received")
            except NativeAbort as exc:
                if exc.category == AbortCategory.SHUTDOWN_FAILURE:
                    raise
                raise NativeAbort(
                    AbortCategory.SHUTDOWN_FAILURE,
                    f"native shutdown failed: {exc}", delivery=exc.delivery,
                    diagnostics=exc.diagnostics, payload=exc.payload,
                ) from exc
            finally:
                self._kill()

    def __enter__(self) -> NativeRunner:
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        if exc_type is not None:
            self._kill()
        else:
            self.close()
