"""
stream.py — real-time EEG streaming via the Emotiv Cortex WebSocket API.

The Emotiv Cortex API runs as a local WebSocket server (wss://localhost:6450)
fronted by the EMOTIV App.  This module handles the full auth handshake,
headset connection, and EEG data subscription, then feeds samples into a
circular buffer that compute_csa() can consume epoch-by-epoch.

Public API
----------
EmotivStream(client_id, client_secret, **kwargs)
    Context-manager that connects, authenticates, and streams EEG.

stream_csa(stream, fs, epoch_sec, overlap, fmax) -> Iterator[CSAResult]
    Yields a fresh CSAResult each time a new epoch is ready.

Environment variables (loaded automatically if python-dotenv is installed)
--------------------------------------------------------------------------
EMOTIV_CLIENT_ID
EMOTIV_CLIENT_SECRET
"""

from __future__ import annotations

import asyncio
import json
import os
import ssl
import threading
import time
import warnings
from collections import deque
from dataclasses import dataclass, field
from typing import Iterator

import numpy as np

from .csa import compute_csa, CSAResult

# ---------------------------------------------------------------------------
# Optional dotenv support
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

_CORTEX_URL = "wss://localhost:6868"
_APP_NAME = "Eric's CSA"


# ---------------------------------------------------------------------------
# Public data classes
# ---------------------------------------------------------------------------

@dataclass
class HeadsetInfo:
    """Metadata returned by the Cortex API for a connected headset."""

    headset_id: str
    status: str
    n_channels: int
    fs: float
    channel_names: list[str]


@dataclass
class EmotivStream:
    """Real-time EEG stream from an Emotiv headset via the Cortex API.

    Usage
    -----
    ::

        with EmotivStream.from_env() as stream:
            for result in stream_csa(stream, epoch_sec=2.0):
                print(result.power_matrix.shape)

    Parameters
    ----------
    client_id : str
        Emotiv Cortex client ID.
    client_secret : str
        Emotiv Cortex client secret.
    buffer_sec : float
        Length of the internal circular sample buffer in seconds.
        Must be at least 2× the longest epoch you intend to compute.
    headset_id : str or None
        Specific headset ID to connect to.  ``None`` connects to the
        first available headset.
    """

    client_id: str
    client_secret: str
    buffer_sec: float = 30.0
    headset_id: str | None = None

    # Runtime state — populated during connect()
    _token: str = field(default="", init=False, repr=False)
    _session_id: str = field(default="", init=False, repr=False)
    _headset: HeadsetInfo | None = field(default=None, init=False, repr=False)
    _buffer: deque = field(default_factory=deque, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _running: bool = field(default=False, init=False, repr=False)
    _thread: threading.Thread | None = field(default=None, init=False, repr=False)
    _loop: asyncio.AbstractEventLoop | None = field(default=None, init=False, repr=False)
    _total_samples: int = field(default=0, init=False, repr=False)

    # ---------------------------------------------------------------------------
    # Constructors
    # ---------------------------------------------------------------------------

    @classmethod
    def from_env(cls, **kwargs) -> "EmotivStream":
        """Construct from ``EMOTIV_CLIENT_ID`` / ``EMOTIV_CLIENT_SECRET`` env vars."""
        client_id = os.environ.get("EMOTIV_CLIENT_ID", "")
        client_secret = os.environ.get("EMOTIV_CLIENT_SECRET", "")
        if not client_id or not client_secret:
            raise EnvironmentError(
                "EMOTIV_CLIENT_ID and EMOTIV_CLIENT_SECRET must be set. "
                "Add them to your .env file."
            )
        return cls(client_id=client_id, client_secret=client_secret, **kwargs)

    # ---------------------------------------------------------------------------
    # Context manager
    # ---------------------------------------------------------------------------

    def __enter__(self) -> "EmotivStream":
        self.connect()
        return self

    def __exit__(self, *_) -> None:
        self.disconnect()

    # ---------------------------------------------------------------------------
    # Connect / disconnect
    # ---------------------------------------------------------------------------

    def connect(self) -> None:
        """Authenticate with Cortex, connect headset, and start streaming."""
        self._running = True
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True, name="emotiv-cortex"
        )
        self._thread.start()
        # Wait until the headset info is populated (auth + subscription done)
        deadline = time.monotonic() + 30.0
        while self._headset is None and time.monotonic() < deadline:
            if not self._running:
                raise RuntimeError(
                    "Emotiv stream failed to start. "
                    "Check that the EMOTIV App is running and credentials are correct."
                )
            time.sleep(0.1)
        if self._headset is None:
            raise TimeoutError(
                "Timed out waiting for Emotiv headset. "
                "Is the EMOTIV App open and a headset paired?"
            )

    def disconnect(self) -> None:
        """Close the Cortex session and stop the background thread."""
        self._running = False
        if self._loop and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(timeout=5.0)

    # ---------------------------------------------------------------------------
    # Buffer access
    # ---------------------------------------------------------------------------

    @property
    def fs(self) -> float:
        """Sampling frequency reported by the headset."""
        if self._headset is None:
            raise RuntimeError("Stream not connected yet.")
        return self._headset.fs

    @property
    def n_channels(self) -> int:
        """Number of EEG channels."""
        if self._headset is None:
            raise RuntimeError("Stream not connected yet.")
        return self._headset.n_channels

    @property
    def channel_names(self) -> list[str]:
        """Channel labels (e.g. AF3, F7, …)."""
        if self._headset is None:
            raise RuntimeError("Stream not connected yet.")
        return self._headset.channel_names

    def read_samples(self, n: int) -> np.ndarray | None:
        """Return the last *n* samples as ``(n_channels, n)`` or ``None`` if
        fewer than *n* samples are buffered."""
        with self._lock:
            if len(self._buffer) < n:
                return None
            samples = list(self._buffer)[-n:]
        return np.array(samples).T  # (n_channels, n)

    def n_buffered(self) -> int:
        """Number of samples currently in the buffer."""
        with self._lock:
            return len(self._buffer)

    @property
    def total_samples(self) -> int:
        """Monotonically increasing count of all samples ever received."""
        with self._lock:
            return self._total_samples

    # ---------------------------------------------------------------------------
    # Async Cortex handshake (runs in background thread)
    # ---------------------------------------------------------------------------

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._cortex_session())
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"Emotiv Cortex session ended: {exc}", RuntimeWarning, stacklevel=1)
            self._running = False

    async def _cortex_session(self) -> None:
        try:
            import websockets  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "websockets is required for real-time streaming. "
                "Install it with: pip install websockets"
            ) from exc

        ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ssl_ctx.check_hostname = False
        ssl_ctx.verify_mode = ssl.CERT_NONE  # Cortex uses a self-signed cert

        async with websockets.connect(_CORTEX_URL, ssl=ssl_ctx) as ws:
            # 1. Request access
            await self._request_access(ws)
            # 2. Authorize → get token
            self._token = await self._authorize(ws)
            # 3. Connect headset
            headset = await self._connect_headset(ws)
            # 4. Create session
            self._session_id = await self._create_session(ws, headset.headset_id)
            # 5. Subscribe to EEG stream
            channel_names, fs = await self._subscribe_eeg(ws)

            n_channels = len(channel_names)
            max_buf = int(self.buffer_sec * fs)
            self._buffer = deque(maxlen=max_buf)
            self._headset = HeadsetInfo(
                headset_id=headset.headset_id,
                status="connected",
                n_channels=n_channels,
                fs=fs,
                channel_names=channel_names,
            )

            # 6. Read samples until disconnect
            await self._read_eeg(ws)

    # ---------------------------------------------------------------------------
    # Cortex JSON-RPC helpers
    # ---------------------------------------------------------------------------

    async def _send(self, ws, method: str, params: dict, req_id: int = 1) -> dict:
        payload = {"jsonrpc": "2.0", "method": method, "params": params, "id": req_id}
        await ws.send(json.dumps(payload))
        while True:
            raw = await ws.recv()
            msg = json.loads(raw)
            if msg.get("id") == req_id:
                if "error" in msg:
                    raise RuntimeError(
                        f"Cortex error [{method}]: {msg['error']}"
                    )
                return msg.get("result", {})

    async def _request_access(self, ws) -> None:
        await self._send(ws, "requestAccess", {
            "clientId": self.client_id,
            "clientSecret": self.client_secret,
        }, req_id=1)

    async def _authorize(self, ws) -> str:
        result = await self._send(ws, "authorize", {
            "clientId": self.client_id,
            "clientSecret": self.client_secret,
            "debit": 1,
        }, req_id=2)
        return result["cortexToken"]

    async def _connect_headset(self, ws) -> HeadsetInfo:
        # Query available headsets
        result = await self._send(ws, "queryHeadsets", {}, req_id=3)
        headsets = result if isinstance(result, list) else result.get("headsets", [])
        if not headsets:
            raise RuntimeError(
                "No Emotiv headsets found. "
                "Make sure your headset is paired in the EMOTIV App."
            )
        if self.headset_id:
            matched = [h for h in headsets if h["id"] == self.headset_id]
            if not matched:
                raise RuntimeError(
                    f"Headset '{self.headset_id}' not found. "
                    f"Available: {[h['id'] for h in headsets]}"
                )
            h = matched[0]
        else:
            h = headsets[0]

        # Connect if not already connected
        if h.get("status") != "connected":
            await self._send(ws, "controlDevice", {
                "command": "connect",
                "headsetId": h["id"],
            }, req_id=4)
            await asyncio.sleep(2.0)  # allow pairing

        return HeadsetInfo(
            headset_id=h["id"],
            status=h.get("status", "unknown"),
            n_channels=0,
            fs=0.0,
            channel_names=[],
        )

    async def _create_session(self, ws, headset_id: str) -> str:
        result = await self._send(ws, "createSession", {
            "cortexToken": self._token,
            "headset": headset_id,
            "status": "active",
        }, req_id=5)
        return result["id"]

    async def _subscribe_eeg(self, ws) -> tuple[list[str], float]:
        result = await self._send(ws, "subscribe", {
            "cortexToken": self._token,
            "session": self._session_id,
            "streams": ["eeg"],
        }, req_id=6)

        # Parse channel names and sampling rate from subscription response
        eeg_info = {}
        for item in result.get("success", []):
            if item.get("streamName") == "eeg":
                eeg_info = item
                break

        cols = eeg_info.get("cols", [])
        # Cortex returns cols like ["COUNTER","INTERPOLATED","AF3","F7",...]
        # Strip metadata columns — keep only electrode labels
        skip = {"COUNTER", "INTERPOLATED", "RAW_CQ", "MARKER_HARDWARE", "MARKERS"}
        channel_names = [c for c in cols if c.upper() not in skip]
        fs = float(eeg_info.get("samplingRate", 128.0))
        return channel_names, fs

    async def _read_eeg(self, ws) -> None:
        while self._running:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            msg = json.loads(raw)
            if "eeg" not in msg:
                continue
            eeg_row = msg["eeg"]
            # eeg_row = [counter, interpolated, ch0, ch1, ..., marker]
            # Slice to just the channel samples (skip first 2, last 1)
            samples = eeg_row[2: 2 + self._headset.n_channels]
            with self._lock:
                self._buffer.append(samples)
                self._total_samples += 1


# ---------------------------------------------------------------------------
# Streaming CSA
# ---------------------------------------------------------------------------

def stream_csa(
    stream: EmotivStream,
    epoch_sec: float = 2.0,
    overlap: float = 0.5,
    fmax: float = 30.0,
    poll_interval: float = 0.05,
    max_polls: int | None = None,
) -> Iterator[CSAResult]:
    """Yield a :class:`~bci.csa.CSAResult` each time a new epoch is ready.

    The function blocks until enough samples are buffered, then yields
    incrementally as new epochs complete.  Run inside a thread or async
    task if you need the main thread free.

    Parameters
    ----------
    stream : EmotivStream
        A connected :class:`EmotivStream` instance.
    epoch_sec : float
        Epoch length in seconds.
    overlap : float
        Fractional overlap between successive epochs.
    fmax : float
        Maximum frequency to include in the CSA output.
    poll_interval : float
        Seconds to sleep between buffer checks.
    max_polls : int or None
        If set, stop after this many loop iterations (useful for testing).

    Yields
    ------
    CSAResult
        One result per epoch, with ``power_matrix`` shape
        ``(n_channels, 1, n_freqs)`` — a single-epoch slice suitable for
        online feature extraction.
    """
    fs = stream.fs
    epoch_len = int(epoch_sec * fs)
    step = max(1, int(epoch_len * (1.0 - overlap)))

    last_yield_at = 0  # stream.total_samples at time of last yield
    polls = 0

    while True:
        if max_polls is not None and polls >= max_polls:
            return
        polls += 1

        total = stream.total_samples
        n_buf = stream.n_buffered()

        if n_buf < epoch_len or (total - last_yield_at) < step:
            time.sleep(poll_interval)
            continue

        signal = stream.read_samples(epoch_len)
        if signal is None:
            time.sleep(poll_interval)
            continue

        result = compute_csa(
            signal,
            fs=fs,
            epoch_sec=epoch_sec,
            overlap=0.0,
            fmax=fmax,
        )
        yield result
        last_yield_at = total
        time.sleep(poll_interval)


# ---------------------------------------------------------------------------
# Band-power streaming (pow stream — no raw-EEG license required)
# ---------------------------------------------------------------------------

@dataclass
class PowStream:
    """Real-time EEG band-power stream via the Emotiv Cortex ``pow`` subscription.

    Streams pre-computed theta/alpha/betaL/betaH/gamma power per channel.
    Does not require an Advanced/Research Emotiv license.

    Usage
    -----
    ::

        with PowStream.from_env() as stream:
            for frame in stream_pow(stream):
                print(frame.power.shape)   # (n_channels, n_bands)
    """

    client_id: str
    client_secret: str
    headset_id: str | None = None
    buffer_size: int = 300  # number of frames to keep

    _token: str = field(default="", init=False, repr=False)
    _session_id: str = field(default="", init=False, repr=False)
    _channel_names: list = field(default_factory=list, init=False, repr=False)
    _band_names: list = field(default_factory=list, init=False, repr=False)
    _buffer: deque = field(default_factory=deque, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _running: bool = field(default=False, init=False, repr=False)
    _thread: threading.Thread | None = field(default=None, init=False, repr=False)
    _loop: asyncio.AbstractEventLoop | None = field(default=None, init=False, repr=False)
    _total_samples: int = field(default=0, init=False, repr=False)

    @classmethod
    def from_env(cls, **kwargs) -> "PowStream":
        client_id = os.environ.get("EMOTIV_CLIENT_ID", "")
        client_secret = os.environ.get("EMOTIV_CLIENT_SECRET", "")
        if not client_id or not client_secret:
            raise EnvironmentError(
                "EMOTIV_CLIENT_ID and EMOTIV_CLIENT_SECRET must be set."
            )
        return cls(client_id=client_id, client_secret=client_secret, **kwargs)

    def __enter__(self) -> "PowStream":
        self.connect()
        return self

    def __exit__(self, *_) -> None:
        self.disconnect()

    def connect(self) -> None:
        self._running = True
        self._buffer = deque(maxlen=self.buffer_size)
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True, name="emotiv-pow"
        )
        self._thread.start()
        deadline = time.monotonic() + 30.0
        while not self._channel_names and time.monotonic() < deadline:
            if not self._running:
                raise RuntimeError("PowStream failed to start.")
            time.sleep(0.1)
        if not self._channel_names:
            raise TimeoutError("Timed out waiting for Emotiv pow stream.")

    def disconnect(self) -> None:
        self._running = False
        if self._loop and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(timeout=5.0)

    @property
    def channel_names(self) -> list[str]:
        return list(self._channel_names)

    @property
    def band_names(self) -> list[str]:
        return list(self._band_names)

    @property
    def n_channels(self) -> int:
        return len(self._channel_names)

    @property
    def n_bands(self) -> int:
        return len(self._band_names)

    @property
    def total_samples(self) -> int:
        with self._lock:
            return self._total_samples

    def n_buffered(self) -> int:
        with self._lock:
            return len(self._buffer)

    def read_latest(self) -> np.ndarray | None:
        """Return the latest ``(n_channels, n_bands)`` power frame, or ``None``."""
        with self._lock:
            if not self._buffer:
                return None
            return self._buffer[-1].copy()

    def read_history(self, n: int) -> np.ndarray | None:
        """Return the last *n* frames as ``(n, n_channels, n_bands)``, or ``None``."""
        with self._lock:
            if len(self._buffer) < n:
                return None
            frames = list(self._buffer)[-n:]
        return np.stack(frames)

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._cortex_session())
        except Exception as exc:
            warnings.warn(f"PowStream session ended: {exc}", RuntimeWarning, stacklevel=1)
            self._running = False

    async def _cortex_session(self) -> None:
        try:
            import websockets
        except ImportError as exc:
            raise ImportError("Install websockets: pip install websockets") from exc

        ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ssl_ctx.check_hostname = False
        ssl_ctx.verify_mode = ssl.CERT_NONE

        async with websockets.connect(_CORTEX_URL, ssl=ssl_ctx) as ws:
            await self._send(ws, "requestAccess", {
                "clientId": self.client_id,
                "clientSecret": self.client_secret,
            }, req_id=1)
            result = await self._send(ws, "authorize", {
                "clientId": self.client_id,
                "clientSecret": self.client_secret,
                "debit": 1,
            }, req_id=2)
            self._token = result["cortexToken"]

            headsets = await self._send(ws, "queryHeadsets", {}, req_id=3)
            if isinstance(headsets, dict):
                headsets = headsets.get("headsets", [])
            if not headsets:
                raise RuntimeError("No headsets found.")
            h = next((x for x in headsets if x["id"] == self.headset_id), headsets[0]) \
                if self.headset_id else headsets[0]

            sess = await self._send(ws, "createSession", {
                "cortexToken": self._token,
                "headset": h["id"],
                "status": "active",
            }, req_id=4)
            self._session_id = sess["id"]

            sub = await self._send(ws, "subscribe", {
                "cortexToken": self._token,
                "session": self._session_id,
                "streams": ["pow"],
            }, req_id=5)

            success = sub.get("success", [])
            if not success:
                raise RuntimeError(f"pow subscription failed: {sub.get('failure')}")

            cols = success[0].get("cols", [])
            ch_names, band_names = [], []
            for col in cols:
                ch, band = col.split("/")
                if ch not in ch_names:
                    ch_names.append(ch)
                if band not in band_names:
                    band_names.append(band)
            self._channel_names = ch_names
            self._band_names = band_names

            await self._read_pow(ws)

    async def _send(self, ws, method: str, params: dict, req_id: int = 1) -> dict:
        payload = {"jsonrpc": "2.0", "method": method, "params": params, "id": req_id}
        await ws.send(json.dumps(payload))
        while True:
            raw = await ws.recv()
            msg = json.loads(raw)
            if msg.get("id") == req_id:
                if "error" in msg:
                    raise RuntimeError(f"Cortex error [{method}]: {msg['error']}")
                return msg.get("result", {})

    async def _read_pow(self, ws) -> None:
        n_ch = len(self._channel_names)
        n_bands = len(self._band_names)
        while self._running:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            msg = json.loads(raw)
            if "pow" not in msg:
                continue
            frame = np.array(msg["pow"]).reshape(n_ch, n_bands)
            with self._lock:
                self._buffer.append(frame)
                self._total_samples += 1


@dataclass
class PowFrame:
    """One frame of band-power data from a :class:`PowStream`."""
    power: np.ndarray          # (n_channels, n_bands)
    channel_names: list[str]
    band_names: list[str]
    sample_index: int


def stream_pow(
    stream: PowStream,
    poll_interval: float = 0.05,
) -> Iterator[PowFrame]:
    """Yield a :class:`PowFrame` for each new band-power sample.

    Parameters
    ----------
    stream : PowStream
        A connected :class:`PowStream` instance.
    poll_interval : float
        Seconds to sleep between buffer checks.

    Yields
    ------
    PowFrame
        Latest ``(n_channels, n_bands)`` power frame.
    """
    last_yield_at = 0

    while True:
        total = stream.total_samples
        if total <= last_yield_at:
            time.sleep(poll_interval)
            continue

        frame = stream.read_latest()
        if frame is None:
            time.sleep(poll_interval)
            continue

        yield PowFrame(
            power=frame,
            channel_names=stream.channel_names,
            band_names=stream.band_names,
            sample_index=total,
        )
        last_yield_at = total
        time.sleep(poll_interval)
