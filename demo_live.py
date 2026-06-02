"""
demo_live.py — stream live EEG band-power to the browser via WebSocket.

Usage
-----
1. Make sure EMOTIV Launcher is open and headset is connected.
2. Run: python demo_live.py
3. Open index.html in your browser.
"""

from __future__ import annotations

import asyncio
import json
import sys
import threading

from bci import PowStream, stream_pow

PORT = 8765
_clients: set = set()


async def _ws_handler(websocket) -> None:
    _clients.add(websocket)
    print(f"  [ws] browser connected  ({len(_clients)} client(s))")
    try:
        await websocket.wait_closed()
    finally:
        _clients.discard(websocket)
        print(f"  [ws] browser disconnected ({len(_clients)} client(s))")


async def _broadcast(msg: str) -> None:
    if _clients:
        await asyncio.gather(
            *[c.send(msg) for c in set(_clients)],
            return_exceptions=True,
        )


async def _serve(stream: PowStream) -> None:
    try:
        import websockets
    except ImportError:
        print("[ERROR] pip install websockets")
        sys.exit(1)

    loop = asyncio.get_running_loop()

    def _stream_thread() -> None:
        for frame in stream_pow(stream, poll_interval=0.02):
            msg = json.dumps({
                "power":         frame.power.tolist(),
                "channel_names": frame.channel_names,
                "band_names":    frame.band_names,
                "sample_index":  frame.sample_index,
            })
            if _clients:
                asyncio.run_coroutine_threadsafe(_broadcast(msg), loop)

    t = threading.Thread(target=_stream_thread, daemon=True)
    t.start()

    async with websockets.serve(_ws_handler, "localhost", PORT):
        print(f"\n  WebSocket server → ws://localhost:{PORT}")
        print("  Open index.html in your browser\n")
        await asyncio.Future()  # run until interrupted


def main() -> None:
    print("=" * 55)
    print("EEG CSA — Live EEG WebSocket Server")
    print("=" * 55)
    print("\nConnecting to Emotiv Cortex...")

    try:
        stream = PowStream.from_env()
    except EnvironmentError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    with stream:
        print(f"  Headset : {stream.channel_names}")
        print(f"  Channels: {stream.n_channels}  Bands: {stream.band_names}")
        try:
            asyncio.run(_serve(stream))
        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    main()
