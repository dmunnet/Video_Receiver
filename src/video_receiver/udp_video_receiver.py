"""Display a video stream sent over UDP (for example, H.264 in MPEG-TS).

Install with: python -m pip install opencv-python
"""

import argparse
import queue
import sys
import threading
import time

import cv2
import numpy as np

DEFAULT_PORTS = (5000, 5001, 5002, 5003, 5004, 5005)
READ_TIMEOUT_MILLISECONDS = 1000
STREAM_IDLE_TIMEOUT_SECONDS = 3


class VideoStream:
    def __init__(self, host: str, port: int) -> None:
        self.port = port
        self.window_name = f"UDP {port}"
        stream_url = (
            f"udp://{host}:{port}?fifo_size=5000000&overrun_nonfatal=1"
        )
        self.capture = cv2.VideoCapture(
            stream_url,
            cv2.CAP_FFMPEG,
            [cv2.CAP_PROP_READ_TIMEOUT_MSEC, READ_TIMEOUT_MILLISECONDS],
        )
        self.frames: queue.Queue[tuple[float, np.ndarray]] = queue.Queue(maxsize=1)
        self.stop_event = threading.Event()
        self.reader: threading.Thread | None = None
        self.last_frame_at: float | None = None
        self.window_open = False

    def start(self) -> None:
        self.reader = threading.Thread(target=self._read_frames, daemon=True)
        self.reader.start()

    def _read_frames(self) -> None:
        while not self.stop_event.is_set():
            received, frame = self.capture.read()
            if not received:
                time.sleep(0.01)
                continue

            item = (time.monotonic(), frame)
            try:
                self.frames.put_nowait(item)
            except queue.Full:
                try:
                    self.frames.get_nowait()
                except queue.Empty:
                    pass
                self.frames.put_nowait(item)

    def stop(self) -> None:
        self.stop_event.set()
        if self.reader is not None:
            self.reader.join(READ_TIMEOUT_MILLISECONDS / 1000 + 1)
        self.capture.release()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--host",
        default="10.203.130.142",
        help="Local interface to listen on (default: all interfaces)",
    )
    parser.add_argument(
        "--port",
        type=int,
        action="append",
        help="UDP port to listen on; may be repeated "
        "(default: 5000, 5001, 5002, 5003, 5004, 5005)",
    )
    args = parser.parse_args()

    ports = args.port or DEFAULT_PORTS
    streams = [VideoStream(args.host, port) for port in ports]
    open_streams = []
    for stream in streams:
        if stream.capture.isOpened():
            open_streams.append(stream)
        else:
            print(
                f"Could not open UDP port {stream.port}. Check that OpenCV has "
                "FFmpeg support and that the port is available.",
                file=sys.stderr,
            )
            stream.capture.release()

    if not open_streams:
        return 1

    for stream in open_streams:
        stream.start()
    print(
        f"Listening for video on UDP ports "
        f"{', '.join(str(stream.port) for stream in open_streams)}. "
        "Press q or Esc to quit."
    )

    try:
        while True:
            now = time.monotonic()
            for stream in open_streams:
                try:
                    frame_at, frame = stream.frames.get_nowait()
                except queue.Empty:
                    pass
                else:
                    stream.last_frame_at = frame_at
                    if not stream.window_open:
                        print(f"Video stream detected on UDP port {stream.port}.")
                        stream.window_open = True
                    cv2.imshow(stream.window_name, frame)

                if (
                    stream.window_open
                    and stream.last_frame_at is not None
                    and now - stream.last_frame_at >= STREAM_IDLE_TIMEOUT_SECONDS
                ):
                    print(f"Video stream stopped on UDP port {stream.port}.")
                    cv2.destroyWindow(stream.window_name)
                    stream.window_open = False
                    stream.last_frame_at = None

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                return 0
    finally:
        for stream in open_streams:
            stream.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
