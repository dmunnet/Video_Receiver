"""Display a video stream sent over UDP (for example, H.264 in MPEG-TS).

Install with: python -m pip install opencv-python
"""

import argparse
import sys
import time

import cv2

READ_TIMEOUT_MILLISECONDS = 1000
STREAM_IDLE_TIMEOUT_SECONDS = 3


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--host",
        default="10.203.130.142",
        help="Local interface to listen on (default: all interfaces)",
    )
    parser.add_argument("--port", type=int, default=5000, help="UDP port (default: 5000)")
    args = parser.parse_args()

    stream_url = (
        f"udp://{args.host}:{args.port}"
        "?fifo_size=5000000&overrun_nonfatal=1"
    )
    capture = cv2.VideoCapture(
        stream_url,
        cv2.CAP_FFMPEG,
        [cv2.CAP_PROP_READ_TIMEOUT_MSEC, READ_TIMEOUT_MILLISECONDS],
    )

    if not capture.isOpened():
        print(
            f"Could not open {stream_url}. Check that OpenCV has FFmpeg support "
            "and that the stream is available.",
            file=sys.stderr,
        )
        return 1

    print(f"Listening for video on UDP port {args.port}. Press q or Esc to quit.")
    last_frame_at = None
    try:
        while True:
            received, frame = capture.read()
            if not received:
                if last_frame_at is None:
                    print("Video stream ended or could not be decoded.", file=sys.stderr)
                    return 1
                if time.monotonic() - last_frame_at >= STREAM_IDLE_TIMEOUT_SECONDS:
                    print("Video stream stopped; closing display.")
                    return 0
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    return 0
                continue

            last_frame_at = time.monotonic()
            cv2.imshow("UDP video", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                return 0
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
