"""Record the demo reel to a video file.

Drives the reel in a real browser at its native 1600x900, then corrects the
capture, because Playwright's recorder gets the timeline wrong in two ways:

  * It stamps frames at 25fps while capturing at whatever rate it manages, so
    a 40 second capture plays back over 55 seconds.
  * Recording starts when the browser context is created, so the file opens
    with the page load and the play gate before the reel begins.

Both are measured here rather than guessed: the script times its own lead-in
and total run, reads the raw duration back from the file, and uses those to
rescale the timestamps and cut the head off accurately.

    python promo/record.py          # needs the promo server on :8010

Produces promo/chrisai-reel.webm. Silent - browsers do not expose recorded
audio, so the soundtrack is muxed in separately.
"""

import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
OUT = HERE / "chrisai-reel.webm"
URL = "http://localhost:8010/promo/demo.html"
RUN_SECONDS = 36

PW = Path.home() / "AppData/Local/ms-playwright"
CHROME = PW / "chromium-1223" / "chrome-win64" / "chrome.exe"
FFMPEG = PW / "ffmpeg-1011" / "ffmpeg-win64.exe"


def probe_duration(path: Path) -> float:
    out = subprocess.run([str(FFMPEG), "-i", str(path)],
                         capture_output=True, text=True).stderr
    m = re.search(r"Duration: (\d+):(\d+):(\d+\.\d+)", out)
    if not m:
        raise RuntimeError(f"could not read duration of {path}")
    h, mnt, s = m.groups()
    return int(h) * 3600 + int(mnt) * 60 + float(s)


def main() -> int:
    for exe in (CHROME, FFMPEG):
        if not exe.exists():
            print(f"missing: {exe}")
            return 1

    tmp = HERE / "_rec"
    shutil.rmtree(tmp, ignore_errors=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(
            executable_path=str(CHROME),
            headless=True,
            args=["--autoplay-policy=no-user-gesture-required",
                  "--force-device-scale-factor=1", "--hide-scrollbars"],
        )
        context = browser.new_context(
            viewport={"width": 1600, "height": 900},
            record_video_dir=str(tmp),
            record_video_size={"width": 1600, "height": 900},
        )
        t0 = time.time()  # recording starts here, not at the click

        page = context.new_page()
        page.goto(URL, wait_until="networkidle")
        # Webfonts must land before the first frame or the title records in a
        # fallback face.
        page.wait_for_function("document.fonts.status === 'loaded'", timeout=15000)
        page.evaluate("document.getElementById('hint').classList.add('gone')")
        time.sleep(1.0)

        lead = time.time() - t0
        page.click("#gate")
        print(f"lead-in before reel starts: {lead:.2f}s")

        time.sleep(RUN_SECONDS)
        total = time.time() - t0

        video = page.video
        context.close()
        browser.close()
        raw = Path(video.path())

    captured = probe_duration(raw)
    scale = total / captured
    print(f"wall clock {total:.2f}s vs file {captured:.2f}s -> timestamp scale {scale:.4f}")

    # One pass: rescale the timestamps, then cut the lead-in. Re-encoding rather
    # than copying, because a stream copy can only cut at a keyframe and VP8
    # keyframes here are far enough apart to leave the gate on screen.
    if OUT.exists():
        OUT.unlink()
    cmd = [str(FFMPEG), "-y", "-itsscale", f"{scale:.6f}", "-i", str(raw),
           "-ss", f"{lead:.3f}", "-c:v", "libvpx", "-b:v", "1800k",
           "-deadline", "good", "-cpu-used", "2", str(OUT)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr[-1200:])
        return 1

    shutil.rmtree(tmp, ignore_errors=True)
    print(f"wrote {OUT} ({OUT.stat().st_size/1e6:.1f} MB, "
          f"{probe_duration(OUT):.1f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
