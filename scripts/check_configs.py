"""Check every config YAML for control characters / encoding damage."""
import pathlib
import sys

import yaml

sys.stdout.reconfigure(encoding="utf-8")

failures = 0
for path in sorted(pathlib.Path("configs").rglob("*.yaml")):
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8")
        decode_ok = True
    except UnicodeDecodeError as error:
        decode_ok = False
        print(f"{path} | NOT VALID UTF-8: {error}")
    # C1 controls U+0080..U+009F are illegal in YAML; other high chars (—, ", …) are fine.
    controls = [ch for ch in text if "\u007f" <= ch <= "\u009f"]
    try:
        yaml.safe_load(text)
        status = "YAML OK"
    except Exception as error:
        status = "YAML FAIL: " + str(error).splitlines()[0][:80]
    ok = decode_ok and not controls and status == "YAML OK"
    failures += 0 if ok else 1
    print(f"{path} | c1-controls: {len(controls)} | {status}")

print("ALL CONFIGS OK" if failures == 0 else f"{failures} FILE(S) BROKEN")
sys.exit(1 if failures else 0)

