import json
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOG = ROOT / "agent.log"
RUNS = 3

PROMPT = (
    "program.md dosyasini oku ve talimatlarini harfiyen uygula. "
    "Kullanici yok, soru sorma; branch etiketini kendin sec. "
    "results.tsv'de baslik disinda satir varsa kurulum zaten yapilmistir: results.tsv'yi TEMIZLEME, "
    "results.tsv ve git log'u okuyup kaldigin yerden donguye devam et. Dongu sonsuz."
)


def _brief(value):
    if not isinstance(value, dict):
        return ""
    for key in ("command", "file_path", "pattern", "query"):
        text = value.get(key)
        if isinstance(text, str) and text.strip():
            return " ".join(text.split())[:160]
    return ""


def visible(line):
    """stream-json satirini terminale basilacak metne cevir."""
    raw = line.strip()
    if not raw:
        return ""
    try:
        ev = json.loads(raw)
    except json.JSONDecodeError:
        return line if line.endswith("\n") else line + "\n"

    if ev.get("type") == "result" and ev.get("is_error"):
        return f"hata: {ev.get('result') or ev.get('error')}\n"

    out = []
    if ev.get("type") == "assistant":
        for block in ev.get("message", {}).get("content") or []:
            if block.get("type") == "text" and block.get("text"):
                out.append(block["text"].rstrip() + "\n")
            elif block.get("type") == "tool_use":
                extra = _brief(block.get("input"))
                name = block.get("name") or "tool"
                out.append(f"$ {name} {extra}\n" if extra else f"$ {name}\n")
    elif ev.get("type") == "user":
        for block in ev.get("message", {}).get("content") or []:
            if block.get("type") == "tool_result" and block.get("is_error"):
                text = block.get("content")
                if isinstance(text, str) and text.strip():
                    out.append("  hata: " + " ".join(text.split())[:240] + "\n")
    return "".join(out)


def run_claude(log):
    proc = subprocess.Popen(
        [
            shutil.which("claude") or "claude",
            "-p",
            PROMPT,
            "--dangerously-skip-permissions",
            "--verbose",
            "--output-format",
            "stream-json",
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    try:
        for line in proc.stdout:
            log.write(line)
            log.flush()
            text = visible(line)
            if text:
                print(text, end="", flush=True)
        return proc.wait()
    finally:
        if proc.poll() is None:
            proc.terminate()


def main():
    with open(LOG, "a", encoding="utf-8") as log:
        for _ in range(RUNS):
            header = f"=== {datetime.now():%Y-%m-%d %H:%M:%S} yeni oturum ===\n"  # noqa: DTZ005
            print(header, end="")
            log.write(header)

            if run_claude(log) == 0:
                time.sleep(5)
            else:
                print("claude hata ile cikti, 60 sn bekleniyor")
                log.write("claude hata ile cikti\n")
                time.sleep(60)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    try:
        main()
    except KeyboardInterrupt:
        print("\ndurduruldu")
