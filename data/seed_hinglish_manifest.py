#!/usr/bin/env python3
"""Create a 120-clip human-recording manifest. Audio is intentionally not synthesized."""
from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

ENDS = [
    "haan theek hai, main kal morning mein update bhej dunga", "mujhe Hyderabad ka weather bata do", "please is order ko cancel kar do",
    "meeting ko Friday afternoon par shift kar dete hain", "main abhi payment complete kar raha hoon", "yeh option mujhe kaafi useful laga",
    "aap customer ko ek polite follow-up bhej do", "Bangalore ka traffic aaj bahut heavy hai", "mujhe English mein summary chahiye",
    "haan, is problem ka solution deploy kar do", "mera package kal tak deliver hona chahiye", "chalo, main baad mein call karta hoon",
]
MID = [
    "mujhe Hyderabad ka weather batao, actually wait, Bangalore bhi", "um main report share karunga, matlab pehle ek cheez check kar loon",
    "please invoice bhej do, uh, ek minute, new address add karna hai", "aaj standup mein main bolunga ki, like, deployment pending hai",
    "customer ko bolo ki refund, actually partial refund process hoga", "main Delhi jaa raha hoon, matlab agar train confirm hui toh",
    "ye file upload kar do aur, um, uske baad validation run karna", "mujhe analytics dashboard kholo, actually sales wala tab",
    "agar API fail ho rahi hai toh, uh, logs bhi dekh lena", "main soch raha tha ki, like, feature flag enable kar dein",
    "kal ka schedule bhej do, matlab team calendar wala", "mujhe courier track karna hai, actually AWB number abhi bhejta hoon",
]
FILLERS = ["umm", "uh", "actually", "matlab", "like"]


def main() -> None:
    rows = []
    for index in range(120):
        is_end = index % 2 == 0
        transcript = ENDS[(index // 2) % len(ENDS)] if is_end else MID[(index // 2) % len(MID)]
        filler = FILLERS[index % len(FILLERS)]
        rows.append({"id": f"{index + 1:03d}", "audio_file": f"hinglish_{index + 1:03d}.wav", "endpoint_bool": str(is_end).lower(),
                     "midfiller": str((not is_end) or filler in transcript).lower(), "endfiller": str(is_end and index % 10 == 0).lower(),
                     "transcript": transcript, "recording_notes": f"Natural conversational delivery; include a {filler} hesitation where appropriate."})
    output = ROOT / "data" / "hinglish_manifest.csv"
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    print(f"Wrote {len(rows)} prompts to {output}. Record matching WAV files in data/hinglish_recordings/.")


if __name__ == "__main__": main()
