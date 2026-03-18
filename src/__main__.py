"""CLI for voicetwin."""
import sys, json, argparse
from .core import Voicetwin

def main():
    parser = argparse.ArgumentParser(description="VoiceTwin — AI Voice Cloning. Clone any voice from a 30-second sample for text-to-speech.")
    parser.add_argument("command", nargs="?", default="status", choices=["status", "run", "info"])
    parser.add_argument("--input", "-i", default="")
    args = parser.parse_args()
    instance = Voicetwin()
    if args.command == "status":
        print(json.dumps(instance.get_stats(), indent=2))
    elif args.command == "run":
        print(json.dumps(instance.process(input=args.input or "test"), indent=2, default=str))
    elif args.command == "info":
        print(f"voicetwin v0.1.0 — VoiceTwin — AI Voice Cloning. Clone any voice from a 30-second sample for text-to-speech.")

if __name__ == "__main__":
    main()
