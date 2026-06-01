"""examples/jepa_chat.py — JEPA-powered chat room example.

Demonstrates local JEPA inference with API fallback.

Usage:
    python examples/jepa_chat.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jepa.jepa_room import JEPARoom

def main():
    print("🧠 JEPA Chat Room Demo")
    print("=" * 40)
    
    room = JEPARoom(room_id="chat", dim=128)
    
    # Feed conversation history
    print("\n📚 Loading conversation history...")
    conversations = [
        ("hello", "Hi! How can I help?"),
        ("fleet status", "All systems green."),
        ("agent count", "50 agents active."),
        ("temperature", "72°C nominal."),
        ("bye", "Goodbye!"),
    ]
    
    for q, a in conversations:
        room.feed_tile({"question": q, "answer": a, "domain": "chat"})
    
    print(f"Loaded {len(conversations)} conversation pairs.")
    
    # Chat loop
    print("\n💬 Chat loop (type 'quit' to exit):")
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            break
        
        result = room.query(user_input, min_confidence=0.6)
        
        if result.source == "jepa":
            print(f"🧠 JEPA [{result.confidence:.2f}]: ", end="")
        else:
            print(f"☁️  API: ", end="")
        
        if result.predicted_tile:
            print(result.predicted_tile.get("answer", "I don't know"))
        else:
            print("I don't understand.")
    
    print("\n📊 Final stats:")
    print(f"   {room.get_stats()}")
    print("\n✅ Demo complete!")

if __name__ == "__main__":
    main()
