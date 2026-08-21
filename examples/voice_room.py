"""examples/voice_room.py — Voice-enabled Plato room example.

Demonstrates soniqo integration: voice input → tile → voice output.

Usage:
    python examples/voice_room.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voice.soniqo_bridge import SoniqoBridge
from jepa.jepa_room import JEPARoom

def main():
    print("🎤 Plato Voice Room Demo")
    print("=" * 40)
    
    # Initialize voice bridge
    bridge = SoniqoBridge(room_id="harbor", node_id="demo")
    bridge.connect()
    
    # Initialize JEPA room
    room = JEPARoom(room_id="harbor", dim=128)
    
    # Feed some knowledge tiles
    room.feed_tile({"question": "What is the fleet status?", "answer": "All systems nominal.", "domain": "harbor"})
    room.feed_tile({"question": "How many agents?", "answer": "50 active agents.", "domain": "harbor"})
    room.feed_tile({"question": "System temperature?", "answer": "72°C, within normal range.", "domain": "harbor"})
    
    print("\n🗣️  Simulating voice interactions...")
    
    # Simulate voice query
    voice_input = "fleet status"
    print(f"\nUser: '{voice_input}'")
    
    # Submit as voice tile
    tile = bridge.submit_voice_tile(voice_input, "human_operator")
    print(f"📝 Transcribed: {tile.transcript}")
    
    # Query JEPA room
    result = room.query(voice_input, min_confidence=0.5)
    
    if result.source == "jepa":
        print(f"🧠 JEPA prediction (confidence: {result.confidence:.2f})")
        if result.predicted_tile:
            print(f"💬 Answer: {result.predicted_tile.get('answer', 'Unknown')}")
    else:
        print(f"☁️  API fallback: {result.predicted_tile.get('answer', 'Unknown')}")
    
    # Voice output
    response_text = result.predicted_tile.get('answer', 'I am not sure') if result.predicted_tile else 'I am not sure'
    bridge.speak(response_text)
    print(f"🔊 Spoke: '{response_text}'")
    
    # Show stats
    print(f"\n📊 Stats:")
    print(f"   Voice tiles: {len(bridge.get_voice_history())}")
    print(f"   Room stats: {room.get_stats()}")
    
    bridge.disconnect()
    print("\n✅ Demo complete!")

if __name__ == "__main__":
    main()
