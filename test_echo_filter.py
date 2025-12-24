#!/usr/bin/env python3
"""Test script for echo filter functionality."""

import time
from echo_filter import EchoFilter


def test_echo_filter():
    """Test echo filter with various scenarios."""
    print("Testing EchoFilter...")

    # Create filter
    echo_filter = EchoFilter(buffer_ms=500, similarity_threshold=0.80, max_stored_responses=3)

    # Test 1: Exact match
    print("\n--- Test 1: Exact Match ---")
    echo_filter.set_tts_response("The weather today is sunny", duration_ms=2000)

    assert echo_filter.is_echo("The weather today is sunny") == True, "Should detect exact match"
    print("✓ Exact match detected as echo")

    # Test 2: Partial match (substring)
    print("\n--- Test 2: Partial Match ---")
    assert echo_filter.is_echo("weather today is sunny") == True, "Should detect substring"
    print("✓ Substring detected as echo")

    # Test 3: Different text (not echo)
    print("\n--- Test 3: Different Text ---")
    assert echo_filter.is_echo("Tell me about tomorrow") == False, (
        "Should not detect different text"
    )
    print("✓ Different text not detected as echo")

    # Test 4: Fuzzy match with typo
    print("\n--- Test 4: Fuzzy Match ---")
    echo_filter.set_tts_response("Hello how are you today", duration_ms=1500)
    assert echo_filter.is_echo("Hello how are you todey") == True, "Should detect fuzzy match"
    print("✓ Fuzzy match (with typo) detected as echo")

    # Test 5: Outside time window
    print("\n--- Test 5: Time Window ---")
    echo_filter.set_tts_response("This is a test", duration_ms=100)  # 100ms duration
    time.sleep(0.7)  # Wait 700ms (beyond 100ms + 500ms buffer)
    assert echo_filter.is_echo("This is a test") == False, "Should not detect outside window"
    print("✓ Text outside window not detected as echo")

    # Test 6: Within time window
    print("\n--- Test 6: Within Window ---")
    echo_filter.set_tts_response("Within the window", duration_ms=1000)
    time.sleep(0.2)  # Wait 200ms (well within 1000ms + 500ms buffer)
    assert echo_filter.is_echo("Within the window") == True, "Should detect within window"
    print("✓ Text within window detected as echo")

    # Test 7: Multiple stored responses
    print("\n--- Test 7: Multiple Responses ---")
    echo_filter.set_tts_response("First response", duration_ms=1000)
    echo_filter.set_tts_response("Second response", duration_ms=1000)
    echo_filter.set_tts_response("Third response", duration_ms=1000)

    assert echo_filter.is_echo("First response") == True, "Should detect first"
    assert echo_filter.is_echo("Second response") == True, "Should detect second"
    assert echo_filter.is_echo("Third response") == True, "Should detect third"
    print("✓ Multiple stored responses all detected")

    # Test 8: Storage limit
    print("\n--- Test 8: Storage Limit ---")
    echo_filter.set_tts_response("Fourth response", duration_ms=1000)
    stored = echo_filter.get_stored_responses()
    assert len(stored) == 3, f"Should keep max 3 responses, got {len(stored)}"
    assert "First response" not in stored, "Should have removed oldest"
    print("✓ Storage limit maintained (oldest dropped)")

    # Test 9: Case insensitivity
    print("\n--- Test 9: Case Insensitivity ---")
    echo_filter.set_tts_response("Hello World", duration_ms=1000)
    assert echo_filter.is_echo("hello world") == True, "Should be case insensitive"
    assert echo_filter.is_echo("HELLO WORLD") == True, "Should be case insensitive"
    print("✓ Case insensitive matching works")

    # Test 10: Clear filter
    print("\n--- Test 10: Clear Filter ---")
    echo_filter.clear()
    stored = echo_filter.get_stored_responses()
    assert len(stored) == 0, "Should have no stored responses"
    print("✓ Clear filter works")

    print("\n✅ All tests passed!")


if __name__ == "__main__":
    test_echo_filter()
