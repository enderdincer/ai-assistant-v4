"""Test script for Ollama client wrapper with qwen3:0.6b model."""

from ai_assistant.shared.ollama import OllamaClient, OllamaError


def test_chat_completion():
    """Test chat completion with qwen3:0.6b."""
    print("Testing chat completion with qwen3:0.6b...")
    print("-" * 50)

    try:
        client = OllamaClient()

        # Test basic chat
        messages = [{"role": "user", "content": "Hello! Can you tell me a short joke?"}]

        print("Sending chat request...")
        response = client.chat(model="qwen3:0.6b", messages=messages, options={"temperature": 0.7})

        print("\nResponse:")
        print(response["message"]["content"])
        print("\nMetadata:")
        print(f"Model: {response.get('model')}")
        print(f"Done: {response.get('done')}")
        if "total_duration" in response:
            print(f"Total duration: {response['total_duration'] / 1e9:.2f}s")

    except OllamaError as e:
        print(f"Error: {e}")
    finally:
        client.close()


def test_streaming_chat():
    """Test streaming chat with qwen3:0.6b."""
    print("\n" + "=" * 50)
    print("Testing streaming chat with qwen3:0.6b...")
    print("-" * 50)

    try:
        client = OllamaClient()

        messages = [
            {
                "role": "user",
                "content": "Count from 1 to 5 and explain why each number is interesting.",
            }
        ]

        print("Streaming response:")
        print("-" * 50)

        for chunk in client.chat(
            model="qwen3:0.6b", messages=messages, stream=True, options={"temperature": 0.7}
        ):
            if "message" in chunk:
                content = chunk["message"].get("content", "")
                if content:
                    print(content, end="", flush=True)

            # Print final metadata
            if chunk.get("done"):
                print("\n" + "-" * 50)
                print("Stream completed!")
                if "total_duration" in chunk:
                    print(f"Total duration: {chunk['total_duration'] / 1e9:.2f}s")

    except OllamaError as e:
        print(f"\nError: {e}")
    finally:
        client.close()


def test_generate():
    """Test simple generation with qwen3:0.6b."""
    print("\n" + "=" * 50)
    print("Testing text generation with qwen3:0.6b...")
    print("-" * 50)

    try:
        client = OllamaClient()

        prompt = "Write a haiku about coding."

        print(f"Prompt: {prompt}")
        print("Generating...")

        response = client.generate(model="qwen3:0.6b", prompt=prompt, options={"temperature": 0.8})

        print("\nResponse:")
        print(response["response"])
        print("\nMetadata:")
        print(f"Done: {response.get('done')}")
        if "total_duration" in response:
            print(f"Total duration: {response['total_duration'] / 1e9:.2f}s")

    except OllamaError as e:
        print(f"Error: {e}")
    finally:
        client.close()


def test_list_models():
    """Test listing available models."""
    print("\n" + "=" * 50)
    print("Testing model listing...")
    print("-" * 50)

    try:
        client = OllamaClient()

        models = client.list_models()

        print("Available models:")
        for model in models.get("models", []):
            name = model.get("name", "unknown")
            size = model.get("size", 0) / (1024**3)  # Convert to GB
            print(f"  - {name} ({size:.2f} GB)")

        # Check if qwen3:0.6b is available
        model_names = [m.get("name") for m in models.get("models", [])]
        if "qwen3:0.6b" in model_names:
            print("\n✓ qwen3:0.6b is available!")
        else:
            print("\n✗ qwen3:0.6b is NOT available.")
            print("  Run: ollama pull qwen3:0.6b")

    except OllamaError as e:
        print(f"Error: {e}")
    finally:
        client.close()


def test_context_manager():
    """Test using client with context manager."""
    print("\n" + "=" * 50)
    print("Testing context manager usage...")
    print("-" * 50)

    try:
        with OllamaClient() as client:
            response = client.chat(
                model="qwen3:0.6b",
                messages=[
                    {"role": "user", "content": "Say 'Hello World!' in 3 different languages."}
                ],
            )

            print("Response using context manager:")
            print(response["message"]["content"])
            print("\n✓ Context manager works correctly!")

    except OllamaError as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    print("=" * 50)
    print("Ollama Client Wrapper Test Suite")
    print("Model: qwen3:0.6b")
    print("=" * 50)

    # Run all tests
    test_list_models()
    test_chat_completion()
    test_streaming_chat()
    test_generate()
    test_context_manager()

    print("\n" + "=" * 50)
    print("All tests completed!")
    print("=" * 50)
