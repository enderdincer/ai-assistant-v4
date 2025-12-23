"""Vision processor using Qwen3-VL for image understanding."""

import time
from typing import Any, Optional

from ai_assistant.perception.processors.base import BaseProcessor
from ai_assistant.shared.interfaces import IEventBus, IEvent
from ai_assistant.shared.events import VisionDescriptionEvent
from ai_assistant.shared.logging import get_logger

logger = get_logger(__name__)


class VisionProcessor(BaseProcessor):
    """Vision processor using Qwen3-VL for image understanding.

    This processor subscribes to CameraFrameEvents, uses a Vision-Language Model
    (VLM) to describe the scene, and publishes VisionDescriptionEvents when
    significant changes are detected.

    Features:
    - Frame skipping to reduce processing load (default: 1 FPS at 30 FPS camera)
    - Change detection using word-level Jaccard distance
    - Only emits events when scene changes exceed threshold
    - Supports Qwen3-VL-2B-Instruct with optional quantization

    Configuration options:
        model_name: HuggingFace model identifier (default: "Qwen/Qwen3-VL-2B-Instruct")
        device: Device to run model on ("auto", "mps", "cuda", "cpu")
        quantization: Quantization mode ("none", "int8", "int4") - note: int8/int4 require CUDA
        frame_skip: Process every Nth frame (default: 30 for 1 FPS at 30 FPS camera)
        change_threshold: Minimum change score to emit event (default: 0.08)
        prompt: Prompt for the VLM (default: descriptive prompt)
        max_tokens: Maximum tokens in response (default: 128)
    """

    def __init__(
        self,
        processor_id: str,
        event_bus: IEventBus,
        config: Optional[dict[str, Any]] = None,
    ) -> None:
        """Initialize the vision processor.

        Args:
            processor_id: Unique identifier for this processor
            event_bus: Event bus for subscribing and publishing events
            config: Optional configuration dictionary
        """
        config = config or {}

        super().__init__(
            processor_id=processor_id,
            processor_type="vision",
            event_bus=event_bus,
            input_event_types=["camera.frame"],
            output_event_types=["vision.description"],
            config=config,
        )

        # Model configuration
        self._model_name = config.get("model_name", "Qwen/Qwen3-VL-2B-Instruct")
        self._device = config.get("device", "auto")
        self._quantization = config.get("quantization", "none")
        self._max_tokens = config.get("max_tokens", 150)  # Optimized default for speed

        # Processing configuration
        self._frame_skip = config.get("frame_skip", 30)
        self._change_threshold = config.get("change_threshold", 0.12)
        self._prompt = config.get(
            "prompt",
            "Describe what you see in this image, including people, objects, and activities.",
        )

        # State
        self._frame_count = 0
        self._previous_description: Optional[str] = None

        # Model components (loaded during initialization)
        self._model = None
        self._processor = None

    def _validate_config(self) -> None:
        """Validate processor configuration.

        Raises:
            ValueError: If configuration is invalid
        """
        if self._frame_skip < 1:
            raise ValueError("frame_skip must be at least 1")

        if not (0.0 <= self._change_threshold <= 1.0):
            raise ValueError("change_threshold must be between 0.0 and 1.0")

        if self._max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")

        valid_quantizations = ("none", "int8", "int4")
        if self._quantization not in valid_quantizations:
            raise ValueError(f"quantization must be one of {valid_quantizations}")

    def _initialize_processor(self) -> None:
        """Initialize the VLM model.

        Raises:
            RuntimeError: If model loading fails
        """
        self._logger.info(f"Loading vision model: {self._model_name}")
        start_time = time.time()

        try:
            import torch
            from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

            # Determine device
            if self._device == "auto":
                if torch.cuda.is_available():
                    device = "cuda"
                elif torch.backends.mps.is_available():
                    device = "mps"
                else:
                    device = "cpu"
            else:
                device = self._device

            self._logger.info(f"Using device: {device}")

            # Load model with appropriate configuration
            model_kwargs: dict[str, Any] = {
                "torch_dtype": torch.float16 if device != "cpu" else torch.float32,
            }

            # For MPS, don't use device_map="auto" as it causes issues
            if device == "cuda":
                model_kwargs["device_map"] = "auto"

            # Apply quantization if requested (requires CUDA)
            if self._quantization in ("int8", "int4") and device == "cuda":
                from transformers import BitsAndBytesConfig

                if self._quantization == "int8":
                    model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
                    self._logger.info("Using INT8 quantization")
                elif self._quantization == "int4":
                    model_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                    )
                    self._logger.info("Using INT4 quantization")
            elif self._quantization != "none" and device != "cuda":
                self._logger.warning(
                    f"Quantization '{self._quantization}' requires CUDA, using fp16/fp32 instead"
                )

            # Load model
            self._model = Qwen3VLForConditionalGeneration.from_pretrained(
                self._model_name,
                **model_kwargs,
            )

            # Move to device if needed
            if device in ("cpu", "mps"):
                self._model = self._model.to(device)

            # Load processor
            self._processor = AutoProcessor.from_pretrained(self._model_name)

            load_time = time.time() - start_time
            self._logger.info(f"Vision model loaded in {load_time:.1f}s")

        except ImportError as e:
            error_msg = (
                f"Failed to import required packages: {e}. "
                "Install with: uv pip install -e '.[vision]'"
            )
            self._logger.error(error_msg)
            raise RuntimeError(error_msg) from e

        except Exception as e:
            error_msg = f"Failed to load vision model '{self._model_name}': {e}"
            self._logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def _cleanup_processor(self) -> None:
        """Clean up model resources."""
        if self._model is not None:
            del self._model
            self._model = None

        if self._processor is not None:
            del self._processor
            self._processor = None

        self._previous_description = None
        self._frame_count = 0

        # Try to free GPU memory
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            pass

        self._logger.info("Vision processor cleaned up")

    def _process_event(self, event: IEvent) -> list[IEvent]:
        """Process camera frame event.

        Uses frame skipping and change detection to only emit events
        when the scene changes significantly.

        Args:
            event: CameraFrameEvent to process

        Returns:
            List containing VisionDescriptionEvent if scene changed,
            empty list otherwise
        """
        if event.event_type != "camera.frame":
            return []

        # Increment frame counter
        self._frame_count += 1

        # Skip frames according to frame_skip setting
        if self._frame_count % self._frame_skip != 0:
            return []

        if self._model is None or self._processor is None:
            self._logger.error("Model not initialized")
            return []

        # Extract frame data
        frame_data = event.data
        frame = frame_data["frame"]
        frame_number = frame_data["frame_number"]
        frame_timestamp = event.timestamp.timestamp()  # Convert datetime to Unix timestamp

        self._logger.debug(f"Processing frame {frame_number}")

        start_time = time.time()

        try:
            # Generate description
            description = self._generate_description(frame)

            if description is None:
                return []

            processing_time = time.time() - start_time

            # Compute change score
            change_score = self._compute_change_score(description)

            self._logger.debug(
                f"Frame {frame_number}: change_score={change_score:.3f}, "
                f"threshold={self._change_threshold}, time={processing_time:.2f}s"
            )

            # Check if change exceeds threshold
            if change_score < self._change_threshold:
                self._logger.debug(f"Change score {change_score:.3f} below threshold, skipping")
                return []

            # Update previous description
            self._previous_description = description

            # Create vision description event
            vision_event = VisionDescriptionEvent.create(
                source=self._processor_id,
                description=description,
                frame_number=frame_number,
                frame_timestamp=frame_timestamp,
                processing_time=processing_time,
                change_score=change_score,
                model_name=self._model_name,
                source_event_id=str(id(event)),
            )

            self._logger.info(
                f"Vision: '{description}' "
                f"(frame={frame_number}, change={change_score:.2f}, time={processing_time:.2f}s)"
            )

            return [vision_event]

        except Exception as e:
            self._logger.error(f"Vision processing failed: {e}", exc_info=True)
            return []

    def _generate_description(self, frame) -> Optional[str]:
        """Generate a description of the frame using the VLM.

        Args:
            frame: Camera frame (numpy array, BGR format from OpenCV)

        Returns:
            Text description of the image, or None if generation fails
        """
        try:
            import torch
            from PIL import Image
            from qwen_vl_utils import process_vision_info

            # Convert BGR (OpenCV) to RGB
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame_rgb = frame[:, :, ::-1]
            else:
                frame_rgb = frame

            # Convert to PIL Image
            pil_image = Image.fromarray(frame_rgb)

            # Prepare messages in Qwen3-VL format
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": pil_image},
                        {"type": "text", "text": self._prompt},
                    ],
                }
            ]

            # Apply chat template
            text = self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            # Process vision info
            image_inputs, video_inputs = process_vision_info(messages)

            # Prepare inputs
            inputs = self._processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )

            # Move inputs to model device
            inputs = inputs.to(self._model.device)

            # Generate
            with torch.no_grad():
                generated_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=self._max_tokens,
                    do_sample=False,  # Greedy decoding for consistency
                )

            # Trim input tokens and decode
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]

            output_text = self._processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

            return output_text.strip()

        except Exception as e:
            self._logger.error(f"Description generation failed: {e}", exc_info=True)
            return None

    def _compute_change_score(self, new_description: str) -> float:
        """Compute how different the new description is from the previous one.

        Uses word-level Jaccard distance (1 - Jaccard similarity).

        Args:
            new_description: The new description to compare

        Returns:
            Change score between 0.0 (identical) and 1.0 (completely different)
        """
        if self._previous_description is None:
            return 1.0  # First frame always counts as changed

        # Tokenize into words (simple whitespace split, lowercase)
        prev_words = set(self._previous_description.lower().split())
        new_words = set(new_description.lower().split())

        # Handle edge case of empty descriptions
        if not prev_words and not new_words:
            return 0.0
        if not prev_words or not new_words:
            return 1.0

        # Compute Jaccard similarity
        intersection = len(prev_words & new_words)
        union = len(prev_words | new_words)

        jaccard_similarity = intersection / union

        # Return change score (inverse of similarity)
        return 1.0 - jaccard_similarity
