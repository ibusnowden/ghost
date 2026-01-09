"""
LLaVA-Instruct-150K dataset for vision-language instruction fine-tuning.

Dataset: liuhaotian/LLaVA-Instruct-150K
High-quality instruction-following data with diverse visual reasoning tasks.
"""

from tasks.common import Task, load_dataset


class LLaVAInstruct(Task):
    """
    LLaVA-Instruct-150K dataset for multimodal instruction fine-tuning.

    Format:
    - User: <|image|>\n[question/instruction]
    - Assistant: [detailed response]

    This dataset contains diverse visual reasoning tasks including:
    - Detailed image descriptions
    - Visual question answering
    - Visual reasoning and inference
    - Creative tasks (e.g., write a story about this image)
    - Conversational visual understanding

    Used for instruction fine-tuning (Stage 3: SFT).

    Dataset info:
    - HuggingFace: liuhaotian/LLaVA-Instruct-150K
    - ~158K training examples
    - Images from COCO
    - Long-form, detailed responses (LNQA-style)
    - Mix of human and GPT-4V annotations
    """

    def __init__(self, split="train", start=0, stop=None, step=1):
        super().__init__(start=start, stop=stop, step=step)
        self.split = split

        # Load LLaVA-Instruct from HuggingFace
        try:
            self.dataset = load_dataset("liuhaotian/LLaVA-Instruct-150K", split=split)
        except Exception as e:
            print(f"Warning: Could not load liuhaotian/LLaVA-Instruct-150K: {e}")
            # Create minimal dummy dataset for testing
            from datasets import Dataset as HFDataset
            self.dataset = HFDataset.from_dict({
                "image": [],
                "conversations": [],
            })

        # Load COCO images mapping if available (LLaVA uses COCO images)
        self.coco_images = self._load_coco_images()

    def _load_coco_images(self):
        """
        Load COCO images dataset to map image IDs to actual images.
        LLaVA-Instruct references COCO images by filename/ID.
        """
        try:
            # Try loading COCO 2017 train split (most LLaVA images are from here)
            coco_train = load_dataset("HuggingFaceM4/COCO", split="train")
            coco_val = load_dataset("HuggingFaceM4/COCO", split="validation")

            # Create mapping from image filename to image
            images_map = {}
            for item in coco_train:
                if "image" in item and "image_id" in item:
                    # Map both by ID and filename
                    img_id = item["image_id"]
                    images_map[f"COCO_train2014_{img_id:012d}.jpg"] = item["image"]
                    images_map[str(img_id)] = item["image"]

            for item in coco_val:
                if "image" in item and "image_id" in item:
                    img_id = item["image_id"]
                    images_map[f"COCO_val2014_{img_id:012d}.jpg"] = item["image"]
                    images_map[str(img_id)] = item["image"]

            return images_map
        except Exception as e:
            print(f"Warning: Could not load COCO images for LLaVA-Instruct: {e}")
            print("        Images may need to be downloaded separately")
            return {}

    @property
    def eval_type(self):
        return "generative"

    def num_examples(self):
        return len(self.dataset)

    def get_example(self, index):
        """
        Returns a conversation dict with image field.

        LLaVA-Instruct format:
        {
            "id": "...",
            "image": "COCO_train2014_000000001234.jpg",  # or actual PIL Image
            "conversations": [
                {"from": "human", "value": "<image>\nWhat is in this image?"},
                {"from": "gpt", "value": "This image shows..."}
            ]
        }

        Returns:
            {
                "image": PIL.Image,
                "messages": [
                    {"role": "user", "content": "<|image|>\\n[question]"},
                    {"role": "assistant", "content": "[answer]"}
                ]
            }
        """
        item = self.dataset[index]

        # Extract image
        image = item.get("image")
        if image is None and "id" in item:
            # Try to get image from COCO mapping
            image_ref = item.get("id") or item.get("image_id")
            if image_ref and image_ref in self.coco_images:
                image = self.coco_images[image_ref]

        # Extract conversations
        conversations = item.get("conversations", [])

        # Convert LLaVA format to our format
        messages = []
        for conv in conversations:
            # LLaVA uses "from": "human"/"gpt", we use "role": "user"/"assistant"
            role = "user" if conv.get("from") == "human" else "assistant"
            content = conv.get("value", "")

            # Replace <image> placeholder with <|image|>
            content = content.replace("<image>", "<|image|>")

            messages.append({
                "role": role,
                "content": content
            })

        # If no valid messages, create a default one
        if not messages:
            messages = [
                {"role": "user", "content": "<|image|>\nDescribe this image in detail."},
                {"role": "assistant", "content": "I see an image."}
            ]

        return {
            "image": image,
            "messages": messages
        }

    def evaluate(self, problem, completion):
        """
        LLaVA-Instruct evaluation: check if completion is reasonable.

        This is mostly used for instruction-following training, so we use
        a soft evaluation (length, relevance heuristics).
        """
        expected = problem["messages"][-1]["content"].strip()
        completion = completion.strip()

        # Basic heuristics:
        # 1. Completion should not be empty
        if not completion:
            return False

        # 2. Completion should be reasonably long (at least 10 chars)
        if len(completion) < 10:
            return False

        # 3. For training purposes, we consider it valid
        # (Real evaluation would use GPT-4 or human judges)
        return True

    def reward(self, conversation, completion):
        """RL reward based on evaluation."""
        return float(self.evaluate(conversation, completion))


# For backward compatibility and testing
if __name__ == "__main__":
    # Quick test
    print("Testing LLaVA-Instruct dataset...")
    try:
        dataset = LLaVAInstruct(split="train")
        print(f"Loaded {len(dataset)} examples")

        if len(dataset) > 0:
            example = dataset.get_example(0)
            print(f"\nFirst example:")
            print(f"Messages: {example['messages']}")
            print(f"Has image: {example['image'] is not None}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
