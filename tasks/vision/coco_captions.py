"""
COCO Captions dataset for vision-language pretraining.
Dataset: MS COCO 2017 with captions.
"""

from tasks.common import Task, load_dataset


class COCOCaptions(Task):
    """
    COCO Captions dataset for image captioning pretraining.

    Format:
    - User: <|image|>\nDescribe this image.
    - Assistant: [caption]

    Used for vision-language alignment (Stage 2: Mid-training).
    """

    def __init__(self, split="train", start=0, stop=None, step=1):
        super().__init__(start=start, stop=stop, step=step)
        self.split = split

        # Load COCO dataset from HuggingFace
        # Using HuggingFaceM4/COCO which has images + captions
        try:
            self.dataset = load_dataset("HuggingFaceM4/COCO", split=split)
        except Exception as e:
            print(f"Warning: Could not load HuggingFaceM4/COCO: {e}")
            print("Falling back to nlphuji/flickr30k for testing")
            # Fallback to a smaller dataset for testing
            self.dataset = load_dataset("nlphuji/flickr30k", split="test")

    @property
    def eval_type(self):
        return "generative"

    def num_examples(self):
        return len(self.dataset)

    def get_example(self, index):
        """
        Returns a conversation dict with image field.

        Returns:
            {
                "image": PIL.Image,
                "messages": [
                    {"role": "user", "content": "<|image|>\\nDescribe this image."},
                    {"role": "assistant", "content": "caption"}
                ]
            }
        """
        item = self.dataset[index]

        # Extract image and caption
        image = item.get("image") or item.get("jpg") or item.get("img")

        # Get caption (handle different formats)
        if "sentences" in item and isinstance(item["sentences"], dict):
            # Flickr30k format
            caption = item["sentences"]["raw"][0] if item["sentences"]["raw"] else "An image."
        elif "captions" in item and isinstance(item["captions"], list):
            # Some COCO formats
            caption = item["captions"][0] if item["captions"] else "An image."
        elif "caption" in item:
            # Standard format
            caption = item["caption"]
        else:
            caption = "An image."

        return {
            "image": image,
            "messages": [
                {
                    "role": "user",
                    "content": "<|image|>\nDescribe this image in detail."
                },
                {
                    "role": "assistant",
                    "content": caption
                }
            ]
        }

    def evaluate(self, problem, completion):
        """
        For captioning, evaluation is typically done with BLEU/CIDEr/METEOR.
        For now, we just do exact match (not very meaningful for captions).
        """
        # Extract the expected caption from problem
        expected = problem["messages"][-1]["content"]
        # Normalize whitespace
        expected = " ".join(expected.split())
        completion = " ".join(completion.split())
        return expected.lower() == completion.lower()

    def reward(self, conversation, completion):
        """
        For pretraining, we don't need sophisticated rewards.
        Just return 1.0 for any completion (unsupervised learning).
        """
        return 1.0
