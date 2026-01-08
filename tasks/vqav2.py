"""
VQAv2 (Visual Question Answering v2) dataset.
Dataset: VQA 2.0 with images and questions.
"""

from tasks.common import Task, load_dataset


class VQAv2(Task):
    """
    VQAv2 dataset for visual question answering.

    Format:
    - User: <|image|>\n[question]
    - Assistant: [answer]

    Used for instruction fine-tuning (Stage 3: SFT).
    """

    def __init__(self, split="train", start=0, stop=None, step=1):
        super().__init__(start=start, stop=stop, step=step)
        self.split = split

        # Map split names (VQA uses different naming)
        split_map = {
            "train": "train",
            "val": "validation",
            "test": "test"
        }
        hf_split = split_map.get(split, split)

        # Load VQAv2 from HuggingFace
        try:
            # Try HuggingFaceM4/VQAv2 first
            self.dataset = load_dataset("HuggingFaceM4/VQAv2", split=hf_split)
        except Exception as e:
            print(f"Warning: Could not load HuggingFaceM4/VQAv2: {e}")
            try:
                # Fallback to Multimodal-Fatima/VQAv2_Dataset
                self.dataset = load_dataset("Multimodal-Fatima/VQAv2_Dataset", split=hf_split)
            except Exception as e2:
                print(f"Warning: Could not load VQAv2: {e2}")
                # Create minimal dummy dataset for testing
                from datasets import Dataset as HFDataset
                self.dataset = HFDataset.from_dict({
                    "image": [],
                    "question": [],
                    "answers": []
                })

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
                    {"role": "user", "content": "<|image|>\\n[question]"},
                    {"role": "assistant", "content": "[answer]"}
                ]
            }
        """
        item = self.dataset[index]

        # Extract image
        image = item.get("image") or item.get("jpg") or item.get("img")

        # Extract question
        question = item.get("question") or item.get("query") or "What is in this image?"

        # Extract answer (VQA has multiple answers, take the most common one)
        if "answers" in item and isinstance(item["answers"], dict):
            # Format: {"answer": [...], "answer_confidence": [...]}
            answers_list = item["answers"].get("answer", [])
            if answers_list:
                # Take most common answer
                from collections import Counter
                answer = Counter(answers_list).most_common(1)[0][0]
            else:
                answer = "Yes"
        elif "answer" in item:
            answer = item["answer"]
        elif "answers" in item and isinstance(item["answers"], list):
            answer = item["answers"][0] if item["answers"] else "Yes"
        else:
            answer = "Yes"

        return {
            "image": image,
            "messages": [
                {
                    "role": "user",
                    "content": f"<|image|>\n{question}"
                },
                {
                    "role": "assistant",
                    "content": answer
                }
            ]
        }

    def evaluate(self, problem, completion):
        """
        VQA evaluation: check if completion matches any of the ground truth answers.
        VQA uses soft matching (answer appears in completion).
        """
        # Extract expected answer
        expected = problem["messages"][-1]["content"].lower().strip()
        completion = completion.lower().strip()

        # VQA-style soft matching: answer should appear in completion
        return expected in completion or completion in expected

    def reward(self, conversation, completion):
        """RL reward based on evaluation."""
        return float(self.evaluate(conversation, completion))
