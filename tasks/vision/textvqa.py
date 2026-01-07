"""
TextVQA dataset for visual question answering with text in images.
Dataset: TextVQA - questions about text appearing in images.
"""

from tasks.common import Task, load_dataset


class TextVQA(Task):
    """
    TextVQA dataset for OCR-based visual question answering.

    Format:
    - User: <|image|>\n[question]
    - Assistant: [answer]

    Focuses on reading and understanding text in images.
    Used for instruction fine-tuning (Stage 3: SFT).
    """

    def __init__(self, split="train", start=0, stop=None, step=1):
        super().__init__(start=start, stop=stop, step=step)
        self.split = split

        # Map split names
        split_map = {
            "train": "train",
            "val": "validation",
            "test": "test"
        }
        hf_split = split_map.get(split, split)

        # Load TextVQA from HuggingFace
        try:
            self.dataset = load_dataset("textvqa", split=hf_split)
        except Exception as e:
            print(f"Warning: Could not load textvqa: {e}")
            # Create minimal dummy dataset
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
        question = item.get("question") or "What text is in this image?"

        # Extract answer (TextVQA has multiple answers)
        if "answers" in item:
            answers = item["answers"]
            if isinstance(answers, list) and answers:
                # Take first answer
                answer = answers[0]
            elif isinstance(answers, dict) and "answer" in answers:
                answer = answers["answer"][0] if answers["answer"] else "text"
            else:
                answer = "text"
        elif "answer" in item:
            answer = item["answer"]
        else:
            answer = "text"

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
        TextVQA evaluation: check if completion contains the answer.
        OCR tasks use soft matching.
        """
        expected = problem["messages"][-1]["content"].lower().strip()
        completion = completion.lower().strip()

        # Soft matching: answer should appear in completion
        return expected in completion or completion in expected

    def reward(self, conversation, completion):
        """RL reward based on evaluation."""
        return float(self.evaluate(conversation, completion))
