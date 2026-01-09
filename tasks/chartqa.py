"""
ChartQA dataset for visual question answering about charts and plots.
Dataset: ChartQA - questions about charts, graphs, and data visualizations.
"""

from tasks.common import Task, load_dataset


class ChartQA(Task):
    """
    ChartQA dataset for chart and graph understanding.

    Format:
    - User: <|image|>\n[question]
    - Assistant: [answer]

    Focuses on understanding data visualizations like bar charts, line graphs,
    pie charts, and answering questions about them.

    Used for instruction fine-tuning (Stage 3: SFT).

    Dataset info:
    - HuggingFace: HuggingFace/chartqa or ahmed-masry/ChartQA
    - ~18K training examples, ~2K validation examples
    - Answer types: numeric values, text labels, yes/no
    - Chart types: bar, line, pie, scatter, etc.
    """

    def __init__(self, split="train", start=0, stop=None, step=1):
        super().__init__(start=start, stop=stop, step=step)
        self.split = split

        # Map split names
        split_map = {
            "train": "train",
            "val": "validation",
            "validation": "validation",
            "test": "test"
        }
        hf_split = split_map.get(split, split)

        # Load ChartQA from HuggingFace
        try:
            # Try ahmed-masry/ChartQA (most common)
            self.dataset = load_dataset("ahmed-masry/ChartQA", split=hf_split)
        except Exception as e:
            print(f"Warning: Could not load ahmed-masry/ChartQA: {e}")
            try:
                # Fallback to HuggingFace/chartqa
                self.dataset = load_dataset("HuggingFace/chartqa", split=hf_split)
            except Exception as e2:
                print(f"Warning: Could not load chartqa: {e2}")
                # Create minimal dummy dataset for testing
                from datasets import Dataset as HFDataset
                self.dataset = HFDataset.from_dict({
                    "image": [],
                    "question": [],
                    "answer": []
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

        # Extract image (chart/graph image)
        image = item.get("image") or item.get("png") or item.get("jpg") or item.get("img")

        # Extract question
        question = item.get("question") or item.get("query") or "What does this chart show?"

        # Extract answer
        # ChartQA has different answer formats depending on the split
        if "answer" in item:
            answer = item["answer"]
            # Handle list format
            if isinstance(answer, list):
                answer = answer[0] if answer else "0"
        elif "answers" in item:
            answers = item["answers"]
            if isinstance(answers, list) and answers:
                answer = answers[0]
            elif isinstance(answers, dict) and "answer" in answers:
                answer = answers["answer"][0] if answers["answer"] else "0"
            else:
                answer = "0"
        else:
            answer = "0"

        # Ensure answer is string
        answer = str(answer)

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
        ChartQA evaluation: check if completion matches the answer.

        For numeric answers: use relaxed numeric matching (allow small differences)
        For text answers: use soft text matching
        """
        expected = problem["messages"][-1]["content"].strip()
        completion = completion.strip()

        # Normalize both for comparison
        expected_lower = expected.lower()
        completion_lower = completion.lower()

        # Try numeric matching first (for numeric answers)
        try:
            expected_num = float(expected.replace(",", ""))
            # Extract first number from completion
            import re
            completion_nums = re.findall(r'-?\d+\.?\d*', completion)
            if completion_nums:
                completion_num = float(completion_nums[0].replace(",", ""))
                # Allow 5% relative error for numeric answers
                if expected_num == 0:
                    return abs(completion_num - expected_num) < 0.01
                else:
                    rel_error = abs(completion_num - expected_num) / abs(expected_num)
                    return rel_error < 0.05
        except (ValueError, IndexError):
            pass

        # Text matching for non-numeric answers
        # Soft matching: answer should appear in completion or vice versa
        if expected_lower in completion_lower:
            return True
        if completion_lower in expected_lower:
            return True

        # Exact match after normalization
        return expected_lower == completion_lower

    def reward(self, conversation, completion):
        """RL reward based on evaluation."""
        return float(self.evaluate(conversation, completion))


# For backward compatibility and testing
if __name__ == "__main__":
    # Quick test
    print("Testing ChartQA dataset...")
    try:
        dataset = ChartQA(split="train")
        print(f"Loaded {len(dataset)} examples")

        if len(dataset) > 0:
            example = dataset.get_example(0)
            print(f"\nFirst example:")
            print(f"Question: {example['messages'][0]['content']}")
            print(f"Answer: {example['messages'][1]['content']}")
            print(f"Has image: {example['image'] is not None}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
