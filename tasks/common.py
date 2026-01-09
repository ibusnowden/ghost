"""
Common utilities for vision-language tasks.
"""

from datasets import load_dataset as hf_load_dataset


# Re-export load_dataset from HuggingFace datasets
load_dataset = hf_load_dataset


class Task:
    """
    Base class for vision-language datasets.

    All task classes should inherit from this and implement:
    - num_examples(): return total number of examples
    - get_example(index): return a single example as a dict
    - evaluate(problem, completion): evaluate a completion (optional)
    - reward(conversation, completion): compute RL reward (optional)
    """

    def __init__(self, start=0, stop=None, step=1):
        """
        Initialize task with slicing parameters.

        Args:
            start: Start index for dataset slice
            stop: Stop index for dataset slice (None = end of dataset)
            step: Step size for dataset slice
        """
        self.start = start
        self.stop = stop
        self.step = step

    def __iter__(self):
        """Iterate over dataset with start/stop/step slicing."""
        total = self.num_examples()
        stop = self.stop if self.stop is not None else total
        for i in range(self.start, min(stop, total), self.step):
            yield self.get_example(i)

    def __len__(self):
        """Return number of examples in the dataset slice."""
        total = self.num_examples()
        stop = self.stop if self.stop is not None else total
        return max(0, (min(stop, total) - self.start + self.step - 1) // self.step)

    def num_examples(self):
        """Return total number of examples in the dataset."""
        raise NotImplementedError("Subclasses must implement num_examples()")

    def get_example(self, index):
        """
        Get a single example from the dataset.

        Args:
            index: Example index

        Returns:
            dict with structure:
            {
                "image": PIL.Image (optional, for vision tasks),
                "messages": [
                    {"role": "user", "content": "..."},
                    {"role": "assistant", "content": "..."}
                ]
            }
        """
        raise NotImplementedError("Subclasses must implement get_example()")

    @property
    def eval_type(self):
        """
        Return evaluation type: 'generative' or 'multiple_choice'.
        Default is 'generative'.
        """
        return "generative"

    def evaluate(self, problem, completion):
        """
        Evaluate a completion against the expected answer.

        Args:
            problem: The problem dict (from get_example)
            completion: Model's completion string

        Returns:
            bool or float: True/1.0 if correct, False/0.0 if incorrect
        """
        # Default: exact match on assistant message
        expected = problem["messages"][-1]["content"]
        return expected.strip() == completion.strip()

    def reward(self, conversation, completion):
        """
        Compute RL reward for a completion.

        Args:
            conversation: The conversation dict
            completion: Model's completion string

        Returns:
            float: Reward value (typically 0.0 or 1.0)
        """
        # Default: use evaluation result as reward
        return float(self.evaluate(conversation, completion))
