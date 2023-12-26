class LabelPromptsProcessor:
    """
    This class takes in label and prompt dictionary, initialises prompts
    and returns label corresponding to a prompt.
    """

    def __init__(self, label_and_prompt):
        self.label_and_prompt = label_and_prompt
        self.prompts = [
            prompt_text
            for prompts in list(self.label_and_prompt.values())
            for prompt_text in prompts
        ]

    def get_label(self, desc):
        for label, prompts in self.label_and_prompt.items():
            if desc in prompts:
                output_label = label
        return output_label
