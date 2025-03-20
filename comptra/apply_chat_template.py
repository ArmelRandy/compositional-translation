import warnings


def apply_chat_template(model_name_or_path: str):
    """
    Applying a chat template when relevant

    Return a function which takes as input a messages and returns it as an instruction
    with the relevant template words (<|user|>, [INST], <|assistant|> etc.)
    Messages is either:
    - a string (the request itself)
    - a list of dictionaries {"role": "system/user/assistant", "content": "..."}
    """
    if model_name_or_path in [
        "TheBloke/Mistral-7B-Instruct-v0.2-AWQ",
        "mistralai/Ministral-8B-Instruct-2410",
    ]:

        def f(messages):
            if isinstance(messages, str):
                prompt = f"<s>[INST] {messages} [/INST]"
            else:
                assert all(
                    [isinstance(message, dict) for message in messages]
                ), "The argument should be a list of dictionaries"
                prompt = "<s> "
                for i, message in enumerate(messages):
                    content = message["content"]
                    if i % 2 == 0:
                        # user turn
                        prompt += f"[INST] {content} [/INST]"
                    else:
                        # assistant turn
                        prompt += f"{content}</s> "
            return prompt

        return f
    elif model_name_or_path in [
        "TheBloke/zephyr-7B-beta-AWQ",
    ]:

        def f(messages):
            if isinstance(messages, str):
                prompt = (
                    f"<|system|>\n</s>\n<|user|>\n{messages}\n</s>\n<|assistant|>\n"
                )
                return prompt
            else:
                assert all(
                    [isinstance(message, dict) for message in messages]
                ), "The argument should be a list of dictionaries"
                header = ""
                prompt = ""
                for message in messages:
                    role = message["role"]
                    content = message["content"]
                    if role == "system":
                        header = f"<|system|>\n{content.strip()}</s> \n"
                    elif role == "user":
                        prompt += f"<|user|>\n{content}</s> "
                    else:
                        assert (
                            role == "assistant"
                        ), f"The role should be assistant, got '{role}' instead."
                        prompt += f"\n<|assistant|>\n{content}</s> \n"
                return header + prompt  # + "\n<|assistant|>\n"

        return f

    elif model_name_or_path in [
        "TheBloke/Llama-2-13B-Chat-AWQ",
        "TheBloke/Llama-2-70B-Chat-AWQ",
        "meta-llama/Llama-2-7b-chat-hf",
        "meta-llama/Llama-2-13b-chat-hf",
    ]:

        def f(messages):
            short = "You are a helpful, respectful and honest assistant."
            long = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
            if isinstance(messages, str):
                prompt = f"<s> [INST] <<SYS>>\n{short}\n<</SYS>>\n\n{messages} [/INST]"
                return prompt
            else:
                assert all(
                    [isinstance(message, dict) for message in messages]
                ), "The argument should be a list of dictionaries"
                header = f"[INST] <<SYS>>\n{short}\n<</SYS>>\n\n"
                prompt = ""
                for i, message in enumerate(messages):
                    role = message["role"]
                    content = message["content"]
                    if role == "system":
                        header = f"<s> [INST] <<SYS>>\n{content.strip()}\n<</SYS>>\n\n"
                    elif role == "user":
                        if i <= 1:
                            prompt += f"{content} [/INST]"
                        else:
                            prompt += f"<s> [INST] {content} [/INST]"
                    else:
                        assert (
                            role == "assistant"
                        ), f"The role should be assistant, got '{role}' instead."
                        prompt += f" {content} </s>"
                return header + prompt

        return f
    elif model_name_or_path in ["TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ"]:
        return lambda prompt: f"<s>[INST]{prompt}[/INST]"
    elif model_name_or_path in [
        "meta-llama/Meta-Llama-3-8B-Instruct",
        "casperhansen/llama-3-70b-instruct-awq",
    ]:

        def f(messages):
            if isinstance(messages, str):
                prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{messages}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
                return prompt
            else:
                assert all(
                    [isinstance(message, dict) for message in messages]
                ), "The argument should be a list of dictionaries"
                header = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|>"
                prompt = ""
                for message in messages:
                    role = message["role"]
                    content = message["content"]
                    if role == "system":
                        header = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{content.strip()}<|eot_id|>"
                    elif role == "user":
                        prompt += f"<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>"
                    else:
                        assert (
                            role == "assistant"
                        ), f"The role should be assistant, got '{role}' instead."
                        prompt += f"<|start_header_id|>assistant<|end_header_id|>\n\n{content}<|eot_id|>"
                # return header + prompt
                return (
                    header
                    + prompt
                    + "<|start_header_id|>assistant<|end_header_id|>\n\n"
                )

        return f
    elif model_name_or_path in [
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4",
        "meta-llama/Llama-3.2-1B-Instruct",
        "meta-llama/Llama-3.2-3B-Instruct",
        # "casperhansen/llama-3.3-70b-instruct-awq"
    ]:

        def f(messages):
            system = "You are a helpful assistant."
            if isinstance(messages, str):
                prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n{system}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{messages}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
                return prompt
            else:
                assert all(
                    [isinstance(message, dict) for message in messages]
                ), "The argument should be a list of dictionaries"
                header = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>"
                prompt = ""
                for message in messages:
                    role = message["role"]
                    content = message["content"]
                    if role == "system":
                        header = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n{content.strip()}<|eot_id|>"
                    elif role == "user":
                        prompt += f"<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>"
                    else:
                        assert (
                            role == "assistant"
                        ), f"The role should be assistant, got '{role}' instead."
                        prompt += f"<|start_header_id|>assistant<|end_header_id|>\n\n{content}<|eot_id|>"
                # return header + prompt
                return (
                    header
                    + prompt
                    + "<|start_header_id|>assistant<|end_header_id|>\n\n"
                )

        return f
    elif model_name_or_path in [
        "google/gemma-2-2b-it",
        "google/gemma-2-9b-it",
        "google/gemma-2-27b-it",
    ]:

        def f(messages):
            if isinstance(messages, str):
                prompt = f"<bos><start_of_turn>user\n{messages}<end_of_turn>\n<start_of_turn>model\n"
                return prompt
            else:
                assert all(
                    [isinstance(message, dict) for message in messages]
                ), "The argument should be a list of dictionaries"
                prompt = "<bos>"
                for message in messages:
                    role = message["role"]
                    content = message["content"]
                    if role == "user":
                        prompt += f"<start_of_turn>{role}\n{content}<end_of_turn>\n"
                    else:
                        assert (
                            role == "assistant"
                        ), f"The role should be assistant, got '{role}' instead."
                        prompt += f"<start_of_turn>model\n{content}<end_of_turn>\n"
                # return prompt
                return prompt + "<start_of_turn>model\n"

        return f
    elif model_name_or_path in [
        "CohereForAI/c4ai-command-r-08-2024",
        "CohereForAI/c4ai-command-r-plus-08-2024",
    ]:

        def f(messages):
            if isinstance(messages, str):
                prompt = f"<BOS_TOKEN><|START_OF_TURN_TOKEN|><|USER_TOKEN|>{messages}<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>"
                return prompt
            else:
                assert all(
                    [isinstance(message, dict) for message in messages]
                ), "The argument should be a list of dictionaries"
                header = ""
                prompt = ""
                for message in messages:
                    role = message["role"]
                    content = message["content"]
                    if role == "system":
                        header = f"<BOS_TOKEN><|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>{content.strip()}<|END_OF_TURN_TOKEN|>"
                    elif role == "user":
                        prompt += f"<|START_OF_TURN_TOKEN|><|USER_TOKEN|>{content}<|END_OF_TURN_TOKEN|>"
                    else:
                        assert (
                            role == "assistant"
                        ), f"The role should be assistant, got '{role}' instead."
                        prompt += f"<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>{content}<|END_OF_TURN_TOKEN|>"
                # return header + prompt
                return header + prompt + "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>"

        return f
    elif model_name_or_path in [
        "command-r-08-2024",
        "command-r-plus-08-2024",
        "command-r7b-12-2024",
    ]:

        def f(messages):
            if isinstance(messages, str):
                return messages
            else:
                assert all(
                    [isinstance(message, dict) for message in messages]
                ), "The argument should be a list of dictionaries"
                new = []
                for i, message in enumerate(messages):
                    if i == 0:
                        if message["role"] != "system":
                            new.append(
                                {
                                    "role": "SYSTEM",
                                    "message": "You are a helpful assistant",
                                }
                            )
                            role = "User" if message["role"] == "user" else "Chatbot"
                            new.append({"role": role, "message": message["content"]})
                        else:
                            new.append(
                                {"role": "SYSTEM", "message": message["content"]}
                            )
                    else:
                        role = "User" if message["role"] == "user" else "Chatbot"
                        new.append({"role": role, "message": message["content"]})
                return new

        return f
    else:
        warnings.warn(
            f"No ift template should be manually applied when using {model_name_or_path}. Feel free to ignore this warning if it is the expected behaviour."
        )
        return lambda prompt: prompt


if __name__ == "__main__":
    print(apply_chat_template("google/gemma-2-2b-it")("I want to eat your pancreas."))
    print(
        apply_chat_template("command-r-08-2024")(
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello, how are you today?"},
                {"role": "assistant", "content": "I am good, how can I help you?"},
                {
                    "role": "user",
                    "content": "Can you explain the origin of the pyramids?",
                },
            ]
        )
    )
