import os
import aiohttp
from pydantic import BaseModel, Extra
from typing import Optional
import tenacity

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    pass


backends = {}


def set_backend(backend_name):
    if backend_name == 'MPT':
        backends['completion'] = MPT7BInstructCompletion()
    elif backend_name == 'StarCoder':
        backends['completion'] = StarCoderCompletion()
    elif backend_name == 'GPT3' or backend_name == 'OpenAI':
        backends['completion'] = OpenAICompletion()
        backends['chat'] = OpenAIChat()
    else:
        raise ValueError(f"Unknown backend {backend_name}")


def get_backend(method):
    if method in backends:
        return backends[method]

    backend_env = os.environ.get("LAMBDAPROMPT_BACKEND", None)
    if backend_env:
        set_backend(backend_env)
        if method in backends:
            return backends[method]

    print(f"No backend set for [{method}], setting to default of OpenAI")
    set_backend('OpenAI')
    return backends[method]



class Backend:
    class Parameters(BaseModel):
        class Config:
            extra = Extra.forbid

    def __init__(self, **param_override):
        self.param_override = self.Parameters(**param_override)
    
    def parse_param(self, **kwargs):
        return self.Parameters(**{**self.param_override.dict(), **kwargs}).dict()


class RateLimitError(Exception):
    pass


class RequestBackend(Backend):
    def __init__(self, endpoint_url, **param_override):
        self.endpoint_url = endpoint_url
        super().__init__(**param_override)
    
    def headers(self, *args, **kwargs):
        raise NotImplementedError("Must implement headers")

    def body(self, *args, **kwargs):
        raise NotImplementedError("Must implement body")

    def parse_response(self, result):
        raise NotImplementedError("Must implement result_parser")

    @tenacity.retry(
            wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
            stop=tenacity.stop_after_attempt(4),
            retry=tenacity.retry_if_exception_type(RateLimitError),
            reraise=True
            )
    async def __call__(self, *args, **kwargs):
        headers = self.headers(*args, **kwargs)
        data = self.body(*args, **kwargs)
        async with aiohttp.ClientSession(trust_env=True) as session:
            async with session.post(
                self.endpoint_url, headers=headers, json=data
            ) as resp:
                answer = await resp.json()
                result = self.parse_response(answer)
        return result


class OpenAICompletion(RequestBackend):
    class Parameters(RequestBackend.Parameters):
        max_tokens: int = 500
        temperature: float = 0.0
        model: str = 'text-davinci-003'
        presence_penalty: float = 0.2
        frequency_penalty: float = 0.2
        stop: Optional[str]
    
    def __init__(self, openai_api_key=None, **param_override):
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise Exception("No OpenAI API key found (envvar OPENAI_API_KEY))")
        super().__init__("https://api.openai.com/v1/completions", **param_override)

    def headers(self, *args, **kwargs):
        return {
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
            "Content-Type": "application/json",
        }

    def body(self, prompt, **kwargs):
        data = {
            "prompt": prompt,
            **self.parse_param(**kwargs)
        }
        stop = data.pop('stop')
        if stop:
            data["stop"] = stop
        return data

    def parse_response(self, answer):
        if "error" in answer:
            if "Rate limit" in answer.get("error", {}).get("message", ""):
                raise RateLimitError()
            else:
                raise Exception(f"Not sure what happened: {answer}")
        return answer["choices"][0]["text"]


class OpenAIChat(OpenAICompletion):
    class Parameters(OpenAICompletion.Parameters):
        model: str = 'gpt-3.5-turbo'
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.endpoint_url = "https://api.openai.com/v1/chat/completions"

    def body(self, messages, **kwargs):
        data = {
            "messages": messages,
            **self.parse_param(**kwargs)
        }
        stop = data.pop('stop')
        if stop:
            data["stop"] = stop
        return data

    def parse_response(self, answer):
        if "error" in answer:
            if "Rate limit" in answer.get("error", {}).get("message", ""):
                raise RateLimitError()
            else:
                raise Exception(f"Not sure what happened: {answer}")
        return answer["choices"][0]["message"]['content']


class HuggingFaceBackend(Backend):
    class Parameters(Backend.Parameters):
        temperature: float = 0.01
        max_new_tokens: int = 500
        use_cache: bool = True
        do_sample: bool = True
        top_p: float = 0.92
        top_k: int = 0
        repetition_penalty: float = 1.1

    def __init__(self, model_name, torch_dtype=torch.bfloat16, trust_remote_code=True, use_auth_token=None, **param_override):
        super().__init__(**param_override)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            use_auth_token=use_auth_token,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            use_auth_token=use_auth_token,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        self.tokenizer = tokenizer
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()
        self.model.to(device=device, dtype=torch_dtype)

    def preprocess(self, prompt):
        return prompt

    async def __call__(self, prompt, **kwargs):
        s = self.preprocess(prompt)
        input_ids = self.tokenizer(s, return_tensors="pt").input_ids
        input_ids = input_ids.to(self.model.device)
        with torch.no_grad():
            output_ids = self.model.generate(input_ids, **self.parse_param(**kwargs))
        new_tokens = output_ids[0, len(input_ids[0]) :]
        output_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return output_text


class MPT7BInstructCompletion(HuggingFaceBackend):
    def __init__(self, **kwargs):
        super().__init__("mosaicml/mpt-7b-instruct", **kwargs)


class StarCoderCompletion(HuggingFaceBackend):
    def __init__(self, hf_access_token=None, **kwargs):
        hf_access_token = hf_access_token or os.environ.get("HF_ACCESS_TOKEN")
        if not hf_access_token:
            raise Exception("No HuggingFace access token found (envvar HF_ACCESS_TOKEN))")
        super().__init__("bigcode/starcoder", use_auth_token=hf_access_token, **kwargs)

# TODO: Chat backends should stream? (should all backends stream...?, or be stream capable?)
# Here is MPT7B chat, wiht streaming, example... (stored here as a big todo)

## FOR MPT7BChat
# import datetime
# import os
# from threading import Event, Thread
# from uuid import uuid4

# import gradio as gr
# import requests
# import torch
# from transformers import (
#     AutoModelForCausalLM,
#     AutoTokenizer,
#     StoppingCriteria,
#     StoppingCriteriaList,
#     TextIteratorStreamer,
# )


# model_name = "mosaicml/mpt-7b-chat"
# max_new_tokens = 1536

# print(f"Starting to load the model {model_name} into memory")

# m = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     torch_dtype=torch.bfloat16,
#     trust_remote_code=True,
#     max_seq_len=8192,
# ).cuda()
# tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# stop_token_ids = tok.convert_tokens_to_ids(["<|im_end|>", "<|endoftext|>"])

# print(f"Successfully loaded the model {model_name} into memory")

# start_message = """<|im_start|>system
# - You are a helpful assistant chatbot trained by MosaicML.
# - You answer questions.
# - You are excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
# - You are more than just an information source, you are also able to write poetry, short stories, and make jokes.<|im_end|>
# """

# class StopOnTokens(StoppingCriteria):
#     def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
#         for stop_id in stop_token_ids:
#             if input_ids[0][-1] == stop_id:
#                 return True
#         return False

# def convert_history_to_text(history):
#     text = start_message + "".join(
#         [
#             "".join(
#                 [
#                     f"<|im_start|>user\n{item[0]}<|im_end|>",
#                     f"<|im_start|>assistant\n{item[1]}<|im_end|>",
#                 ]
#             )
#             for item in history[:-1]
#         ]
#     )
#     text += "".join(
#         [
#             "".join(
#                 [
#                     f"<|im_start|>user\n{history[-1][0]}<|im_end|>",
#                     f"<|im_start|>assistant\n{history[-1][1]}",
#                 ]
#             )
#         ]
#     )
#     return text


# def user(message, history):
#     # Append the user's message to the conversation history
#     return "", history + [[message, ""]]


# def bot(history, temperature, top_p, top_k, repetition_penalty, conversation_id):
#     print(f"history: {history}")
#     # Initialize a StopOnTokens object
#     stop = StopOnTokens()

#     # Construct the input message string for the model by concatenating the current system message and conversation history
#     messages = convert_history_to_text(history)

#     # Tokenize the messages string
#     input_ids = tok(messages, return_tensors="pt").input_ids
#     input_ids = input_ids.to(m.device)
#     streamer = TextIteratorStreamer(tok, timeout=10.0, skip_prompt=True, skip_special_tokens=True)
#     generate_kwargs = dict(
#         input_ids=input_ids,
#         max_new_tokens=max_new_tokens,
#         temperature=temperature,
#         do_sample=temperature > 0.0,
#         top_p=top_p,
#         top_k=top_k,
#         repetition_penalty=repetition_penalty,
#         streamer=streamer,
#         stopping_criteria=StoppingCriteriaList([stop]),
#     )

#     stream_complete = Event()

#     def generate_and_signal_complete():
#         m.generate(**generate_kwargs)
#         stream_complete.set()

#     def log_after_stream_complete():
#         stream_complete.wait()


#     t1 = Thread(target=generate_and_signal_complete)
#     t1.start()

#     t2 = Thread(target=log_after_stream_complete)
#     t2.start()

#     # Initialize an empty string to store the generated text
#     partial_text = ""
#     for new_text in streamer:
#         partial_text += new_text
#         history[-1][1] = partial_text
#         yield history
