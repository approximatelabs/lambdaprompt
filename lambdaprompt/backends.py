import os
import aiohttp
from pydantic import BaseModel
from typing import Optional, Union, List
import tenacity


backends = {}


def set_backend(backend_name):
    if backend_name == 'MPT':
        backends['completion'] = MPT7BInstructCompletion()
    elif backend_name == 'StarCoder':
        backends['completion'] = StarCoderCompletion()
    elif backend_name == 'StarCoderGGML':
        backends['completion'] = StarCoderGGMLQuantizedCompletion()
    elif backend_name == 'SantaCoderGGML':
        backends['completion'] = SantaCoderGGMLQuantizedCompletion()
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
            extra = 'forbid'

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
        model: str = 'gpt-3.5-turbo-instruct'
        presence_penalty: float = 0.2
        frequency_penalty: float = 0.2
        stop: Optional[Union[str, List[str]]] = None
    
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
        return {k: v for k, v in data.items() if v is not None}

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
        return {k: v for k, v in data.items() if v is not None}

    def parse_response(self, answer):
        if "error" in answer:
            if "Rate limit" in answer.get("error", {}).get("message", ""):
                raise RateLimitError()
            else:
                raise Exception(f"Not sure what happened: {answer}")
        return answer["choices"][0]["message"]['content']


class CTransformersBackend(Backend):
    # https://github.com/marella/ctransformers

    class Parameters(Backend.Parameters):
        max_new_tokens: int = 200
        temperature: float = 0.01
        top_p: float = 0.92
        top_k: int = 0
        repetition_penalty: float = 1.1
        stop: Optional[Union[str, List[str]]]
    
    def __init__(self, model_name, model_type, **param_override):
        try:
            from ctransformers import AutoModelForCausalLM
        except ImportError:
            raise ImportError("You must install ctransformers to use this backend (`pip install ctransformers`)")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            model_type=model_type)
        super().__init__(**param_override)
    
    def preprocess(self, prompt):
        return prompt
    
    async def __call__(self, prompt, **kwargs):
        prompt = self.preprocess(prompt)
        genkwargs = self.parse_param(**kwargs)
        max_new_tokens = genkwargs.pop("max_new_tokens")

        tokens = self.model.tokenize(prompt)
        stop = genkwargs.pop("stop", None) or []
        if isinstance(stop, str):
            stop = [stop]
        end_ids = [self.model.tokenize(x) for x in stop]

        def should_stop(response_tokens):
            for end in end_ids:
                if all(x == y for x, y in zip(response_tokens[-len(end):], end)):
                    return True
            if len(response_tokens) >= max_new_tokens:
                return True
            return False

        response = []
        for token in self.model.generate(tokens, **genkwargs):
            response.append(token)
            if should_stop(response):
                break

        return self.model.detokenize(response)


class StarCoderGGMLQuantizedCompletion(CTransformersBackend):
    def __init__(self, **kwargs):
        super().__init__("nouamanetazi/starcoder-ggml", model_type='starcoder', **kwargs)


class SantaCoderGGMLQuantizedCompletion(CTransformersBackend):
    def __init__(self, **kwargs):
        super().__init__("danforbes/santacoder-ggml-q4_1", model_type='starcoder', **kwargs)


class HuggingFaceBackend(Backend):
    class Parameters(Backend.Parameters):
        temperature: float = 0.01
        max_new_tokens: int = 500
        use_cache: bool = True
        do_sample: bool = True
        top_p: float = 0.92
        top_k: int = 0
        repetition_penalty: float = 1.1
        stop: Optional[Union[str, List[str]]]

    def __init__(self, model_name, torch_dtype=None, trust_remote_code=True, use_auth_token=None, use_device_map=True, load_config=True, **param_override):
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        torch_dtype = torch_dtype or torch.bfloat16
        super().__init__(**param_override)
        init_kwargs = {
            "torch_dtype": torch_dtype,
            "trust_remote_code": trust_remote_code,
            "use_auth_token": use_auth_token,
        }
        if load_config:
            init_kwargs['config'] = AutoConfig.from_pretrained(
                model_name,
                trust_remote_code=True
            )
        if use_device_map:
            init_kwargs['device_map'] = "auto"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **init_kwargs
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            use_auth_token=use_auth_token,
            **({"device_map":"auto"} if use_device_map else {})
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        self.tokenizer = tokenizer
        self.model.eval()

    def preprocess(self, prompt):
        return prompt

    async def __call__(self, prompt, **kwargs):
        import torch
        from transformers import StoppingCriteriaList
        genkwargs = self.parse_param(**kwargs)

        def get_stopping_for_ends(end_ids):
            # This assumes that stop is a nice "end_id" token
            # we're not decoding and checking the text, staying in id_land, so could cause some weirdness
            if len(end_ids) == 0:
                return StoppingCriteriaList([lambda *args, **kwargs: False])
            max_stop_length = max(x.shape[0] for x in end_ids)
            def stop_on_any(input_ids: torch.LongTensor, score: torch.FloatTensor, **kwargs) -> bool:
                last_tokens = input_ids[0, -max_stop_length:]
                for end_id in end_ids:
                    if torch.equal(last_tokens[-end_id.shape[0]:], end_id):
                        return True
                return False

            return StoppingCriteriaList([stop_on_any])
        
        stop = genkwargs.pop("stop", None) or []
        if isinstance(stop, str):
            stop = [stop]
        end_ids = [self.tokenizer(x, return_tensors="pt").input_ids[0].to(self.model.device) for x in stop]
        s = self.preprocess(prompt)
        input_ids = self.tokenizer(s, return_tensors="pt").input_ids
        input_ids = input_ids.to(self.model.device)
        with torch.no_grad():
            output_ids = self.model.generate(input_ids, stopping_criteria=get_stopping_for_ends(end_ids), **genkwargs)
        new_tokens = output_ids[0, len(input_ids[0]) :]
        for end_id in end_ids:
            if torch.equal(new_tokens[-end_id.shape[0]:], end_id):
                new_tokens = new_tokens[:-end_id.shape[0]]
        output_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return output_text


class MPT7BInstructCompletion(HuggingFaceBackend):
    def __init__(self, **kwargs):
        super().__init__("mosaicml/mpt-7b-instruct", use_device_map=False, **kwargs)


class StarCoderCompletion(HuggingFaceBackend):
    def __init__(self, hf_access_token=None, **kwargs):
        hf_access_token = hf_access_token or os.environ.get("HF_ACCESS_TOKEN")
        if not hf_access_token:
            raise Exception("No HuggingFace access token found (envvar HF_ACCESS_TOKEN))")
        super().__init__("bigcode/starcoder", use_auth_token=hf_access_token, load_config=False, **kwargs)
