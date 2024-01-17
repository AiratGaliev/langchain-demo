from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.embeddings import HuggingFaceEmbeddings, HuggingFaceBgeEmbeddings

from llms.llamacpp import LlamaCpp
from utils.templates import chat_ml_template, alpaca_template, openchat_template, amazon_template, \
    solar_instruct_template


def load_llm(model_path: str, temperature: float = 0.0, ctx_size: int = 4096):
    return LlamaCpp(
        model_path=model_path,
        n_gpu_layers=-1,
        n_batch=384,
        n_ctx=ctx_size,
        max_tokens=ctx_size,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
        verbose=False,
        temperature=temperature,
        stop=["<|end_of_turn|>", "<|endoftext|>", "<|im_start|>", "<|im_end|>", "</s>", "Human:", "AI:",
              "Assistant:",
              "### System:", "### User:", "### Assistant:", "<|prompter|>", "<|assistant|>"],
    )


def load_emb(model_name: str, use_gpu: bool = False):
    device = 'cuda' if use_gpu else 'cpu'
    return HuggingFaceEmbeddings(
        model_name=model_name,
        cache_folder="/home/airat/MTEB",
        model_kwargs={'device': device}
    )


def load_gte_base_emb(use_gpu: bool = False):
    return load_emb(model_name="thenlper/gte-base", use_gpu=use_gpu)


def load_bge_emb(model_name: str, use_gpu: bool = True):
    device = 'cuda' if use_gpu else 'cpu'
    return HuggingFaceBgeEmbeddings(
        model_name=model_name,
        cache_folder="/home/airat/MTEB",
        model_kwargs={'device': device}
    )


def load_bge_base_angle_emb(use_gpu: bool = False):
    return load_bge_emb(model_name="khoa-klaytn/bge-base-en-v1.5-angle", use_gpu=use_gpu)


def load_bge_base_emb(use_gpu: bool = False):
    return load_bge_emb(model_name="BAAI/bge-base-en-v1.5", use_gpu=use_gpu)


def load_openhermes_llm(temperature: float = 0.0):
    return load_llm(model_path="/home/airat/LLMs/openhermes-2.5-mistral-7b.Q4_K_M.gguf", temperature=temperature)


def load_openhermes(system_prompt: str = "", temperature: float = 0.0):
    return chat_ml_template(system_prompt) | load_openhermes_llm(temperature=temperature)


def load_openhermes_16k_llm(temperature: float = 0.0):
    return load_llm(model_path="/home/airat/LLMs/openhermes-2.5-mistral-7b-16k.Q4_K_M.gguf", temperature=temperature,
                    ctx_size=16384)


def load_openhermes_16k(system_prompt: str = "", temperature: float = 0.0):
    return chat_ml_template(system_prompt) | load_openhermes_16k_llm(temperature=temperature)


def load_solar_instruct_llm(temperature: float = 0.0):
    return load_llm(model_path="/home/airat/LLMs/solar-10.7b-instruct-v1.0.Q4_K_M.gguf", temperature=temperature,
                    ctx_size=4096)


def load_solar_instruct(system_prompt: str = "", temperature: float = 0.0):
    return solar_instruct_template(system_prompt) | load_solar_instruct_llm(temperature=temperature)


def load_xdan_llm(temperature: float = 0.0):
    return load_llm(model_path="/home/airat/LLMs/xdan-l1-chat-rl-v1.Q4_K_M.gguf", temperature=temperature)


def load_xdan(system_prompt: str = "", temperature: float = 0.0):
    return alpaca_template(system_prompt) | load_xdan_llm(temperature)


def load_openchat_llm(temperature: float = 0.0):
    return load_llm(model_path="/home/airat/LLMs/openchat-3.5-0106.Q4_K_M.gguf", temperature=temperature, ctx_size=8192)


def load_openchat(system_prompt: str = "", temperature: float = 0.0):
    return openchat_template(system_prompt) | load_openchat_llm(temperature)


def load_dolphin_llm(temperature: float = 0.0):
    return load_llm(model_path="/home/airat/LLMs/dolphin-2.6-mistral-7b.Q4_K_M.gguf", temperature=temperature,
                    ctx_size=16384)


def load_dolphin(system_prompt: str = "", temperature: float = 0.0):
    return chat_ml_template(system_prompt) | load_dolphin_llm(temperature=temperature)


def load_dolphin_dpo_laser_llm(temperature: float = 0.0):
    return load_llm(model_path="/home/airat/LLMs/dolphin-2.6-mistral-7b-dpo-laser.Q4_K_M.gguf",
                    temperature=temperature, ctx_size=16384)


def load_dolphin_dpo_laser(system_prompt: str = "", temperature: float = 0.0):
    return chat_ml_template(system_prompt) | load_dolphin_dpo_laser_llm(temperature=temperature)


def load_yarn_mistral_llm(temperature: float = 0.0):
    return load_llm(model_path="/home/airat/LLMs/yarn-mistral-7b-128k.Q4_K_M.gguf", temperature=temperature,
                    ctx_size=131072)


def load_yarn_mistral(system_prompt: str = "", temperature: float = 0.0):
    return alpaca_template(system_prompt) | load_yarn_mistral_llm(temperature=temperature)


def load_mistrallite_llm(temperature: float = 0.0):
    return load_llm(model_path="/home/airat/LLMs/mistrallite.Q4_K_M.gguf", temperature=temperature, ctx_size=16384)


def load_mistrallite(system_prompt: str = "", temperature: float = 0.0):
    return amazon_template(system_prompt) | load_mistrallite_llm(temperature=temperature)


def load_neuralmarcoro_llm(temperature: float = 0.0):
    return load_llm(model_path="/home/airat/LLMs/neuralmarcoro14-7b.Q4_K_M.gguf", temperature=temperature)


def load_neuralmarcoro(system_prompt: str = "", temperature: float = 0.0):
    return alpaca_template(system_prompt) | load_neuralmarcoro_llm(temperature=temperature)
