from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

from llms.llamacpp import LlamaCpp
from utils.templates import chat_ml_template, alpaca_template


def load_llm(model_path: str, temperature: float = 0.0):
    return LlamaCpp(
        model_path=model_path,
        n_gpu_layers=30,
        n_batch=448,
        n_ctx=8192,
        max_tokens=8192,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
        verbose=False,
        temperature=temperature,
        stop=["<|end_of_turn|>", "<|endoftext|>", "<|im_start|>", "<|im_end|>", "</s>", "Human:", "AI:",
              "Assistant:", "### System:", "### User:", "### Assistant:", "<|prompter|>", "<|assistant|>",
              "\nObservation"]
    )


def load_bge_emb(model_name: str, use_gpu: bool = True):
    device = 'cuda' if use_gpu else 'cpu'
    return HuggingFaceBgeEmbeddings(
        model_name=model_name,
        cache_folder="/home/airat/MTEB",
        model_kwargs={'device': device}
    )


def load_bge_base_angle_emb(use_gpu: bool = False):
    return load_bge_emb(model_name="khoa-klaytn/bge-base-en-v1.5-angle", use_gpu=use_gpu)


def load_openhermes_llm(temperature: float = 0.0):
    return load_llm(model_path="/home/airat/LLMs/openhermes-2.5-mistral-7b.Q4_K_M.gguf", temperature=temperature)


def load_openhermes(system_prompt: str = "", temperature: float = 0.0):
    return chat_ml_template(system_prompt) | load_openhermes_llm(temperature=temperature)


def load_openhermes_16k_llm(temperature: float = 0.0):
    return load_llm(model_path="/home/airat/LLMs/openhermes-2.5-mistral-7b-16k.Q4_K_M.gguf", temperature=temperature)


def load_openhermes_16k(system_prompt: str = "", temperature: float = 0.0):
    return chat_ml_template(system_prompt) | load_openhermes_16k_llm(temperature=temperature)


def load_xdan_llm(temperature: float = 0.0):
    return load_llm(model_path="/home/airat/LLMs/xdan-l1-chat-rl-v1.Q4_K_M.gguf", temperature=temperature)


def load_xdan(system_prompt: str = "", temperature: float = 0.0):
    return alpaca_template(system_prompt) | load_xdan_llm(temperature)


def load_dolphin_llm(temperature: float = 0.0):
    return load_llm(model_path="/home/airat/LLMs/dolphin-2.6-mistral-7b.Q4_K_M.gguf", temperature=temperature)


def load_dolphin(system_prompt: str = "", temperature: float = 0.0):
    return chat_ml_template(system_prompt) | load_dolphin_llm(temperature=temperature)


def load_dolphin_dpo_laser_llm(temperature: float = 0.0):
    return load_llm(model_path="/home/airat/LLMs/dolphin-2.6-mistral-7b-dpo-laser.Q4_K_M.gguf", temperature=temperature)


def load_dolphin_dpo_laser(system_prompt: str = "", temperature: float = 0.0):
    return chat_ml_template(system_prompt) | load_dolphin_dpo_laser_llm(temperature=temperature)


def load_neuralmarcoro_llm(temperature: float = 0.0):
    return load_llm(model_path="/home/airat/LLMs/neuralmarcoro14-7b.Q4_K_M.gguf", temperature=temperature)


def load_neuralmarcoro(system_prompt: str = "", temperature: float = 0.0):
    return alpaca_template(system_prompt) | load_neuralmarcoro_llm(temperature=temperature)


def load_westlake_llm(temperature: float = 0.0):
    return load_llm(model_path="/home/airat/LLMs/westlake-7b-v2.Q4_K_M.gguf", temperature=temperature)


def load_westlake(system_prompt: str = "", temperature: float = 0.0):
    return chat_ml_template(system_prompt) | load_westlake_llm(temperature=temperature)


def load_szcf_llm(temperature: float = 0.0):
    return load_llm(
        model_path="/home/airat/LLMs/LoneStriker/speechless-zephyr-code-functionary-7b-GGUF/speechless-zephyr-code-functionary-7b-Q4_K_M.gguf",
        temperature=temperature)


def load_szcf(system_prompt: str = "", temperature: float = 0.0):
    return chat_ml_template(system_prompt) | load_szcf_llm(temperature=temperature)
