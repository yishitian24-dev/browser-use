"""
RTX 3060 Laptop (6GB VRAM) optimized browser-use with local Ollama.

Tuned for qwen2.5:7b and gemma4 running on 6GB VRAM:
- num_ctx=4096 keeps VRAM under 5GB for qwen2.5:7b
- temperature=0 improves instruction-following for browser tasks
- num_predict caps runaway generation
"""

import asyncio
from browser_use import Agent
from browser_use.llm.ollama.chat import ChatOllama
from ollama import Options

RTX3060_OPTIONS = Options(
    num_ctx=4096,       # 4K context fits comfortably in 6GB with qwen2.5:7b
    temperature=0,      # deterministic — better for structured browser actions
    num_predict=1024,   # cap output length per step
    num_gpu=99,         # offload all layers to GPU
)

def get_llm(model: str = "qwen2.5:7b") -> ChatOllama:
    """
    Returns a ChatOllama instance tuned for RTX 3060 6GB.
    Supported models: qwen2.5:7b, gemma4
    """
    return ChatOllama(
        model=model,
        host="http://localhost:11434",
        timeout=120.0,
        ollama_options=RTX3060_OPTIONS,
    )


async def run(task: str, model: str = "qwen2.5:7b") -> str:
    llm = get_llm(model)
    agent = Agent(task=task, llm=llm)
    result = await agent.run()
    return str(result)


if __name__ == "__main__":
    import sys
    task = sys.argv[1] if len(sys.argv) > 1 else "搜索今天的科技新闻头条，返回前3条标题"
    model = sys.argv[2] if len(sys.argv) > 2 else "qwen2.5:7b"
    print(asyncio.run(run(task, model)))
