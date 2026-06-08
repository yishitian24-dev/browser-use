"""
Batch task runner — run multiple browser tasks from a YAML file.

Usage:
    python examples/batch_runner.py tasks.yaml
    python examples/batch_runner.py tasks.yaml --model qwen2.5:7b --output reports/

tasks.yaml format:
    - task: "搜索最新 AI 新闻，返回标题"
      id: news
    - task: "打开 github.com/trending 截图"
      id: github_trending
"""

import argparse
import asyncio
import json
import os
from datetime import datetime
from pathlib import Path

import yaml

from browser_use import Agent
from browser_use.llm.anthropic.chat import ChatAnthropic
from browser_use.llm.ollama.chat import ChatOllama
from ollama import Options


def get_llm(model: str):
    if model.startswith("claude"):
        return ChatAnthropic(model=model, api_key=os.environ["ANTHROPIC_API_KEY"])
    return ChatOllama(
        model=model,
        host="http://localhost:11434",
        timeout=120.0,
        ollama_options=Options(num_ctx=4096, temperature=0, num_gpu=99),
    )


async def run_task(task_def: dict, llm) -> dict:
    task_id = task_def.get("id", "task")
    task_text = task_def["task"]
    started = datetime.now().isoformat()
    try:
        agent = Agent(task=task_text, llm=llm)
        result = await agent.run()
        return {"id": task_id, "task": task_text, "status": "ok", "result": str(result), "started": started}
    except Exception as e:
        return {"id": task_id, "task": task_text, "status": "error", "error": str(e), "started": started}


async def main(tasks_file: str, model: str, output_dir: str):
    tasks_path = Path(tasks_file)
    with open(tasks_path) as f:
        tasks = yaml.safe_load(f)

    print(f"Loaded {len(tasks)} tasks from {tasks_file}")
    llm = get_llm(model)
    results = []

    for i, task_def in enumerate(tasks, 1):
        print(f"[{i}/{len(tasks)}] Running: {task_def.get('id', 'task')} ...")
        result = await run_task(task_def, llm)
        results.append(result)
        status = "✓" if result["status"] == "ok" else "✗"
        print(f"  {status} {result['status']}")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    report_file = out_path / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    ok = sum(1 for r in results if r["status"] == "ok")
    print(f"\nDone: {ok}/{len(results)} succeeded → {report_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("tasks_file")
    parser.add_argument("--model", default="qwen2.5:7b")
    parser.add_argument("--output", default="reports")
    args = parser.parse_args()
    asyncio.run(main(args.tasks_file, args.model, args.output))
