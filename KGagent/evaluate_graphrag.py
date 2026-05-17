"""Evaluate GraphRAG answers and write annotated results.

Usage:
    python evaluate_graphrag.py
    python evaluate_graphrag.py --input test1.json --output eval_result_test1.json

Default behavior:
    - Prefer `test1.json` if it exists, otherwise fall back to `test.json`
    - Write to `eval_result.json` for `test.json`, or `eval_result_<stem>.json`
      for other inputs

The script:
    1. Loads test cases
    2. Calls the current GraphRAG answerer
    3. Uses Qwen as a judge to score correctness
    4. Writes a JSON report with per-case answers and overall accuracy
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from openai import OpenAI

from src.config import Settings
from src.graph_store import Neo4jGraphStore
from src.graphrag import GraphRAGService


DEFAULT_INPUT_CANDIDATES = (Path("test1.json"), Path("test.json"))
DEFAULT_OUTPUT = Path("eval_result.json")


@dataclass
class JudgeResult:
    """Structured verdict from the judge model."""

    is_correct: bool
    score: int
    reason: str


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Evaluate GraphRAG on a test JSON file.")
    parser.add_argument("--input", type=str, default=None, help="Path to the test JSON file.")
    parser.add_argument("--output", type=str, default=None, help="Path to the evaluation result JSON file.")
    parser.add_argument("--limit", type=int, default=None, help="Only evaluate the first N samples.")
    parser.add_argument("--judge-model", type=str, default=None, help="Qwen model name used for judging.")
    return parser.parse_args()


def resolve_input_path(cli_value: str | None) -> Path:
    """Resolve the input test file.

    If no explicit input is provided, prefer test1.json so the new harder
    samples are used by default.
    """
    if cli_value:
        return Path(cli_value)

    for candidate in DEFAULT_INPUT_CANDIDATES:
        if candidate.exists():
            return candidate

    raise FileNotFoundError("No test file found. Expected test1.json or test.json.")


def resolve_output_path(input_path: Path, cli_value: str | None) -> Path:
    """Resolve the output report path."""
    if cli_value:
        return Path(cli_value)
    if input_path.stem == "test":
        return DEFAULT_OUTPUT
    return Path(f"eval_result_{input_path.stem}.json")


def load_cases(path: Path) -> list[dict[str, Any]]:
    """Load test cases from JSON."""
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Test file must contain a JSON list of cases.")
    return data


def build_judge_client(settings: Settings) -> OpenAI:
    """Create an OpenAI-compatible client for evaluation."""
    return OpenAI(
        api_key=settings.dashscope_api_key,
        base_url=settings.dashscope_base_url,
    )


def safe_json_loads(text: str) -> dict[str, Any]:
    """Best-effort JSON parser for model outputs."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.S)
        if match:
            return json.loads(match.group(0))
    raise ValueError(f"Judge output is not valid JSON: {text}")


def judge_answer(
    client: OpenAI,
    model_name: str,
    question: str,
    gold_answer: str,
    model_answer: str,
    evidence_path: list[dict[str, Any]],
) -> JudgeResult:
    """Ask Qwen to judge whether the model answer matches the gold answer."""
    prompt = f"""你是一名严格的知识图谱问答评估员。请判断“模型回答”是否正确。

判定标准：
1. 模型回答必须与标准答案表达同一条图谱路径或同一组事实。
2. 允许措辞不同，但实体、关系和方向不能错。
3. 如果缺少中间实体、关系方向错误、关系链不一致，判为错误。
4. 只输出 JSON，不要输出多余文本。

问题：
{question}

标准答案：
{gold_answer}

模型回答：
{model_answer}

证据路径：
{json.dumps(evidence_path, ensure_ascii=False)}

请按以下格式输出：
{{"is_correct": true/false, "score": 0-100, "reason": "简短原因"}}
"""

    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "你是一个严格、稳定的知识图谱问答评估器，只输出 JSON。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        content = completion.choices[0].message.content or ""
        payload = safe_json_loads(content)
        return JudgeResult(
            is_correct=bool(payload.get("is_correct", False)),
            score=int(payload.get("score", 0)),
            reason=str(payload.get("reason", "")).strip(),
        )
    except Exception as exc:  # noqa: BLE001
        return JudgeResult(False, 0, f"Judge failed: {exc}")


def write_report(output_path: Path, report: dict[str, Any]) -> None:
    """Write the current report to disk."""
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def evaluate(input_path: Path, output_path: Path, judge_model: str | None = None, limit: int | None = None) -> dict[str, Any]:
    """Run end-to-end evaluation and return the full report."""
    settings = Settings.from_env()
    graph = Neo4jGraphStore(settings)
    rag = GraphRAGService(graph, settings)
    judge_client = build_judge_client(settings)
    effective_judge_model = judge_model or settings.dashscope_model

    cases = load_cases(input_path)
    if limit is not None:
        cases = cases[: max(limit, 0)]

    results: list[dict[str, Any]] = []
    correct_count = 0

    try:
        for idx, case in enumerate(cases, start=1):
            question = str(case.get("question", "")).strip()
            gold_answer = str(case.get("answer", "")).strip()
            evidence_path = case.get("evidence_path", []) or []

            case_error = ""
            try:
                model_answer, graph_context = rag.answer(question)
            except Exception as exc:  # noqa: BLE001
                model_answer = f"[GraphRAG Error] {exc}"
                graph_context = ""
                case_error = str(exc)

            if case_error:
                verdict = JudgeResult(False, 0, f"GraphRAG failed: {case_error}")
            else:
                verdict = judge_answer(
                    client=judge_client,
                    model_name=effective_judge_model,
                    question=question,
                    gold_answer=gold_answer,
                    model_answer=model_answer,
                    evidence_path=evidence_path,
                )

            if verdict.is_correct:
                correct_count += 1

            results.append(
                {
                    **case,
                    "model_answer": model_answer,
                    "graph_context": graph_context,
                    "judge_result": asdict(verdict),
                    "is_correct": verdict.is_correct,
                    "score": verdict.score,
                    "reason": verdict.reason,
                }
            )

            report = {
                "metadata": {
                    "input_path": str(input_path),
                    "output_path": str(output_path),
                    "answer_model": settings.dashscope_model,
                    "judge_model": effective_judge_model,
                },
                "summary": {
                    "total": len(results),
                    "correct": correct_count,
                    "wrong": len(results) - correct_count,
                    "accuracy": round(correct_count / len(results), 4) if results else 0.0,
                },
                "results": results,
            }
            write_report(output_path, report)

            print(f"[{idx}/{len(cases)}] {case.get('id', idx)} -> {'correct' if verdict.is_correct else 'wrong'}")
    finally:
        graph.close()

    total = len(results)
    accuracy = round(correct_count / total, 4) if total else 0.0
    return {
        "metadata": {
            "input_path": str(input_path),
            "output_path": str(output_path),
            "answer_model": settings.dashscope_model,
            "judge_model": effective_judge_model,
        },
        "summary": {
            "total": total,
            "correct": correct_count,
            "wrong": total - correct_count,
            "accuracy": accuracy,
        },
        "results": results,
    }


def main() -> None:
    args = parse_args()
    input_path = resolve_input_path(args.input)
    output_path = resolve_output_path(input_path, args.output)
    report = evaluate(
        input_path=input_path,
        output_path=output_path,
        judge_model=args.judge_model,
        limit=args.limit,
    )
    write_report(output_path, report)
    print(f"Saved evaluation report to {output_path.resolve()}")
    print(
        "Accuracy:",
        f"{report['summary']['accuracy'] * 100:.2f}%",
        f"({report['summary']['correct']}/{report['summary']['total']})",
    )


if __name__ == "__main__":
    main()
