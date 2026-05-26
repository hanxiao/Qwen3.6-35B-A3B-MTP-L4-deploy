#!/usr/bin/env python3
"""
Chat client for llama-server with built-in tools support.
Automatically fetches server tools and runs the agentic loop.

Usage:
    python chat_with_tools.py "What is the latest news about Jina AI?"
    python chat_with_tools.py --base-url http://35.252.252.236:8080 "your question"
"""

import argparse
import json
import sys
import requests

DEFAULT_BASE_URL = "http://35.252.252.236:8080"
MAX_TOOL_ROUNDS = 10


def fetch_builtin_tools(base_url: str) -> list[dict]:
    """GET /tools to fetch server built-in tool definitions."""
    resp = requests.get(f"{base_url}/tools", timeout=10)
    resp.raise_for_status()
    tools_raw = resp.json()
    # Each entry has a nested "definition" with the OpenAI format
    tools = [t["definition"] for t in tools_raw]
    return tools


def execute_tool(base_url: str, tool_name: str, params: dict) -> str:
    """POST /tools to execute a built-in tool on the server."""
    resp = requests.post(
        f"{base_url}/tools",
        json={"tool": tool_name, "params": params},
        timeout=60,
    )
    resp.raise_for_status()
    result = resp.json()
    if "error" in result:
        return f"Error: {result['error']}"
    if "plaintext" in result:
        return result["plaintext"]
    return json.dumps(result)


def chat(base_url: str, messages: list[dict], tools: list[dict], max_tokens: int = 4096) -> dict:
    """Send a chat completion request with tools."""
    payload = {
        "model": "qwen3.6",
        "messages": messages,
        "tools": tools,
        "max_tokens": max_tokens,
    }
    resp = requests.post(
        f"{base_url}/v1/chat/completions",
        json=payload,
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()


def run(base_url: str, user_message: str, max_tokens: int = 4096, verbose: bool = False):
    # 1. Fetch built-in tools
    tools = fetch_builtin_tools(base_url)
    tool_names = [t["function"]["name"] for t in tools]
    print(f"[info] Built-in tools: {tool_names}")

    # 2. Start conversation
    messages = [{"role": "user", "content": user_message}]

    for round_i in range(MAX_TOOL_ROUNDS):
        if verbose:
            print(f"\n[round {round_i + 1}] Sending {len(messages)} messages...")

        result = chat(base_url, messages, tools, max_tokens)
        choice = result["choices"][0]
        msg = choice["message"]
        finish_reason = choice.get("finish_reason", "")

        # Show thinking if present
        if msg.get("reasoning_content"):
            thinking = msg["reasoning_content"]
            print(f"\n[thinking] ({len(thinking)} chars)")
            if verbose:
                print(thinking[:500])

        # Check for tool calls
        tool_calls = msg.get("tool_calls")
        if not tool_calls or finish_reason == "stop":
            # No tool calls - final answer
            content = msg.get("content", "")
            print(f"\n[answer]\n{content}")
            usage = result.get("usage", {})
            print(f"\n[usage] prompt={usage.get('prompt_tokens')}, completion={usage.get('completion_tokens')}, total={usage.get('total_tokens')}")
            return

        # Process tool calls
        # Append assistant message with tool_calls
        messages.append(msg)

        for tc in tool_calls:
            func = tc.get("function", tc)
            tool_name = func["name"]
            try:
                tool_args = json.loads(func["arguments"]) if isinstance(func["arguments"], str) else func["arguments"]
            except (json.JSONDecodeError, TypeError):
                tool_args = {}

            print(f"\n[tool_call] {tool_name}({json.dumps(tool_args, ensure_ascii=False)[:200]})")

            # Execute on server
            tool_result = execute_tool(base_url, tool_name, tool_args)
            print(f"[tool_result] {tool_result[:200]}...")

            # Append tool result
            messages.append({
                "role": "tool",
                "tool_call_id": tc.get("id", ""),
                "content": tool_result,
            })

    print("\n[warn] Max tool rounds reached, stopping.")


def main():
    parser = argparse.ArgumentParser(description="Chat with llama-server built-in tools")
    parser.add_argument("message", help="User message")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help=f"Server URL (default: {DEFAULT_BASE_URL})")
    parser.add_argument("--max-tokens", type=int, default=4096, help="Max tokens (default: 4096)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    run(args.base_url, args.message, args.max_tokens, args.verbose)


if __name__ == "__main__":
    main()
