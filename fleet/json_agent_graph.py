"""
JSON Agent Graph Executor — Declarative workflow engine for the sunset ecosystem.

Parses JSON agent graphs (DAGs) and executes them using the fleet's
subagent infrastructure. Supports parallel branches, dynamic routing,
and iterative refinement loops.

Inspired by the JSON agent graph concepts from ideaa.md, adapted to
our FleetConductorV2 + SenseDecideAct stack.

JSON Schema:
    {
      "graph": {
        "nodes": [
          {"id": "start", "type": "input", "next": "router"},
          {"id": "router", "type": "router", "prompt": "Classify...", "next": {"destination": "flight_agent", "activity": "hotel_agent"}},
          {"id": "flight_agent", "type": "agent", "tool": "search_flights", "next": "synthesis"},
          {"id": "synthesis", "type": "llm", "prompt": "Combine...", "next": "end"},
          {"id": "end", "type": "output"}
        ]
      }
    }

Usage:
    executor = JsonAgentGraphExecutor(graph_json)
    result = await executor.run({"user_input": "Book a flight to Paris"})
"""

import asyncio
import json
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable, Set


@dataclass
class GraphNode:
    id: str
    type: str  # input, output, llm, agent, tool, router, judge
    prompt: Optional[str] = None
    tool: Optional[str] = None
    next: Optional[Any] = None  # str (single) or Dict[str, str] (conditional)
    parallel: bool = False  # if True, spawn all branches in parallel
    max_iterations: int = 3  # for judge/refine loops
    threshold: float = 0.8  # for judge acceptance


@dataclass
class GraphResult:
    final_output: Any
    execution_trace: List[Dict[str, Any]] = field(default_factory=list)
    iterations: int = 0
    nodes_executed: Set[str] = field(default_factory=set)


class JsonAgentGraphExecutor:
    """Execute JSON-defined agent graphs."""

    def __init__(
        self,
        graph_json: Dict[str, Any],
        llm_callback: Optional[Callable[[str, Dict], Any]] = None,
        agent_callback: Optional[Callable[[str, Dict, Any], Any]] = None,
        tool_callback: Optional[Callable[[str, Dict], Any]] = None,
    ):
        self.graph = self._parse_graph(graph_json)
        self.llm_callback = llm_callback or self._default_llm
        self.agent_callback = agent_callback or self._default_agent
        self.tool_callback = tool_callback or self._default_tool

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------
    def _parse_graph(self, graph_json: Dict[str, Any]) -> Dict[str, GraphNode]:
        nodes = {}
        for node_data in graph_json.get("graph", {}).get("nodes", []):
            node = GraphNode(
                id=node_data["id"],
                type=node_data.get("type", "unknown"),
                prompt=node_data.get("prompt"),
                tool=node_data.get("tool"),
                next=node_data.get("next"),
                parallel=node_data.get("parallel", False),
                max_iterations=node_data.get("max_iterations", 3),
                threshold=node_data.get("threshold", 0.8),
            )
            nodes[node.id] = node
        return nodes

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------
    async def run(self, initial_input: Dict[str, Any]) -> GraphResult:
        result = GraphResult(final_output=None)
        state = {"input": initial_input, "context": {}, "iterations": 0}
        current_node_id = self._find_start_node()

        while current_node_id:
            node = self.graph.get(current_node_id)
            if not node:
                break

            result.nodes_executed.add(current_node_id)
            trace_entry = {"node": current_node_id, "type": node.type}

            # Execute node
            output = await self._execute_node(node, state)
            trace_entry["output"] = output
            result.execution_trace.append(trace_entry)

            # Determine next node
            next_id = self._resolve_next(node, output)

            # Handle iterative refinement (judge loop)
            if node.type == "judge":
                if (
                    isinstance(output, dict)
                    and output.get("score", 0) >= node.threshold
                ):
                    # Judge accepted — exit loop
                    state["context"]["judge_accepted"] = True
                else:
                    state["iterations"] = state.get("iterations", 0) + 1
                    if state["iterations"] < node.max_iterations:
                        # Route back to planner for refinement
                        next_id = (
                            output.get("refine_target", "planner")
                            if isinstance(output, dict)
                            else "planner"
                        )
                    else:
                        state["context"]["judge_maxed"] = True

            # Handle parallel execution
            if node.parallel and isinstance(next_id, dict):
                branch_ids = list(next_id.values())
                branch_results = await self._execute_parallel(branch_ids, state)
                state["context"]["parallel_results"] = branch_results
                # After parallel, route to the merge node (specified in node's "merge" field)
                next_id = (
                    node.next.get("__merge__") if isinstance(node.next, dict) else None
                )

            current_node_id = next_id
            result.iterations += 1

            # Safety break
            if result.iterations > 100:
                break

        result.final_output = state.get("context", {}).get(
            "final_output", state.get("input")
        )
        return result

    async def _execute_node(self, node: GraphNode, state: Dict[str, Any]) -> Any:
        if node.type == "input":
            return state["input"]

        elif node.type == "output":
            state["context"]["final_output"] = state["context"].get("last_output")
            return state["context"]["final_output"]

        elif node.type == "llm":
            prompt = node.prompt or ""
            if "{input}" in prompt:
                prompt = prompt.format(**state["input"])
            response = await self.llm_callback(prompt, state)
            state["context"]["last_output"] = response
            return response

        elif node.type == "agent":
            response = await self.agent_callback(
                node.tool or "default", state, state["input"]
            )
            state["context"]["last_output"] = response
            return response

        elif node.type == "tool":
            response = await self.tool_callback(node.tool or "default", state)
            state["context"]["last_output"] = response
            return response

        elif node.type == "router":
            prompt = node.prompt or ""
            if "{input}" in prompt:
                prompt = prompt.format(**state["input"])
            routing = await self.llm_callback(prompt, state)
            # Expect routing to be a dict or string key
            if isinstance(routing, dict) and "next" in routing:
                return routing
            return routing

        elif node.type == "judge":
            prompt = (
                node.prompt
                or "Evaluate the output and return a score 0-1 and feedback."
            )
            if "{input}" in prompt:
                prompt = prompt.format(**state)
            evaluation = await self.llm_callback(prompt, state)
            if isinstance(evaluation, dict):
                return evaluation
            # Default: assume score is embedded in text
            return {
                "score": 0.5,
                "feedback": str(evaluation),
                "refine_target": "planner",
            }

        elif node.type == "aggregator":
            parallel_results = state["context"].get("parallel_results", [])
            prompt = (
                node.prompt
                or "Synthesize the following outputs into a single response."
            )
            combined = "\n".join(str(r) for r in parallel_results)
            response = await self.llm_callback(f"{prompt}\n\n{combined}", state)
            state["context"]["last_output"] = response
            return response

        else:
            return {"error": f"Unknown node type: {node.type}"}

    async def _execute_parallel(
        self, branch_ids: List[str], state: Dict[str, Any]
    ) -> List[Any]:
        """Execute multiple branches in parallel."""

        async def run_branch(branch_id: str) -> Any:
            node = self.graph.get(branch_id)
            if not node:
                return {"error": f"Unknown branch: {branch_id}"}
            return await self._execute_node(node, state)

        tasks = [run_branch(bid) for bid in branch_ids]
        return await asyncio.gather(*tasks, return_exceptions=True)

    def _resolve_next(self, node: GraphNode, output: Any) -> Optional[str]:
        if node.next is None:
            return None
        if isinstance(node.next, str):
            return node.next
        if isinstance(node.next, dict):
            # Conditional routing based on output
            if isinstance(output, dict):
                for key, target in node.next.items():
                    if key in str(output):
                        return target
            # Fallback to first key
            return list(node.next.values())[0] if node.next else None
        return None

    def _find_start_node(self) -> str:
        for node_id, node in self.graph.items():
            if node.type == "input":
                return node_id
        # Fallback: return first node
        return list(self.graph.keys())[0] if self.graph else None

    # ------------------------------------------------------------------
    # Default callbacks (stubs — override in production)
    # ------------------------------------------------------------------
    async def _default_llm(self, prompt: str, state: Dict[str, Any]) -> Any:
        # In production, this calls the actual LLM
        return {"prompt": prompt, "mock_response": "LLM output"}

    async def _default_agent(
        self, tool: str, state: Dict[str, Any], input_data: Any
    ) -> Any:
        # In production, this spawns a subagent
        return {"tool": tool, "mock_response": "Agent output"}

    async def _default_tool(self, tool: str, state: Dict[str, Any]) -> Any:
        return {"tool": tool, "mock_response": "Tool output"}

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------
    @classmethod
    def from_json_string(cls, json_str: str, **callbacks) -> "JsonAgentGraphExecutor":
        return cls(json.loads(json_str), **callbacks)

    @classmethod
    def from_file(cls, path: str, **callbacks) -> "JsonAgentGraphExecutor":
        with open(path) as f:
            return cls(json.load(f), **callbacks)
