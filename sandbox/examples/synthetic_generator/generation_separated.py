# /// script
# dependencies = [
#   "openai>=1.6.0",
#   "datasets>=3.6.0",
#   "tenacity>=8.0.0",
#   "httpx>=0.27.0",
#   "rich>=13.9.5",
# ]
# requires-python = ">=3.10"
# ///

"""
Optimized Model Evaluation Pipeline using SoS (Sea of Simulation) - Separated Phases

This script separates inference and execution phases for better performance:

Phase 1 - Batch Inference:
1. Load dataset and model
2. Generate all commands upfront 
3. Save commands to intermediate file

Phase 2 - Batch Execution:
1. Load pre-generated commands
2. Execute each through sandbox: setup → execute → test
3. Save results to final output

This separation allows:
- Faster iteration on inference without sandbox overhead
- Better resource utilization and optimization opportunities
- Easy retry of failed executions without re-inference
- Independent scaling of inference vs execution
"""

import os
import json
import uuid
import asyncio
import httpx
import argparse
import hashlib
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict
from rich import print
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn

from openai import AsyncOpenAI
from datasets import load_dataset, Dataset
from tenacity import retry, stop_after_attempt
from sos import SoSClient

import re


def short_id() -> str:
    """uuid4 first 8 hex chars."""
    return uuid.uuid4().hex[:8]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def hash_setup_commands(setup_commands: str) -> str:
    """Create a hash for setup commands to group compatible tasks."""
    # Normalize the setup commands string
    normalized = setup_commands.strip().lower()
    return hashlib.md5(normalized.encode()).hexdigest()[:8]


class SandboxPool:
    """Pool of sandboxes for efficient reuse and batching."""
    
    def __init__(self, sos: SoSClient, sandbox_image: str, pool_size: int = 4):
        self.sos = sos
        self.sandbox_image = sandbox_image
        self.pool_size = pool_size
        
        # Pool management
        self.available_sandboxes: Dict[str, List[str]] = defaultdict(list)  # setup_hash -> [sandbox_ids]
        self.busy_sandboxes: Dict[str, str] = {}  # sandbox_id -> setup_hash
        self.setup_commands_cache: Dict[str, str] = {}  # setup_hash -> setup_commands
        
        # Concurrency control
        self.semaphore = asyncio.Semaphore(pool_size)
        
    async def get_sandbox(self, setup_commands: str) -> Tuple[str, str]:
        """Get a sandbox for the given setup commands. Returns (sandbox_id, setup_hash)."""
        setup_hash = hash_setup_commands(setup_commands)
        
        async with self.semaphore:
            # Try to reuse existing sandbox with same setup
            if setup_hash in self.available_sandboxes and self.available_sandboxes[setup_hash]:
                sandbox_id = self.available_sandboxes[setup_hash].pop()
                self.busy_sandboxes[sandbox_id] = setup_hash
                # Reset sandbox state
                await self._reset_sandbox(sandbox_id)
                return sandbox_id, setup_hash
            
            # Create new sandbox
            sandbox_id = await self.sos.create_sandbox(
                image=self.sandbox_image,
                setup_commands=[setup_commands] if setup_commands else []
            )
            await self.sos.start_sandbox(sandbox_id)
            
            # Track it
            self.busy_sandboxes[sandbox_id] = setup_hash
            self.setup_commands_cache[setup_hash] = setup_commands
            
            return sandbox_id, setup_hash
    
    async def return_sandbox(self, sandbox_id: str):
        """Return a sandbox to the pool for reuse."""
        if sandbox_id in self.busy_sandboxes:
            setup_hash = self.busy_sandboxes.pop(sandbox_id)
            self.available_sandboxes[setup_hash].append(sandbox_id)
    
    async def _reset_sandbox(self, sandbox_id: str):
        """Reset sandbox state between tasks."""
        # Clean up common temporary files and reset to home directory
        reset_commands = [
            "cd ~",
            "rm -rf /tmp/* 2>/dev/null || true",
            "rm -rf ~/tmp_* 2>/dev/null || true", 
            "unset $(env | grep '^TEMP_' | cut -d= -f1) 2>/dev/null || true"
        ]
        
        for cmd in reset_commands:
            try:
                await self.sos.exec_command(sandbox_id, cmd)
            except Exception:
                # Ignore reset errors
                pass
    
    async def cleanup_all(self):
        """Clean up all sandboxes in the pool."""
        all_sandbox_ids = []
        
        # Collect all sandbox IDs
        for sandbox_id in self.busy_sandboxes.keys():
            all_sandbox_ids.append(sandbox_id)
            
        for sandbox_list in self.available_sandboxes.values():
            all_sandbox_ids.extend(sandbox_list)
        
        # Stop and remove all sandboxes
        cleanup_tasks = []
        for sandbox_id in all_sandbox_ids:
            cleanup_tasks.append(self.sos.stop_sandbox(sandbox_id, remove=True))
        
        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)
        
        # Clear tracking
        self.available_sandboxes.clear()
        self.busy_sandboxes.clear()
        self.setup_commands_cache.clear()


class ModelInterface:
    """Interface for OpenAI-compatible model inference."""
    
    def __init__(self, model_name: str, base_url: str = None, api_key: str = None, mock_mode: bool = False):
        self.model_name = model_name
        self.mock_mode = mock_mode
        
        if mock_mode:
            print(f"Initializing MOCK model: {model_name}")
            self.client = None
            return
        
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            raise ValueError("API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        print(f"Initializing OpenAI client for model: {model_name}")
        print(f"Base URL: {self.base_url}")
        
        self.client = AsyncOpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            http_client=httpx.AsyncClient(timeout=60.0),
        )
    
    async def generate_command(self, task: str, temperature: float = 0.6, max_tokens: int = 32_000) -> str:
        """Generate shell command for a given task."""
        if self.mock_mode:
            return self._mock_generate_command(task)
        
        return await self._real_generate_command(task, temperature, max_tokens)
    
    def _mock_generate_command(self, task: str) -> str:
        """Generate mock shell commands based on simple keyword matching."""
        task_lower = task.lower()
        
        # Simple rule-based mock responses
        if "create" in task_lower and "file" in task_lower:
            return "touch example.txt"
        elif "create" in task_lower and "directory" in task_lower:
            return "mkdir test_dir"
        elif "list" in task_lower and ("file" in task_lower or "directory" in task_lower):
            return "ls -la"
        elif "copy" in task_lower or "cp" in task_lower:
            return "cp source.txt dest.txt"
        elif "move" in task_lower or "mv" in task_lower:
            return "mv oldname.txt newname.txt"
        elif "delete" in task_lower or "remove" in task_lower:
            return "rm -f unwanted.txt"
        elif "find" in task_lower:
            return "find . -name '*.txt'"
        elif "search" in task_lower or "grep" in task_lower:
            return "grep -r 'pattern' ."
        elif "permission" in task_lower or "chmod" in task_lower:
            return "chmod 755 script.sh"
        elif "count" in task_lower and "line" in task_lower:
            return "wc -l file.txt"
        elif "echo" in task_lower or "print" in task_lower:
            return "echo 'Hello World'"
        else:
            # Default fallback
            return "echo 'Task completed'"
    
    @retry(stop=stop_after_attempt(3))
    async def _real_generate_command(self, task: str, temperature: float, max_tokens: int) -> str:
        """Generate shell command using real API."""
        messages = [
            {
            "role": "user", 
            "content": f'{task}. Wrap the shell command in {{"command":"[your_command_here]"}}'
            }
        ]
        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=0.95,
                extra_body={
                    "top_k": 20,
                    "min_p": 0,
                }, 
            )
            
            generated_text = response.choices[0].message.content or ""
            print(f"Raw model output: {repr(generated_text)}")
            
            # Clean up the response - remove thinking tokens and extract command
            lines = generated_text.strip().split('\n')
            command = ""
            
            # Try to parse JSON response first
            try:
                # Look for JSON-like pattern
                json_match = re.search(r'\{"command"\s*:\s*"([^"]+)"\}', generated_text)
                if json_match:
                    command = json_match.group(1)
                else:
                    # Try to parse as actual JSON
                    parsed_json = json.loads(generated_text.strip())
                    if isinstance(parsed_json, dict) and "command" in parsed_json:
                        command = parsed_json["command"]
                    else:
                        raise ValueError("No command field found")
            except (json.JSONDecodeError, ValueError):
                # Fall back to line-by-line parsing
                # Look for content after </think> token
                think_end = generated_text.find('</think>')
                if think_end != -1:
                    # Extract everything after </think>
                    content_after_think = generated_text[think_end + len('</think>'):].strip()
                    command = content_after_think
                else:
                    # If no </think> found, use the original text as-is
                    command = generated_text.strip()
            # Remove common prefixes if present
            prefixes_to_remove = ["$ ", "# ", "bash: ", "shell: ", "`", "```bash", "```"]
            for prefix in prefixes_to_remove:
                if command.startswith(prefix):
                    command = command[len(prefix):]
                    break
            
            # Remove trailing backticks
            command = command.rstrip('`').strip()
            print(f"Result Output: {repr(command)}")
            
            return command
            
        except Exception as e:
            print(f"Error generating command: {e}")
            raise


async def batch_generate_commands(
    model: ModelInterface,
    dataset: Dataset,
    output_file: Path,
    temperature: float = 0.6,
    max_tokens: int = 2000,
) -> None:
    """
    Phase 1: Generate all commands upfront and save to file.
    
    Args:
        model: Model interface for generating commands
        dataset: Dataset with tasks to generate commands for
        output_file: Path to save generated commands (JSONL format)
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
    """
    print(f"Phase 1: Generating commands for {len(dataset)} tasks...")
    
    # Clear output file if it exists
    if output_file.exists():
        output_file.unlink()
    
    # Enhanced progress bar with elapsed time and ETA
    progress_columns = [
        SpinnerColumn(),
        TextColumn("[bold blue]Phase 1: Generating commands"),
        BarColumn(),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ]
    
    with Progress(*progress_columns) as progress:
        task_progress = progress.add_task("", total=len(dataset))
        
        for i, sample in enumerate(dataset):
            try:
                # Generate command
                generated_command = await model.generate_command(
                    sample['task'], 
                    temperature=temperature, 
                    max_tokens=max_tokens
                )
                
                # Create command record
                command_record = {
                    "task_id": short_id(),
                    "task": sample['task'],
                    "setup_commands": sample['setup_commands'],
                    "success_condition": sample['success_condition'],
                    "generated_command": generated_command,
                    "timestamp": utc_now_iso(),
                }
                
                # Save immediately
                append_jsonl(output_file, command_record)
                
                progress.update(task_progress, advance=1)
                
            except Exception as e:
                print(f"Error generating command for task {i}: {e}")
                # Save error record
                error_record = {
                    "task_id": short_id(),
                    "task": sample['task'],
                    "setup_commands": sample['setup_commands'], 
                    "success_condition": sample['success_condition'],
                    "generated_command": None,
                    "error": str(e),
                    "timestamp": utc_now_iso(),
                }
                append_jsonl(output_file, error_record)
    
    print(f"Phase 1 complete: Commands saved to {output_file}")


async def execute_single_command_with_pool(
    command_data: Dict[str, Any],
    sandbox_pool: SandboxPool,
) -> Dict[str, Any]:
    """
    Execute a single pre-generated command using a sandbox pool.
    
    Args:
        command_data: Dict with command info from batch_generate_commands
        sandbox_pool: Sandbox pool for efficient reuse
    
    Returns:
        Dict with execution results
    """
    start_time = utc_now_iso()
    
    try:
        # Skip if command generation failed
        if command_data.get("generated_command") is None:
            return {
                **command_data,
                "command_success": False,
                "test_passed": False,
                "overall_success": False,
                "error": command_data.get("error", "Command generation failed"),
                "start_time": start_time,
                "end_time": utc_now_iso(),
            }
        
        # Prepare setup commands
        setup_commands = command_data['setup_commands']
        if isinstance(setup_commands, list):
            setup_commands = "; ".join(setup_commands)
        
        # Get sandbox from pool
        sandbox_id, setup_hash = await sandbox_pool.get_sandbox(setup_commands)
        
        try:
            # Execute the generated command
            generated_command = command_data['generated_command']
            output, exit_code, exited = await sandbox_pool.sos.exec_command(sandbox_id, generated_command)
            command_success = exit_code == 0
            
            # Test success condition
            success_condition = command_data['success_condition']
            _, test_exit_code, _ = await sandbox_pool.sos.exec_command(sandbox_id, success_condition, standalone=True)
            test_passed = test_exit_code == 0
            
            # Get full trajectory for debugging
            trajectory = await sandbox_pool.sos.get_sandbox_trajectory(sandbox_id, formatted=False)
            
            overall_success = command_success and test_passed
            
            result = {
                **command_data,
                "command_output": output,
                "command_exit_code": exit_code,
                "command_success": command_success,
                "test_exit_code": test_exit_code,
                "test_passed": test_passed,
                "overall_success": overall_success,
                "trajectory": trajectory.get('trajectory', []),
                "setup_hash": setup_hash,
                "start_time": start_time,
                "end_time": utc_now_iso(),
                "exited": exited,
            }
            
            return result
            
        finally:
            # Return sandbox to pool for reuse
            await sandbox_pool.return_sandbox(sandbox_id)
            
    except Exception as e:
        return {
            **command_data,
            "command_output": "",
            "command_exit_code": -1,
            "command_success": False,
            "test_exit_code": -1,
            "test_passed": False,
            "overall_success": False,
            "error": str(e),
            "start_time": start_time,
            "end_time": utc_now_iso(),
        }


async def batch_execute_commands(
    commands_file: Path,
    sos: SoSClient,
    output_file: Path,
    sandbox_image: str = "deathbyknowledge/shellm-sandbox:latest",
    concurrency: int = 4,
    pool_size: int = 8,
) -> None:
    """
    Phase 2: Execute pre-generated commands through sandboxes with concurrency and pooling.
    
    Args:
        commands_file: Path to file with pre-generated commands (JSONL)
        sos: SoS client for sandbox management
        output_file: Path to save execution results (JSONL)
        sandbox_image: Docker image for sandboxes
        concurrency: Number of concurrent executions
        pool_size: Maximum number of sandboxes in pool
    """
    # Load pre-generated commands
    commands_data = load_jsonl(commands_file)
    if not commands_data:
        print(f"No commands found in {commands_file}")
        return
    
    print(f"Phase 2: Executing {len(commands_data)} commands with {concurrency} concurrent workers")
    print(f"Sandbox pool size: {pool_size}")
    
    # Group commands by setup to optimize sandbox reuse
    setup_groups = defaultdict(list)
    for cmd_data in commands_data:
        setup_commands = cmd_data['setup_commands']
        if isinstance(setup_commands, list):
            setup_commands = "; ".join(setup_commands)
        setup_hash = hash_setup_commands(setup_commands)
        setup_groups[setup_hash].append(cmd_data)
    
    print(f"Grouped into {len(setup_groups)} setup configurations:")
    for setup_hash, commands in setup_groups.items():
        print(f"  {setup_hash}: {len(commands)} commands")
    
    # Clear output file if it exists
    if output_file.exists():
        output_file.unlink()
    
    # Initialize sandbox pool
    sandbox_pool = SandboxPool(sos, sandbox_image, pool_size)
    
    success_count = 0
    completed_count = 0
    eval_start_time = datetime.now(timezone.utc)
    
    # Concurrency semaphore
    execution_semaphore = asyncio.Semaphore(concurrency)
    
    async def execute_with_semaphore(command_data: Dict[str, Any]) -> Dict[str, Any]:
        async with execution_semaphore:
            return await execute_single_command_with_pool(command_data, sandbox_pool)
    
    # Enhanced progress bar with success rate tracking
    progress_columns = [
        SpinnerColumn(),
        TextColumn("[bold green]Phase 2: Executing commands"),
        BarColumn(),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TextColumn("[bold cyan]{task.fields[success_rate]:.1f}% success"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ]
    
    try:
        with Progress(*progress_columns) as progress:
            task_progress = progress.add_task("", total=len(commands_data), success_rate=0.0)
            
            # Process commands in batches to avoid overwhelming the system
            batch_size = max(concurrency * 2, 10)
            
            for i in range(0, len(commands_data), batch_size):
                batch = commands_data[i:i + batch_size]
                
                # Execute batch concurrently
                batch_tasks = [execute_with_semaphore(cmd_data) for cmd_data in batch]
                results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                
                # Process results
                for result in results:
                    if isinstance(result, Exception):
                        # Create error result
                        result = {
                            "overall_success": False,
                            "error": str(result),
                            "start_time": utc_now_iso(),
                            "end_time": utc_now_iso(),
                        }
                    
                    if result.get('overall_success', False):
                        success_count += 1
                    
                    completed_count += 1
                    
                    # Save result immediately
                    append_jsonl(output_file, result)
                    
                    # Update progress with current success rate
                    current_success_rate = (success_count / completed_count * 100) if completed_count > 0 else 0
                    progress.update(
                        task_progress, 
                        advance=1,
                        success_rate=current_success_rate
                    )
    
    finally:
        # Clean up sandbox pool
        print("Cleaning up sandbox pool...")
        await sandbox_pool.cleanup_all()
    
    total_samples = len(commands_data)
    success_rate = success_count / total_samples if total_samples > 0 else 0
    print(f"Phase 2 complete: {success_count}/{total_samples} successful ({success_rate:.1%})")
    print(f"Results saved to {output_file}")


async def run_inference_phase(
    model_name: str,
    dataset_path: str,
    output_dir: str,
    max_samples: Optional[int] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: float = 0.6,
    max_tokens: int = 2000,
) -> Path:
    """Run only the inference phase."""
    # Initialize model
    mock_mode = model_name.lower() == "mock" or base_url is None
    model = ModelInterface(model_name, base_url=base_url, api_key=api_key, mock_mode=mock_mode)
    
    # Load dataset
    if os.path.exists(dataset_path):
        if dataset_path.endswith('.jsonl'):
            dataset_data = load_jsonl(Path(dataset_path))
            dataset = Dataset.from_list(dataset_data)
        else:
            dataset = load_dataset(dataset_path, split="test")
    else:
        dataset = load_dataset(dataset_path, split="test")
    
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    # Setup output
    output_path = Path(output_dir)
    commands_file = output_path / "commands.jsonl"
    
    await batch_generate_commands(model, dataset, commands_file, temperature, max_tokens)
    return commands_file


async def run_execution_phase(
    commands_file: Path,
    output_dir: str,
    port: int = 3000,
    sandbox_image: str = "deathbyknowledge/shellm-sandbox:latest",
    concurrency: int = 4,
    pool_size: int = 8,
) -> Path:
    """Run only the execution phase."""
    # Initialize SoS client
    sos = SoSClient(server_url=f"http://localhost:{port}")
    
    # Setup output
    output_path = Path(output_dir)
    results_file = output_path / "results.jsonl"
    
    await batch_execute_commands(commands_file, sos, results_file, sandbox_image, concurrency, pool_size)
    return results_file


async def run_both_phases(
    model_name: str,
    dataset_path: str,
    output_dir: str = "evaluation_results",
    max_samples: Optional[int] = None,
    port: int = 3000,
    sandbox_image: str = "deathbyknowledge/shellm-sandbox:latest",
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: float = 0.6,
    max_tokens: int = 2000,
    concurrency: int = 4,
    pool_size: int = 8,
) -> None:
    """Run both inference and execution phases."""
    print("=== Running Both Phases ===")
    
    # Phase 1: Generate commands
    commands_file = await run_inference_phase(
        model_name, dataset_path, output_dir, max_samples,
        base_url, api_key, temperature, max_tokens
    )
    
    # Phase 2: Execute commands
    results_file = await run_execution_phase(
        commands_file, output_dir, port, sandbox_image, concurrency, pool_size
    )
    
    # Generate summary
    results = load_jsonl(results_file)
    total_samples = len(results)
    successful_samples = sum(1 for r in results if r.get('overall_success', False))
    success_rate = successful_samples / total_samples if total_samples > 0 else 0
    
    # Count setup groups for additional stats
    setup_groups = defaultdict(int)
    for result in results:
        setup_hash = result.get('setup_hash', 'unknown')
        setup_groups[setup_hash] += 1
    
    summary = {
        "model_name": model_name,
        "dataset_path": dataset_path,
        "total_samples": total_samples,
        "successful_samples": successful_samples,
        "success_rate": success_rate,
        "failed_samples": total_samples - successful_samples,
        "setup_groups": len(setup_groups),
        "concurrency": concurrency,
        "pool_size": pool_size,
        "evaluation_time": utc_now_iso(),
        "sandbox_image": sandbox_image,
    }
    
    summary_file = Path(output_dir) / "summary.json"
    with summary_file.open("w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"=== Evaluation Complete ===")
    print(f"Total: {total_samples}, Success: {successful_samples} ({success_rate:.1%}), Failed: {total_samples - successful_samples}")
    print(f"Setup groups: {len(setup_groups)}, Concurrency: {concurrency}, Pool size: {pool_size}")
    print(f"Commands: {commands_file}")
    print(f"Results: {results_file}")
    print(f"Summary: {summary_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate OpenAI-compatible model on shell tasks (separated phases)")
    parser.add_argument("--model", help="Model name (e.g., 'gpt-4', 'gpt-3.5-turbo', 'mock' for testing)")
    parser.add_argument("--dataset", help="Dataset path or HuggingFace dataset name")
    parser.add_argument("--output-dir", default="evaluation_results", help="Output directory")
    parser.add_argument("--max-samples", type=int, help="Maximum samples to evaluate")
    parser.add_argument("--sandbox-image", default="deathbyknowledge/shellm-sandbox:latest", help="Docker image for sandbox")
    parser.add_argument("--base-url", help="OpenAI-compatible API base URL (default: https://api.openai.com/v1)")
    parser.add_argument("--api-key", help="API key (can also use OPENAI_API_KEY env var)")
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=2000, help="Maximum tokens to generate")
    parser.add_argument("--sos-port", type=int, default=3000, help="Port of SoS sandbox server")
    
    # Phase selection
    parser.add_argument("--phase", choices=["inference", "execution", "both"], default="both",
                       help="Which phase to run: inference only, execution only, or both")
    parser.add_argument("--commands-file", help="Path to commands file (required for execution-only phase)")
    
    # Performance tuning
    parser.add_argument("--concurrency", type=int, default=4, 
                       help="Number of concurrent sandbox executions (default: 4)")
    parser.add_argument("--pool-size", type=int, default=8,
                       help="Maximum number of sandboxes in pool (default: 8)")
    
    args = parser.parse_args()
    
    # Validate required arguments based on phase
    if args.phase in ["inference", "both"]:
        if not args.model:
            parser.error("--model is required for inference phase")
        if not args.dataset:
            parser.error("--dataset is required for inference phase")
    
    if args.phase == "execution":
        if not args.commands_file:
            parser.error("--commands-file is required for execution-only phase")
    
    async def main():
        if args.phase == "inference":
            commands_file = await run_inference_phase(
                model_name=args.model,
                dataset_path=args.dataset,
                output_dir=args.output_dir,
                max_samples=args.max_samples,
                base_url=args.base_url,
                api_key=args.api_key,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
            print(f"Inference complete. Commands saved to: {commands_file}")
            
        elif args.phase == "execution":
            if not args.commands_file:
                print("Error: --commands-file is required for execution-only phase")
                return
            
            commands_file = Path(args.commands_file)
            if not commands_file.exists():
                print(f"Error: Commands file not found: {commands_file}")
                return
                
            results_file = await run_execution_phase(
                commands_file=commands_file,
                output_dir=args.output_dir,
                port=args.sos_port,
                sandbox_image=args.sandbox_image,
                concurrency=args.concurrency,
                pool_size=args.pool_size,
            )
            print(f"Execution complete. Results saved to: {results_file}")
            
        else:  # both
            await run_both_phases(
                model_name=args.model,
                dataset_path=args.dataset,
                output_dir=args.output_dir,
                max_samples=args.max_samples,
                port=args.sos_port,
                sandbox_image=args.sandbox_image,
                base_url=args.base_url,
                api_key=args.api_key,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                concurrency=args.concurrency,
                pool_size=args.pool_size,
            )
    
    asyncio.run(main())