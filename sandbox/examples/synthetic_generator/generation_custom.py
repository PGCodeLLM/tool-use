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
Model Evaluation Pipeline using SoS (Sea of Simulation)

This script evaluates an OpenAI-compatible model's ability to generate shell commands.
Given a dataset with [setup_commands, success_condition, task] columns:

1. Load model using OpenAI API (or compatible endpoint)
2. For each task: generate shell command using the model
3. Create sandbox and run setup commands
4. Execute the model's generated command
5. Test success by running success_condition (exit code 0 = success)
6. Log results and statistics
"""

import os
import json
import uuid
import asyncio
import httpx
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from rich import print
from rich.progress import Progress, TaskID

from openai import AsyncOpenAI
from datasets import load_dataset, Dataset
from tenacity import retry, stop_after_attempt
from sos import SoSClient


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


def setup_logger(output_dir: Path, task_id: str) -> logging.Logger:
    """Setup a logger for a specific evaluation task."""
    logger = logging.getLogger(f"eval_{task_id}")
    logger.setLevel(logging.ERROR)
    
    # Remove any existing handlers
    logger.handlers.clear()
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create file handler
    log_file = output_dir / f"task_{task_id}.log"
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.ERROR)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    return logger


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
    
    async def generate_command(self, task: str, temperature: float = 0.1, max_tokens: int = 512) -> str:
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
    async def _real_generate_command(self, task: str, temperature: float = 0.1, max_tokens: int = 128) -> str:
        """Generate shell command using real API."""
        # messages = [
        #     {
        #         "role": "system", 
        #         "content": "You are an expert Linux shell user. Given a task description, provide ONLY the shell command needed to complete it. Do not include any thinking, explanations, reasoning, or additional text. Respond with just the raw shell command that can be executed directly."
        #     },
        #     {
        #         "role": "user", 
        #         "content": f"Task: {task}\n\nShell command:"
        #     }
            
        # ]
        messages = [
            {
                "role": "user", 
                "content": f"{task}"
            }
            
        ]
        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            
            generated_text = response.choices[0].message.content or ""
            print(f"Raw model output: {repr(generated_text)}")
            
            # Clean up the response - remove thinking tokens and extract command
            lines = generated_text.strip().split('\n')
            command = ""
            
            for line in lines:
                line = line.strip()
                # Skip thinking tokens, explanations, or empty lines
                if (line.startswith('<think>') or line.startswith('</think>') or 
                    line.startswith('#') or line == "" or
                    line.lower().startswith('here') or line.lower().startswith('the command')):
                    continue
                # Take the first valid line as the command
                if line:
                    command = line
                    break
            
            # Remove common prefixes if present
            prefixes_to_remove = ["$ ", "# ", "bash: ", "shell: ", "`", "```bash", "```"]
            for prefix in prefixes_to_remove:
                if command.startswith(prefix):
                    command = command[len(prefix):]
                    break
            
            # Remove trailing backticks
            command = command.rstrip('`').strip()
            
            return command
            
        except Exception as e:
            print(f"Error generating command: {e}")
            raise


@retry(stop=stop_after_attempt(3))
async def evaluate_single_task(
    model: ModelInterface, 
    task_data: Dict[str, Any], 
    sos: SoSClient,
    output_dir: Path,
    sandbox_image: str = "deathbyknowledge/shellm-sandbox:latest"
) -> Dict[str, Any]:
    """
    Evaluate model on a single task.
    
    Args:
        model: HuggingFace model interface
        task_data: Dict with 'task', 'setup_commands', 'success_condition'
        sos: SoS client
        sandbox_image: Docker image for sandbox
    
    Returns:
        Dict with evaluation results
    """
    task_id = short_id()
    start_time = utc_now_iso()
    
    # Setup logger for this task
    logger = setup_logger(output_dir, task_id)
    
    try:
        logger.info(f"STEP 1: Generate command for task")
        logger.info(f"Task: {task_data['task']}")
        
        # 1. Generate command using model
        generated_command = await model.generate_command(task_data['task'])
        logger.info(f"Generated command: {generated_command}")
        
        logger.info(f"STEP 2: Setup sandbox environment")
        # 2. Create sandbox and run setup
        setup_commands = task_data['setup_commands']
        if isinstance(setup_commands, list):
            setup_commands = "; ".join(setup_commands)
        
        logger.info(f"Setup commands: {setup_commands}")
        
        sid = await sos.create_sandbox(
            image=sandbox_image, 
            setup_commands=[setup_commands] if setup_commands else []
        )
        logger.info(f"Created sandbox: {sid}")
        
        try:
            logger.info(f"Starting sandbox {sid[:8]}...")
            await sos.start_sandbox(sid)  # This runs setup commands internally
            logger.info(f"Sandbox {sid[:8]} started and setup completed successfully")
            
            logger.info(f"STEP 3: Execute generated command")
            # 3. Execute the generated command
            logger.info(f"Executing: {generated_command}")
            output, exit_code, exited = await sos.exec_command(sid, generated_command)
            command_success = exit_code == 0
            logger.info(f"Command exit code: {exit_code} (success: {command_success})")
            if output.strip():
                logger.info(f"Command output: {output.strip()}")
            
            logger.info(f"STEP 4: Test success condition")
            # 4. Test success condition
            success_condition = task_data['success_condition']
            logger.info(f"Testing success condition: {success_condition}")
            _, test_exit_code, _ = await sos.exec_command(sid, success_condition, standalone=True)
            test_passed = test_exit_code == 0
            logger.info(f"Success test exit code: {test_exit_code} (passed: {test_passed})")
            
            # Get full trajectory for debugging
            trajectory = await sos.get_sandbox_trajectory(sid, formatted=False)
            
            overall_success = command_success and test_passed
            logger.info(f"FINAL RESULT: {'SUCCESS' if overall_success else 'FAILED'}")
            
            result = {
                "task_id": task_id,
                "task": task_data['task'],
                "setup_commands": setup_commands,
                "success_condition": success_condition,
                "generated_command": generated_command,
                "command_output": output,
                "command_exit_code": exit_code,
                "command_success": command_success,
                "test_exit_code": test_exit_code,
                "test_passed": test_passed,
                "overall_success": overall_success,
                "trajectory": trajectory.get('trajectory', []),
                "start_time": start_time,
                "end_time": utc_now_iso(),
                "exited": exited,
            }
            
            return result
            
        finally:
            await sos.stop_sandbox(sid, remove=True)
            
    except Exception as e:
        logger.error(f"ERROR: {str(e)}")
        return {
            "task_id": task_id,
            "task": task_data['task'],
            "setup_commands": task_data.get('setup_commands', ''),
            "success_condition": task_data['success_condition'],
            "generated_command": generated_command if 'generated_command' in locals() else None,
            "error": str(e),
            "overall_success": False,
            "start_time": start_time,
            "end_time": utc_now_iso(),
        }


async def evaluate_model(
    model_name: str,
    dataset_path: str,
    output_dir: str = "evaluation_results",
    max_samples: Optional[int] = None,
    sandbox_image: str = "deathbyknowledge/shellm-sandbox:latest",
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: float = 0.1,
    max_tokens: int = 512
) -> None:
    """
    Main evaluation pipeline.
    
    Args:
        model_name: OpenAI-compatible model name
        dataset_path: Path to dataset or HuggingFace dataset name
        output_dir: Directory to save results
        max_samples: Maximum number of samples to evaluate (None = all)
        sandbox_image: Docker image for sandboxes
        base_url: OpenAI-compatible API base URL
        api_key: API key for the model service
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
    """
    # Initialize
    output_path = Path(output_dir)
    results_file = output_path / "results.jsonl"
    summary_file = output_path / "summary.json"
    
    # Load model
    mock_mode = model_name.lower() == "mock" or base_url is None
    model = ModelInterface(model_name, base_url=base_url, api_key=api_key, mock_mode=mock_mode)
    
    # Load dataset
    if os.path.exists(dataset_path):
        # Local file
        if dataset_path.endswith('.jsonl'):
            dataset_data = load_jsonl(Path(dataset_path))
            dataset = Dataset.from_list(dataset_data)
        else:
            dataset = load_dataset(dataset_path, split="test")
    else:
        # HuggingFace dataset
        dataset = load_dataset(dataset_path, split="test")
    
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    print(f"Evaluating {len(dataset)} samples")
    
    # Initialize SoS client
    sos = SoSClient(server_url="http://localhost:3000")
    
    # Run evaluation with progress bar
    results = []
    success_count = 0
    
    with Progress() as progress:
        task_progress = progress.add_task("Evaluating", total=len(dataset))
        
        for i, sample in enumerate(dataset):
            result = await evaluate_single_task(model, sample, sos, output_path, sandbox_image)
            results.append(result)
            
            if result.get('overall_success', False):
                success_count += 1
            
            # Save result immediately
            append_jsonl(results_file, result)
            
            # Update progress
            progress.update(
                task_progress, 
                advance=1,
                description=f"Evaluating ({success_count}/{i+1} successful)"
            )
            
            # Print status every 10 samples
            if (i + 1) % 10 == 0:
                success_rate = success_count / (i + 1) * 100
                print(f"Progress: {i+1}/{len(dataset)} - Success rate: {success_rate:.1f}%")
    
    # Generate summary
    total_samples = len(results)
    successful_samples = sum(1 for r in results if r.get('overall_success', False))
    success_rate = successful_samples / total_samples if total_samples > 0 else 0
    
    summary = {
        "model_name": model_name,
        "dataset_path": dataset_path,
        "total_samples": total_samples,
        "successful_samples": successful_samples,
        "success_rate": success_rate,
        "failed_samples": total_samples - successful_samples,
        "evaluation_time": utc_now_iso(),
        "sandbox_image": sandbox_image,
    }
    
    with summary_file.open("w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"Evaluation Complete - Total: {total_samples}, Success: {successful_samples} ({success_rate:.1%}), Failed: {total_samples - successful_samples}")
    print(f"Results: {results_file}, Summary: {summary_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate OpenAI-compatible model on shell tasks")
    parser.add_argument("--model", required=True, help="Model name (e.g., 'gpt-4', 'gpt-3.5-turbo', 'mock' for testing)")
    parser.add_argument("--dataset", required=True, help="Dataset path or HuggingFace dataset name")
    parser.add_argument("--output-dir", default="evaluation_results", help="Output directory")
    parser.add_argument("--max-samples", type=int, help="Maximum samples to evaluate")
    parser.add_argument("--sandbox-image", default="deathbyknowledge/shellm-sandbox:latest", help="Docker image for sandbox")
    parser.add_argument("--base-url", help="OpenAI-compatible API base URL (default: https://api.openai.com/v1)")
    parser.add_argument("--api-key", help="API key (can also use OPENAI_API_KEY env var)")
    parser.add_argument("--temperature", type=float, default=0.1, help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=512, help="Maximum tokens to generate")
    
    args = parser.parse_args()
    
    async def main():
        await evaluate_model(
            model_name=args.model,
            dataset_path=args.dataset,
            output_dir=args.output_dir,
            max_samples=args.max_samples,
            sandbox_image=args.sandbox_image,
            base_url=args.base_url,
            api_key=args.api_key,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
    
    asyncio.run(main())
