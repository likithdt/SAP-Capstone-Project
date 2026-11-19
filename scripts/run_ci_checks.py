"""
Local CI/CD Verification Runner.
Executes all Continuous Integration stages locally:
  1. Python and JavaScript Syntax & Compilation Validation
  2. Multi-Sensor Physics Simulator & Feature Extraction Unit Tests
  3. Closed-Loop ERP Action Jury Demonstration (4 Scenarios)
  4. Capstone KPI Benchmark Verification
"""

import sys
import time
import subprocess
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()


def run_stage(name: str, cmd: list, cwd: Path = PROJECT_ROOT) -> tuple[bool, float, str]:
    """Execute a single CI pipeline stage and measure latency."""
    console.print(f"[bold cyan]▶ Running Stage:[/bold cyan] {name}...")
    start_time = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        elapsed = time.perf_counter() - start_time
        success = (proc.returncode == 0)
        return success, elapsed, proc.stdout
    except Exception as exc:
        elapsed = time.perf_counter() - start_time
        return False, elapsed, str(exc)


def main():
    console.print(Panel.fit(
        "[bold green]Industrial IoT Digital Twin · Local CI/CD Pipeline Runner[/bold green]\n"
        "[dim]Simulating GitHub Actions CI Stages: Syntax, Tests, Demo & KPI Benchmarking[/dim]",
        border_style="green"
    ))

    py_exe = sys.executable

    stages = [
        (
            "1. Python Syntax & Compilation",
            [py_exe, "-m", "compileall", "-q", "src", "amazon", "scripts", "main.py"],
        ),
        (
            "2. Unit Tests (Simulator & Features)",
            [py_exe, "-m", "pytest", "tests/test_simulator.py", "tests/test_features.py", "-q"],
        ),
        (
            "3. Closed-Loop Jury Demonstration (4 Scenarios)",
            [py_exe, "main.py", "demo"],
        ),
        (
            "4. Capstone KPI Benchmark Verification",
            [py_exe, "main.py", "benchmark", "--samples", "1"],
        ),
    ]

    results = []
    overall_success = True

    for name, cmd in stages:
        success, elapsed, output = run_stage(name, cmd)
        results.append((name, success, elapsed, output))
        if not success:
            overall_success = False
            console.print(f"[bold red]✖ FAILED:[/bold red] {name} ({elapsed:.2f}s)")
            console.print(Panel(output[:1000], title="Stage Output", border_style="red"))
            break
        else:
            console.print(f"[bold green]✔ PASSED:[/bold green] {name} ({elapsed:.2f}s)")

    # Print Summary Table
    table = Table(title="CI/CD Pipeline Execution Summary", border_style="cyan")
    table.add_column("Stage", style="bold white")
    table.add_column("Duration", style="yellow")
    table.add_column("Status", style="bold")

    for name, success, elapsed, _ in results:
        status_str = "[bold green]PASS[/bold green]" if success else "[bold red]FAIL[/bold red]"
        table.add_row(name, f"{elapsed:.2f}s", status_str)

    console.print(table)

    if overall_success:
        console.print("\n[bold green]🎉 All CI/CD pipeline checks passed successfully! Ready for push & deployment.[/bold green]\n")
        sys.exit(0)
    else:
        console.print("\n[bold red]❌ Pipeline failed. Please resolve the errors above before pushing.[/bold red]\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
