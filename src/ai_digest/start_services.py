"""Start Docker, Qdrant, and the Ollama server, then report readiness.

Defaults are suitable for a local development setup:
    Qdrant HTTP: localhost:6333
    Qdrant gRPC: localhost:6334
    Ollama API:  localhost:11434

Usage:
    from start_services import start_services

    if start_services():
        # Continue with the rest of your application.
        ...

Run directly to test the dependencies:
    python start_services.py

Optional environment variables:
    QDRANT_IMAGE                 qdrant/qdrant:latest
    QDRANT_CONTAINER_NAME        amg-qdrant
    QDRANT_HTTP_PORT             6333
    QDRANT_GRPC_PORT             6334
    QDRANT_VOLUME                amg_qdrant_storage
    OLLAMA_HOST                  127.0.0.1:11434
    STARTUP_TIMEOUT_SECONDS      180

Notes:
- Docker CLI and Ollama must already be installed.
- On Linux, starting the Docker daemon may require passwordless sudo or that
  Docker is already enabled/running. The script does not prompt for a password.
- `ollama serve` starts the Ollama API server; it does not download or load a
  particular model. Pull/run a model separately when your application needs it.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Configuration (can be overridden with environment variables)
# ---------------------------------------------------------------------------
QDRANT_IMAGE = os.getenv("QDRANT_IMAGE", "qdrant/qdrant:latest")
QDRANT_CONTAINER_NAME = os.getenv("QDRANT_CONTAINER_NAME", "amg-qdrant")
QDRANT_HTTP_PORT = int(os.getenv("QDRANT_HTTP_PORT", "6333"))
QDRANT_GRPC_PORT = int(os.getenv("QDRANT_GRPC_PORT", "6334"))
QDRANT_VOLUME = os.getenv("QDRANT_VOLUME", "amg_qdrant_storage")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "127.0.0.1:11434")
STARTUP_TIMEOUT_SECONDS = int(os.getenv("STARTUP_TIMEOUT_SECONDS", "180"))
POLL_INTERVAL_SECONDS = 2


def _run_command(
    command: list[str], *, timeout: int = 30
) -> subprocess.CompletedProcess[str]:
    """Run a command and capture its output without raising on non-zero exit."""
    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _docker_info_is_available() -> bool:
    """Return True when the Docker daemon is reachable through the CLI."""
    docker = shutil.which("docker")
    if not docker:
        print("[ERROR] Docker CLI was not found. Install Docker and add it to PATH.")
        return False

    try:
        result = _run_command([docker, "info"], timeout=10)
        return result.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def _launch_docker_desktop() -> bool:
    """Try to launch Docker Desktop on Windows or macOS."""
    system = platform.system()

    if system == "Windows":
        candidates = [
            Path(os.getenv("ProgramFiles", r"C:\Program Files"))
            / "Docker"
            / "Docker"
            / "Docker Desktop.exe",
            Path(os.getenv("LOCALAPPDATA", ""))
            / "Programs"
            / "Docker"
            / "Docker"
            / "Docker Desktop.exe",
        ]
        executable = next((path for path in candidates if path.is_file()), None)
        if executable is None:
            print("[ERROR] Docker Desktop executable was not found.")
            return False

        try:
            subprocess.Popen(
                [str(executable)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=(
                    getattr(subprocess, "DETACHED_PROCESS", 0)
                    | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                ),
            )
            print("[INFO] Starting Docker Desktop...")
            return True
        except OSError as exc:
            print(f"[ERROR] Could not launch Docker Desktop: {exc}")
            return False

    if system == "Darwin":
        try:
            subprocess.Popen(
                ["open", "-a", "Docker"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            print("[INFO] Starting Docker Desktop...")
            return True
        except OSError as exc:
            print(f"[ERROR] Could not launch Docker Desktop: {exc}")
            return False

    return False

def _try_start_linux_docker() -> bool:
    """Start Docker Desktop on Linux using systemd user service."""

    if platform.system() != "Linux":
        return False

    if not shutil.which("systemctl"):
        print("[ERROR] systemctl is not available.")
        return False

    print("[INFO] Starting Docker Desktop...")

    try:
        result = _run_command(
            ["systemctl", "--user", "start", "docker-desktop"],
            timeout=45,
        )

        if result.returncode != 0:
            print(
                "[ERROR] Could not start Docker Desktop:\n"
                f"{result.stderr.strip()}"
            )
            print(
                "[INFO] Check using: "
                "systemctl --user status docker-desktop"
            )
            return False

        print("[OK] Docker Desktop startup requested.")

    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f"[ERROR] Docker Desktop startup failed: {exc}")
        return False

    docker = shutil.which("docker")

    if not docker:
        print("[ERROR] Docker CLI not found.")
        return False

    # Wait for Docker Desktop context to become available.
    deadline = time.monotonic() + 30

    while time.monotonic() < deadline:

        context = _run_command(
            [docker, "context", "inspect", "desktop-linux"],
            timeout=10,
        )

        if context.returncode == 0:

            # Switch Docker CLI to Docker Desktop.
            switch = _run_command(
                [docker, "context", "use", "desktop-linux"],
                timeout=10,
            )

            if switch.returncode == 0:
                print("[OK] Docker context: desktop-linux")
                return True

            print(
                f"[ERROR] Could not switch Docker context: "
                f"{switch.stderr.strip()}"
            )
            return False

        time.sleep(POLL_INTERVAL_SECONDS)

    print("[ERROR] Docker Desktop context was not available.")
    return False


def ensure_docker_running(
    timeout: int = STARTUP_TIMEOUT_SECONDS
) -> bool:
    """Ensure Docker Desktop is running and reachable."""

    system = platform.system()

    if system == "Linux":

        # Start Docker Desktop and select desktop-linux context.
        launched = _try_start_linux_docker()

    else:

        if _docker_info_is_available():
            print("[OK] Docker daemon is running.")
            return True

        if system in {"Windows", "Darwin"}:
            launched = _launch_docker_desktop()
        else:
            launched = False

    if not launched:
        return False

    print("[INFO] Waiting for Docker daemon...")

    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:

        if _docker_info_is_available():
            print("[OK] Docker Desktop is ready.")
            return True

        time.sleep(POLL_INTERVAL_SECONDS)

    print(
        f"[ERROR] Docker Desktop did not become ready "
        f"within {timeout} seconds."
    )

    return False


def _container_exists() -> bool:
    result = _run_command(
        ["docker", "container", "inspect", QDRANT_CONTAINER_NAME], timeout=15
    )
    return result.returncode == 0


def _container_is_running() -> bool:
    result = _run_command(
        [
            "docker",
            "container",
            "inspect",
            "--format",
            "{{.State.Running}}",
            QDRANT_CONTAINER_NAME,
        ],
        timeout=15,
    )
    return result.returncode == 0 and result.stdout.strip().lower() == "true"


def start_qdrant() -> bool:
    """Pull the Qdrant image if needed, then create or start its container."""
    try:
        image_check = _run_command(
            ["docker", "image", "inspect", QDRANT_IMAGE], timeout=20
        )
        if image_check.returncode != 0:
            print(f"[INFO] Pulling Qdrant image: {QDRANT_IMAGE}")
            pull = subprocess.run(["docker", "pull", QDRANT_IMAGE], check=False)
            if pull.returncode != 0:
                print("[ERROR] Failed to pull the Qdrant Docker image.")
                return False

        if _container_exists():
            if _container_is_running():
                print(f"[OK] Qdrant container '{QDRANT_CONTAINER_NAME}' is already running.")
            else:
                print(f"[INFO] Starting existing Qdrant container '{QDRANT_CONTAINER_NAME}'...")
                start = _run_command(
                    ["docker", "start", QDRANT_CONTAINER_NAME], timeout=30
                )
                if start.returncode != 0:
                    print(f"[ERROR] Could not start Qdrant: {start.stderr.strip()}")
                    return False
        else:
            command = [
                "docker",
                "run",
                "-d",
                "--restart",
                "unless-stopped",
                "--name",
                QDRANT_CONTAINER_NAME,
                "-p",
                f"{QDRANT_HTTP_PORT}:6333",
                "-p",
                f"{QDRANT_GRPC_PORT}:6334",
                "-v",
                f"{QDRANT_VOLUME}:/qdrant/storage",
                QDRANT_IMAGE,
            ]
            print("[INFO] Creating and starting Qdrant container...")
            run = _run_command(command, timeout=60)
            if run.returncode != 0:
                print(f"[ERROR] Could not create Qdrant container: {run.stderr.strip()}")
                return False

        return wait_for_http(
            f"http://127.0.0.1:{QDRANT_HTTP_PORT}",
            timeout=STARTUP_TIMEOUT_SECONDS,
            service_name="Qdrant",
            paths=("/readyz", "/healthz", "/"),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f"[ERROR] Qdrant startup failed: {exc}")
        return False


def _find_ollama_executable() -> Optional[str]:
    """Find the Ollama executable, including common Windows install paths."""
    executable = shutil.which("ollama")
    if executable:
        return executable

    if platform.system() == "Windows":
        candidates = [
            Path(os.getenv("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe",
            Path(os.getenv("ProgramFiles", r"C:\Program Files")) / "Ollama" / "ollama.exe",
        ]
        for path in candidates:
            if path.is_file():
                return str(path)

    return None


def _ollama_api_url() -> str:
    host = OLLAMA_HOST.strip()
    if "://" not in host:
        host = f"http://{host}"
    return host.rstrip("/")


def start_ollama() -> bool:
    """Start `ollama serve` when the API is not already responding."""
    api_url = _ollama_api_url()
    if wait_for_http(
        f"{api_url}/api/tags",
        timeout=3,
        service_name="Ollama",
        quiet=True,
    ):
        print("[OK] Ollama server is already running.")
        return True

    executable = _find_ollama_executable()
    if not executable:
        print("[ERROR] Ollama was not found. Install Ollama and add it to PATH.")
        return False

    log_path = Path(__file__).resolve().with_name("ollama-serve.log")
    try:
        log_file = log_path.open("a", encoding="utf-8")
        env = os.environ.copy()
        env["OLLAMA_HOST"] = OLLAMA_HOST

        popen_kwargs: dict[str, object] = {
            "stdout": log_file,
            "stderr": subprocess.STDOUT,
            "stdin": subprocess.DEVNULL,
            "env": env,
            "cwd": str(Path.home()),
        }
        if platform.system() == "Windows":
            popen_kwargs["creationflags"] = (
                getattr(subprocess, "DETACHED_PROCESS", 0)
                | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            )
        else:
            popen_kwargs["start_new_session"] = True

        subprocess.Popen([executable, "serve"], **popen_kwargs)  # type: ignore[arg-type]
        # The child owns the duplicated file descriptor after Popen returns.
        log_file.close()
        print(f"[INFO] Starting Ollama server; logs: {log_path}")

        if wait_for_http(
            f"{api_url}/api/tags",
            timeout=STARTUP_TIMEOUT_SECONDS,
            service_name="Ollama",
        ):
            return True

        print(f"[ERROR] Ollama did not become ready. Check {log_path} for details.")
        return False
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"[ERROR] Could not launch Ollama: {exc}")
        return False


def wait_for_http(
    url: str,
    *,
    timeout: int,
    service_name: str,
    paths: tuple[str, ...] = ("",),
    quiet: bool = False,
) -> bool:
    """Poll an HTTP endpoint until it responds with a 2xx/3xx status."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for path in paths:
            endpoint = url.rstrip("/") + path
            try:
                request = urllib.request.Request(
                    endpoint,
                    headers={"User-Agent": "local-ai-service-starter/1.0"},
                )
                with urllib.request.urlopen(request, timeout=3) as response:
                    if 200 <= response.status < 400:
                        if not quiet:
                            print(f"[OK] {service_name} is ready ({endpoint}).")
                        return True
            except (urllib.error.URLError, TimeoutError, OSError):
                continue
        time.sleep(POLL_INTERVAL_SECONDS)

    if not quiet:
        print(f"[ERROR] {service_name} did not respond within {timeout} seconds.")
    return False


def start_services() -> bool:
    """Start Docker, Qdrant, and Ollama. Return True only when both APIs are ready."""
    print("\n========== Starting local AI services ==========")

    if not ensure_docker_running():
        return False

    if not start_qdrant():
        return False

    if not start_ollama():
        return False

    print("\n[SUCCESS] Docker, Qdrant, and Ollama are ready.")
    return True


if __name__ == "__main__":
    if start_services():
        print("[SUCCESS] start_services() returned True. Continue with your application.")
        sys.exit(0)

    print("[FAILED] One or more services could not be started.")
    sys.exit(1)
