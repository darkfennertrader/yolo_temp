import subprocess
import shlex
import time


def get_gpu_process_ids():
    # Getting output from nvidia-smi
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
        stdout=subprocess.PIPE,
    )
    output = result.stdout.decode("utf-8")

    # Parsing output to get PIDs
    pids = [int(pid.strip()) for pid in output.split("\n") if pid.strip().isdigit()]
    return pids


def kill_processes(pids):
    for pid in pids:
        try:
            subprocess.run(["kill", "-9", str(pid)])
            print(f"Process {pid} has been killed.")
        except Exception as e:
            print(f"Failed to kill process {pid}: {e}")


def restart_process(command):
    # Assuming command is a string like 'python my_script.py'
    subprocess.Popen(shlex.split(command))
    print("Process restarted.")


def clear_gpus_memory():
    # Usage Example
    pids = get_gpu_process_ids()
    print(f"Killing PIDs: {pids}")
    kill_processes(pids)
    # Giving a little bit of time for cleanup.
    time.sleep(5)
