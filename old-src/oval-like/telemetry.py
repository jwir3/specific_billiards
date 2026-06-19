import multiprocessing as mp

# =====================================================================
# HOW-TO: NON-BLOCKING TELEMETRY WITH PROCESSPOOLEXECUTOR
# =====================================================================
"""
To reuse this progress tracking architecture for a non-billiard parallel
workload, copy or adapt the boilerplate script layout below.

This model enforces a fire-and-forget streaming paradigm: workers continuously
push individual computational 'ticks' across a Manager.Queue, while the main
orchestration loop catches those signals on a decoupled poll interval[cite: 17, 21].
This eliminates UI stalls during heavy waves of multi-core processing[cite: 22, 25].

Requirements:
    pip install tqdm

Example Script Execution Layout:
-----------------------------------------------------------------------
import time
import math
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from telemetry import Monitor, Heartbeat

def generic_worker_task(param_x, total_iterations, progress_queue):
    '''
    A representative worker task processing computational blocks.
    Streams a heartbeat signal straight to the shared queue.
    '''
    # Initialize the heartbeat reporter with the passed shared queue
    hb = Heartbeat(queue=progress_queue)

    result_accumulator = 0.0
    reporting_interval = 500  # Stream a tick every 500 operations

    for i in range(1, total_iterations + 1):
        # Perform arbitrary heavy numerical work
        result_accumulator += math.sin(i * param_x)

        # Fire a progress signal immediately as work happens
        if i % reporting_interval == 0:
            hb.report(1)

    return param_x, result_accumulator

def run_generic_sweep():
    # 1. Initialize the telemetry engine and clear previous state
    monitor = Monitor()
    monitor.clear()

    # 2. Define your sweep boundaries and granular iteration depths
    test_parameters = [0.1 * i for i in range(1, 33)]  # 32 steps
    iterations_per_step = 100000                       # Work per step

    # 3. Calculate your progress bar ceiling matching your task ticks
    ticks_per_task = iterations_per_step // 500
    total_expected_ticks = len(test_parameters) * ticks_per_task

    pbar = tqdm(total=total_expected_ticks, desc="Sweep Progress", unit="tick")

    # 4. Map task arguments alongside the shared monitor.queue pointer
    task_args = [
        (param, iterations_per_step, monitor.queue)
        for param in test_parameters
    ]

    # 5. Distribute concurrent tasks across your available core architecture
    compiled_results = []
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(generic_worker_task, *arg) for arg in task_args]

        # Actively poll and drain the queue while the CPUs are crunching numbers
        while any(f.running() for f in futures):
            monitor.drain_into_pbar(pbar)
            time.sleep(0.5)  # Decoupled UI frame-rate limiter

        # Perform a final sweep to catch late-arriving heartbeats
        monitor.drain_into_pbar(pbar)
        pbar.n = pbar.total
        pbar.refresh()
        pbar.close()

        # 6. Safely unpack values after complete pool termination
        for f in futures:
            param, final_res = f.result()
            compiled_results.append({"parameter": param, "result": final_res})

    print(f"Sweep complete. Processed {len(compiled_results)} tasks successfully.")
    monitor.clear()

if __name__ == "__main__":
    run_generic_sweep()
-----------------------------------------------------------------------
"""

class Heartbeat:
    """Used by worker processes to report progress directly via a shared queue."""

    def __init__(self, queue=None):
        self.queue = queue

    def report(self, current_tick=1):
        """Pushes a tick to the shared queue."""
        if self.queue:
            self.queue.put(current_tick)

    def cleanup(self):
        """No longer needs to delete files, but kept for interface consistency."""
        pass


class Monitor:
    """Used by the main process to aggregate progress from the queue."""

    def __init__(self):
        self.manager = mp.Manager()
        self.queue = self.manager.Queue()

    def drain_into_pbar(self, pbar):
        """Empty the queue and update the progress bar with all available ticks."""
        while not self.queue.empty():
            try:
                # We update by 1 for every tick sent by the heartbeat
                pbar.update(self.queue.get_nowait())
            except:
                break

    def clear(self):
        """No-op for queue-based telemetry."""
        pass
