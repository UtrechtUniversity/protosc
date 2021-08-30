from multiprocessing import Queue, cpu_count, Process
from tqdm import tqdm


def execute_parallel(jobs, target, n_jobs=-1, args=[], kwargs={}):
    """Parallel running of measure jobs.

    Arguments
    ---------
    jobs: list[dict]
        Jobs to be processed in parallel
    target: function
        Function that executes these jobs.
    n_jobs: int
        Number of processes to use, -1 defaults to the number of cores.
    args: list
        Arguments used for all jobs.
    kwargs: dict
        Keyword arguments for all jobs.

    Returns
    -------
    results: list
        Unordered results of the computation.
    """
    pbar = tqdm(total=len(jobs))
    if n_jobs is None:
        n_jobs = 1
    elif n_jobs == -1:
        n_jobs = cpu_count()

    queue_size = len(jobs)
    job_queue = Queue(maxsize=1000)
    result_queue = Queue()

    args = (job_queue, result_queue, target, *args)
    jobs = [{**job, "job_id": i} for i, job in enumerate(jobs)]

    worker_procs = [
        Process(
            target=_wrapper_target,
            args=args,
            kwargs=kwargs,
            daemon=True,
        )
        for _ in range(n_jobs)
    ]
    for proc in worker_procs:
        proc.start()

    for job in jobs:
        job_queue.put(job)

    results = []
    for _ in range(queue_size):
        results.append(result_queue.get())
        pbar.update(1)

    for _ in range(n_jobs):
        job_queue.put(None)

    for proc in worker_procs:
        proc.join()
    ordered_results = [None]*len(results)
    for res in results:
        job_id = res[0]["job_id"]
        ordered_results[job_id] = res[1]

    pbar.close()

    return ordered_results


def _wrapper_target(job_queue, output_queue, exec_func, *args, **kwargs):
    while True:
        job = job_queue.get(block=True)
        if job is None:
            break
        job_id = job.pop("job_id")
        results = exec_func(*args, **kwargs, **job)
        job["job_id"] = job_id
        output_queue.put((job, results))
