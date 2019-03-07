import time
import glob
from multiprocessing import Process, Manager, cpu_count
from multiprocessing.pool import ThreadPool

from schedule_solver import SolutionManager


def main():
    test_files = glob.glob('test02/test_*')

    total_files = len(test_files)
    results_fetched = 0
    failed = 0
    passed = 0

    results_queue = Manager().Queue()
    pool_data = [(results_queue, idx, test_file) for idx, test_file in enumerate(test_files)]

    num_threads = min(len(test_files) // cpu_count(), 3)
    print('Initializing with {} threads'.format(num_threads))
    process_pool = ThreadPool(num_threads)
    process_pool.map_async(func=perform_test, iterable=pool_data)

    while results_fetched < total_files:
        idx, test_passed, time_elapsed = results_queue.get()
        if test_passed:
            passed += 1
        else:
            failed += 1

        passed_string = 'PASS' if test_passed else 'FAIL'
        results_fetched += 1

        print('[{}/{}/{}] Result {},{}: {} in time {} seconds'.format(results_fetched, passed, failed, idx, test_files[idx], passed_string, time_elapsed))

    print('Result: {}/{} Passed = {}'.format(passed, total_files, passed * 100.0 / total_files))
    print('Result: {}/{} Failed = {}'.format(failed, total_files, failed * 100.0 / total_files))


def perform_test(args):
    results_queue, idx, input_file = args
    out_file = input_file.replace('test_', 'out_test_')
    execution_process = Process(target=execute_solver, args=(input_file, out_file))
    execution_process.start()

    start_time = time.time()
    elapsed_time = 0.0
    while execution_process.is_alive():
        elapsed_time = time.time() - start_time
        if elapsed_time > 60:
            execution_process.terminate()
            results_queue.put((idx, False, elapsed_time))
            return

    results_queue.put((idx, True, elapsed_time))
    return


def execute_solver(input_file, output_file):
    solution_manager = SolutionManager(input_file=input_file, output_file=output_file)
    solution_manager.solve()


if __name__ == "__main__":
    main()