from typing import Tuple
import subprocess
import time
from tabulate import tabulate


def benchmark(command: str) -> Tuple[str, float]:
    start_time = time.time()
    process = subprocess.run(command, shell=True, text=True)
    end_time = time.time()
    process_time = end_time - start_time
    return process.stdout, process_time


if __name__ == '__main__':
    commands = [
        #'python main.py -f benchmark/source.jpg -t benchmark/target-240p.mp4 -o benchmark1.mp4 --cli --threads 12 --optimization fp16 --lowmem --fast-load',
        #'python main.py -f benchmark/source.jpg -t benchmark/target-360p.mp4 -o benchmark2.mp4 --cli --threads 12 --optimization fp32 --lowmem --fast-load',
        #'python main.py -f benchmark/source.jpg -t benchmark/target-540p.mp4 -o benchmark3.mp4 --cli --threads 12 --optimization fp32 --lowmem --fast-load',
        #'python main.py -f benchmark/source.jpg -t benchmark/target-720p.mp4 -o benchmark4.mp4 --cli --threads 12 --optimization fp32 --lowmem --fast-load',
        'python main.py -f benchmark/source.jpg -t benchmark/target-1080p.mp4 -o benchmark5.mp4 --cli --threads 12 --optimization fp16 --lowmem --fast-load --occluder --rembg',
        'python main.py -f benchmark/source.jpg -t benchmark/target-1080p.mp4 -o benchmark5.mp4 --cli --threads 12 --optimization fp16 --lowmem --fast-load --occluder --rembg',
        'python main.py -f benchmark/source.jpg -t benchmark/target-1080p.mp4 -o benchmark5.mp4 --cli --threads 12 --optimization fp16 --lowmem --fast-load --occluder --rembg',
        #'python main.py -f benchmark/source.jpg -t benchmark/target-1440p.mp4 -o benchmark6.mp4 --cli --threads 12 --optimization fp32 --lowmem --fast-load',
        #'python main.py -f benchmark/source.jpg -t benchmark/target-2160p.mp4 -o benchmark7.mp4 --cli --threads 12 --optimization fp32 --lowmem --fast-load'
    ]

    results = []
    for command in commands:
        output, execution_time = benchmark(command)
        results.append([command, f'{execution_time:.2f} seconds'])

    print(tabulate(results, headers=['Command', 'Execution Time']))
