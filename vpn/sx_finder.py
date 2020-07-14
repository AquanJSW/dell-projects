"""
PureVPN sx* servers finder
"""

import time
import os
import subprocess
import multiprocessing
import tqdm
import argparse

parser = argparse.ArgumentParser(description='filter available sx prefixed PureVPN servers')
parser.add_argument('-p', '--path', default='/home/tjh/projects/vpn/ikev.txt',
                    type=str, help='servers list')
parse = parser.parse_args()

cores = multiprocessing.cpu_count()
cores = 3


def main(path):
    jobs_ = jobs(path)
    pool = multiprocessing.Pool(processes=cores)
    # results = pool.map(finder, jobs_)   # 返回 a list of lists of urls
    result = []
    for i in tqdm.tqdm(pool.map(finder, jobs_)):
        result += i
    # result = []
    # for result_ in results:
    #     result += result_
    print(result)
    file = time.asctime(time.localtime(time.time()))
    for url in result:
        subprocess.Popen("echo %s >> /home/tjh/projects/vpn/available_%s.txt" % (url, file), shell=True, stdout=subprocess.PIPE)


# 接受 a list of urls, 返回 a list of urls
def finder(urls):
    result = []
    for url in urls:
        out = subprocess.Popen("ping -qnc 1 {}".format(url), shell=True, stdout=subprocess.PIPE)  # 0 or 1
        out = subprocess.Popen("awk '/received/ {print $4}'", shell=True, stdin=out.stdout, stdout=subprocess.PIPE)
        if out.stdout.read().strip() == b'1':
            result.append(url)
    return result


# 按照核心数分配任务, 返回list of lists
def jobs(path):
    with open(path, mode='r') as f:
        urls = list(iter(f))
        job = len(urls) // cores
        jobs = []
        for i in range(cores):
            jobs.append(urls[i * job: (i+1) * job])
        return jobs


if __name__ == '__main__':
    main(parse.path)

