# 一. 参考资料

[链接](https://www.biaodianfu.com/python-multi-thread-and-multi-process.html)

# 二. 常用代码 

## 1. 获得当前进程ID

```python
import os
os.getpid()
```

## 2. 普通多线程

```python
from threading import Thread    

def test(name):
    print(name)

if __name__ == '__main__':
    t1 = Thread(target=test, args=('thread1',))
    t2 = Thread(target=test, args=('thread2',))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    print('main')
```

## 3. 普通线程池

python由于GIL的存在, 多线程适用于IO操作比较多的函数

```python
import os
import concurrent.futures
import threading
import time

def process_image(image_path):
    file_name = os.path.basename(image_path)
    thread_id = threading.current_thread().ident
    print(f"Processing image: {file_name} in Thread ID: {thread_id}")
    time.sleep(0.1)

def process_images_in_parallel(image_paths, num_threads=4):
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Submit each image processing task to the thread pool
        executor.map(process_image, image_paths)

if __name__ == "__main__":
    image_paths = [f"image_{i}" for i in range(10000)]
    num_threads = 4
    process_images_in_parallel(image_paths, num_threads)
```

## 3. 普通多进程 

```python
from multiprocessing import Process
import os

def run_proc(name):
    print('Run child process %s (%s)...' % (name, os.getpid()))

if __name__=='__main__':
    print('Parent process %s.' % os.getpid())
    p = Process(target=run_proc, args=('test',))
    print('Child process will start.')
    p.start()
    p.join()
print('Child process end.')
```

## 4. 普通进程池 

```python
from multiprocessing import Pool

def test(i):
    print(i)

if __name__ == "__main__":
    lists = range(100)
    pool = Pool(8)
    print("多进程开始执行")
    pool.map(test, lists)
    pool.close()
    pool.join()
```

## 5. 向进程池中添加任务(全部添加结束后执行) 

向进程池中添加任务, 所有任务全部添加完才执行, 进程池中进程数为8, 也就是开始执行后每次最多同时执行8个任务

```python
from multiprocessing import Pool

def test(i):
    print(i)

if __name__ == "__main__":
    pool = Pool(8)
    for i in range(100):
        '''
        For循环中执行步骤：
        （1）循环遍历，将100个子进程添加到进程池（相对父进程会阻塞）
        （2）每次执行8个子进程，等一个子进程执行完后，立马启动新的子进程。（相对父进程不阻塞）
        apply_async为异步进程池写法。异步指的是启动子进程的过程，与父进程本身的执行（print）是异步的，而For循环中往进程池添加子进程的过程，与父进程本身的执行却是同步的。
        '''
        # 只是添加进程并不执行
        pool.apply_async(test, args=(i,))  # 维持执行的进程总数为8，当一个进程执行完后启动一个新进程.
    # 关闭进程池
    pool.close()
    print("多进程开始执行")
    # 等待子进程结束后再继续往下运行，通常用于进程间的同步
    pool.join()
    print("多进程结束执行")
```

## 6. 向进程池中添加任务(边添加边执行) 

向进程池中添加任务, 只要有任务就执行, 进程池中进程数为8, 也就是最多同时执行8个任务

```python
from multiprocessing import Pool

def test(i):
    print(i)

if __name__ == "__main__":
    pool = Pool(8)
    for i in range(100):
        '''
            实际测试发现，for循环内部执行步骤：
            （1）遍历100个可迭代对象，往进程池放一个子进程
            （2）执行这个子进程，等子进程执行完毕，再往进程池放一个子进程，再执行。（同时只执行一个子进程）
            for循环执行完毕，再执行print函数。
        '''
        pool.apply(test, args=(i,))  # 维持执行的进程总数为8，当一个进程执行完后启动一个新进程.
    print("多进程结束执行")
    pool.close()
    pool.join()
```

## 7. JoinableQueue实现多进程之间的通信

多进程间的通信(JoinableQueue)
task_done()：消费者使用此方法发出信号，表示q.get()的返回项目已经被处理。如果调用此方法的次数大于从队列中删除项目的数量，将引发ValueError异常
join():生产者调用此方法进行阻塞，直到队列中所有的项目均被处理。阻塞将持续到队列中的每个项目均调用q.task_done（）方法为止

```python
from multiprocessing import Process, JoinableQueue
import time, random

def consumer(q):
    while True:
        res = q.get()
        print('消费者拿到了 %s' % res)
        q.task_done()

def producer(seq, q):
    for item in seq:
        time.sleep(random.randrange(1,2))
        q.put(item)
        print('生产者做好了 %s' % item)
    q.join()

if __name__ == "__main__":
    q = JoinableQueue()
    seq = ('产品%s' % i for i in range(5))
    p = Process(target=consumer, args=(q,))
    p.daemon = True  # 设置为守护进程，在主线程停止时p也停止，但是不用担心，producer内调用q.join保证了consumer已经处理完队列中的所有元素
    p.start()
    producer(seq, q)
    print('主线程')
```

## 8. 进程间数据共享(不常用)

进程间的数据共享(multiprocessing.Queue)(基本不用, 因为进程间本来就是资源独立的)

```python
from multiprocessing import Process, Queue
import os, time, random

def write(q):
    print('Process to write: %s' % os.getpid())
    for value in ['A', 'B', 'C']:
        print('Put %s to queue...' % value)
        q.put(value)
        time.sleep(random.random())

def read(q):
    print('Process to read: %s' % os.getpid())
    while True:
        value = q.get(True)
        print('Get %s from queue.' % value)

if __name__ == "__main__":
    q = Queue()
    pw = Process(target=write, args=(q,))
    pr = Process(target=read, args=(q,))
    pw.start()
    pr.start()
    pw.join()  # 等待pw结束
    pr.terminate()  # pr进程里是死循环，无法等待其结束，只能强行终止
```

