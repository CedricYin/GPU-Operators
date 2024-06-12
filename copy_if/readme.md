## 性能分析 (V100 GPU)
1. copy_if 0: 0.816320 ms。naive版本，性能瓶颈在于原子函数的同步开销。
2. copy_if 1: 0.295232 ms。细粒度化同步，使得降低同步开销。具体而言，就是将global的原子加改成每个block的shared memory的原子加，然后再选一个leader进行global的原子加。

## ToDo
使用compute-sanitizer时，如果只启动一次内核，没有任何异常，但如果启动多次内核，那么除了第一次的核函数以外，剩余的迭代中核函数会爆写越界，目前原因不明。