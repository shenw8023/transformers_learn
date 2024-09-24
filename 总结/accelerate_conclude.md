
## 让主进程先操作，然后其他进程再操作

- `with accelerator.main_process_first()` 确保主进程先执行，其他进程等待主进程执行完再执行，经过下面代码验证，其他进程也是做了with下的完整操作，而不是直接拷贝主进程的执行结果。
    ```py
    from accelerate import Accelerator

    accelerator = Accelerator()

    def func_process():
        print(f"func call in process: {accelerator.process_index}")
        return 10

    with accelerator.main_process_first():
        res = func_process()


    """
    func call in process: 0
    func call in process: 1
    """
    ```

- `run_clm_no_trainer.py`中使用该方法做数据处理，是让主进程处理以后，其他进程可以直接使用缓存从而避免了重复处理。实现的关键应该是`load_from_cache_file=not args.overwrite_cache`
    - 但是这里有一个问题：1.如果设置为使用缓存，那么第一次执行时，主进程会做处理，随后其他进程按设定会使用缓存，符合预期；可是如果重新训练的时候，就要求手动清除缓存，否则主进程就会错误使用应该丢弃的历史缓存，不符合预期，容易产生bug。2.如果设置为不使用缓存，那么每次主进程都会重新处理，符合预期，没有bug，但是其他进程也都会重新处理，就没法利用缓存加速了，也不太合适。
    ```py
    with accelerator.main_process_first():
        lm_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            load_from_cache_file=not args.overwrite_cache,
            desc=f"Grouping texts in chunks of {block_size}",
        )
    ```


## 等待进程都到同一个点
- `accelerator.wait_for_everyone()` 等待所有进程都到达同一个点，再继续执行。



## accelerator对dataloader的影响
- 经过accelerator.prepare后的dataloader会在每张卡上时，总长度已经被卡数平分了。在每个进程上`len(dataloader)` 不等于原本的`len(dataloader)`。
```py
from accelerate import Accelerator
from torch.utils.data import DataLoader

accelerator = Accelerator()
dataloader = DataLoader(range(10), batch_size=4)
dataloader = accelerator.prepare(dataloader)
for batch in dataloader:
    print(batch)
    gathered_items = accelerator.gather_for_metrics(batch)
    print(str(len(gathered_items))+"\n")

"""
每个进程独立打印出自己拿到的数据，没有先后顺序；
数据获取逻辑：
    在每个step，分别按batch_size取数据分配给每个进程，当数据量不能被batch_size整除的时候，最后一批的数据量不够均分到多个进程，此时（如果无shuffle的话）就会从头开始取数据进行补全，并且最终要保证每个进程分到的批量一致；
    例如：DataLoader(range(10), batch_size=4)，两张卡：
        step1:
            tensor([0, 1, 2, 3], device='cuda:0')
            tensor([4, 5, 6, 7], device='cuda:1')
            此时如果gather_for_metrics，每个进程都会获取总共8条数据，因为没有duplicate

        step2:
            tensor([8, 9, 0, 1], device='cuda:0')
            tensor([2, 3, 4, 5], device='cuda:1')
            此时如果gather_for_metrics，每个进程都会获取总共2条数据，因为0卡上的[0,1]和1卡上的[2,3,4,5]都是补全的duplicate数据，该函数能自动去重，此时两个卡上获取的gather_for_metrics都是[8,9]

    例如：DataLoader(range(6), batch_size=4)，两张卡：
        step1:
            tensor([0, 1, 2, 3], device='cuda:0')
            tensor([4, 5, 0, 1], device='cuda:1')
            此时如果gather_for_metrics，每个进程都会获取总共6条数据，因为1卡上的[0,1]都是补全的重复数据，该函数能自动去重。

- gather_for_metrics()
    - 在每个step调用这个函数的时候，会将所有进程的数据都收集过来，并且自动丢弃补全的数据，然后返回此时的所有数据。
    - 实际上该函数会在每个进程上都执行一次
"""
```