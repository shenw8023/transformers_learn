
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