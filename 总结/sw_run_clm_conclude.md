

- 只需要主进程执行一次的任务
`if accelerator.is_main_process: ...`

- In distributed training, the `load_dataset` function guarantee that only one local process can concurrently download the dataset.

- [datasets加载本地数据](https://huggingface.co/docs/datasets/loading_datasets)

```py
    # Load a txt file
    from datasets import load_dataset
    data_files = {"train":"train_data.txt", "validation":"val_data.txt"}
    raw_datasets = load_dataset("text", data_files=data_files, **dataset_args)

    # Load a CSV file
    >>> from datasets import load_dataset
    >>> ds = load_dataset('csv', data_files='path/to/local/my_dataset.csv')

    # Load a JSON file
    >>> from datasets import load_dataset
    >>> ds = load_dataset('json', data_files='path/to/local/my_dataset.json')

    # Load from a local loading script
    >>> from datasets import load_dataset
    >>> ds = load_dataset('path/to/local/loading_script/loading_script.py', split='train')
```

- In distributed training, the `.from_pretrained` methods guarantee that only one local process can concurrently download model & vocab.

- 多进程相同的print操作，以下方式确保只会打印一次
`accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")`


- accelerator.load_state(checkpoint_path)
    - Loads the current states of the model, optimizer, scaler, RNG generators, and registered objects.





## accelerate deepspeed Integration 
- https://huggingface.co/docs/accelerate/v0.27.2/en/usage_guides/deepspeed#accelerate-deepspeed-plugin

- 两种集成方式
    1. `accelerate config`配置的时候指定一个deepspeed_config.json文件，包含deepspeed库相关的细节配置。
    2. `accelerate config`配置的时候不指定配置文件，而是通过几个主要命令行参数指定，其他的使用默认参数。
        - 当前支持的命令行参数（都是通过`accelerate config`回答问题来自动设置的）
            - `zero_stage`
            - `gradient_accumulation_steps`
            - `gradient_clipping`
            - `offload_optimizer_device`
            - `offload_param_device`
            - `zero3_init_flag`
            - `zero3_save_16bit_model`
            - `mixed_precision`