

## 关于`args.max_train_steps` 和 `arg.num_warmup_steps`的定义理解
- 结论：
    - 这两个定义的命令行参数都是以`update_step`为单位来说的，更贴切的名字应该叫做`max_train_update_steps`和`num_warmup_update_steps`
    - **二者应该都是分布式单卡视角下。（也就是说是在dataloader被prepare均分到多卡后的情况）**
    - 实际初始化`lr_scheduler`时传的`num_warmup_steps`和`num_training_steps`会考虑gradient_accumulation，所以会更大一些。

- 代码
    ```python
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch #这句就证明arg.max_train_steps指的是update_steps
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * args.gradient_accumulation_steps, 
        #[ ] args.num_warmup_steps是按每个update为一个step单位来说的
        #[ ] get_scheduler这个方法的参数`num_warmup_steps`是按每batch_size/每一轮前向反向传播为一个单位step来说的
        #[ ] 至于gradient_accumulation这个操作，仅仅是控制每多少个step做一次参数更新，它不影响任何原来的定义。
        #[ ] 所以，最终的效果是实际num_warmup_steps的数量会比传参的args.num_warmup_steps要更大（因为考虑了accumulation进来）
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps, 
        #[ ]同理，参数定义中的args.max_train_steps指的也是update_step
    )
    ```



## accelerator对dataloader的影响
- 经过accelerator.prepare后的dataloader会在每张卡上时，总长度已经被卡数平分了。在每个进程上`len(dataloader)` 不等于原本的`len(dataloader)`。
- 代码：
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


## 关于get_scheduler处代码的疑惑解答
- 其中`num_training_steps=args.max_train_steps * args.gradient_accumulation_steps`
    - 这里所有的代码采用的单卡视角（当命令行没有指定max_train_steps，原文中计算这个量的时候用到了len(train_dataloader)，原文这里是在prepare前，所以说是单卡视角），原因是后面对lr_scheduler经过prepare后自会处理成分布式视角。
    - 经过prepare后的lr_scheduler会跟着分布式视角进行step，也就是在原来step的基础上每隔n个process才进行一次step。具体参考`prepare_scheduler`方法中的`AcceleratedScheduler`对象的运作逻辑：每个update_step，对应lr_scheduler内部进行n_process次的step。
    - 如果这里传的`num_training_steps`不是单卡视角，而已经是分布式视角，那么在创建`AcceleratedScheduler`的时候要指定split_batches=True）
    - 经过比对Trainer的代码，Trainer中这里的逻辑是：在创建lr_scheduler的时候用到的train_dataloader是经过prepare后的，所以`num_training_steps`是分布式视角，然后lr_scheduler就不进行prepare了，直接在合适时机进行`step()`

- [AcceleratedScheduler](https://hugging-face.cn/docs/accelerate/package_reference/torch_wrappers#accelerate.scheduler.AcceleratedScheduler)
- [AcceleratedScheduler源码](https://github.com/huggingface/accelerate/blob/main/src/accelerate/scheduler.py#L25)


## 关于args.max_train_steps和arg.num_train_epochs 二者同时指定的问题
- 如果同时指定了二者，会以`max_train_steps`为准，指定的`num_train_epochs`就不起作用了
- max_train_steps, If provided, overrides num_train_epochs.
- 代码实现：
    ```py
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps) #[ ]此时经过prepare后的train_dataloader 长度已经被卡数给平分了
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    # 如果max_train_steps为空，就会使用num_train_epochs来决定max_train_steps；如果不为空，就会用max_train_steps来决定num_train_epochs
    ```



## 关于记录到tensorboard
- [accelerate文档](https://hugging-face.cn/docs/accelerate/usage_guides/explore)有更详细的使用说明
- logger.info和accelerator.log的区别 #TODO
- 代码
    ```py
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers("clm_no_trainer", experiment_config)

    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        if args.with_tracking:
            total_loss = 0
        
        for step, batch in enumerate(active_dataloader):
            if args.with_tracking:
                total_loss += loss.detach().float()

        if args.with_tracking:
            accelerator.log(
                {
                    "perplexity": perplexity,
                    "eval_loss": eval_loss,
                    "train_loss": total_loss.item() / len(train_dataloader),
                    "epoch": epoch,
                    "step": completed_steps,
                },
                step=completed_steps,
            )
    if args.with_tracking:
        accelerator.end_training()
    ```


## 关于tqdm使用

- 代码
    ```py
    from tqdm.auto import tqdm
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process) #非本地主进程的都disable了
    completed_steps = 0
    progress_bar.update(completed_steps)

    for epoch in range(starting_epoch, args.num_train_epochs):
        for step, batch in enumerate(active_dataloader):
            progress_bar.update(1)

    ```



## 关于resume_from_checkpoint

```py
if args.resume_from_checkpoint:
    checkpoint_path = args.resume_from_checkpoint
    path = os.path.basename(args.resume_from_checkpoint) #文件夹名

    accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
    accelerator.load_state(checkpoint_path) #[ ]加载accelerator相关state，注意是在prepare之后，也就是所有对象都初始化好了以后
    # Extract `epoch_{i}` or `step_{i}`
    training_difference = os.path.splitext(path)[0] #分离文件名和扩展名

    if "epoch" in training_difference:
        starting_epoch = int(training_difference.replace("epoch_", "")) + 1 #注意+1从下一个epoch开始
        resume_step = None
        completed_steps = starting_epoch * num_update_steps_per_epoch #说明epoch是从0开始计数的
    else:
        # need to multiply `gradient_accumulation_steps` to reflect real steps #[ ]换算成real_ste的视角更好理解
        resume_step = int(training_difference.replace("step_", "")) * args.gradient_accumulation_steps
        starting_epoch = resume_step // len(train_dataloader)
        completed_steps = resume_step // args.gradient_accumulation_steps  #completed_update_steps
        resume_step -= starting_epoch * len(train_dataloader)  #这就不是update_step了  #[ ]这个量用于后面跳过最后一个epoch中已经完成的部分step数据

# update the progress_bar if load from checkpoint
progress_bar.update(completed_steps) #[ ]进度条也要跳过已经完成的step

```




## 关于checkpoint保存和模型保存
- 状态保存：
    - `accelerator.save_state(output_dir)`
    - **保存了模型权重，优化器状态，随机数生成器，scheduler**
    - #TODO 代码中这类保存并没有wait_for_everyone，是不是分布式每个进程都保存各自的内容
- 模型保存：
    - #TODO [官方文档](https://huggingface.co/docs/accelerate/v1.0.1/en/basic_tutorials/migration?save=sharded+checkpoint#save-and-load)
    - `accelerator.save_model(model, save_directory)`
    - `unwrapped_model.save_pretrained(args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save)`
    - 可以根据`checkpointing_steps`这个参数决定是每多少个`update_step`进行保存，还是在每个epoch后保存。

- 在每多少个step后保存一次checkpoint：
    ```py
    for epoch in range(starting_epoch, args.num_train_epochs):
        for step, batch in enumerate(active_dataloader):
            ...

            if accelerator.sync_gradients: #[ ] 这里是为了考虑gradient_accumulation，必须经过这里的判断，才能让completed_steps代表update_steps
                progress_bar.update(1)
                completed_steps += 1

            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0:
                    output_dir = f"step_{completed_steps}"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir) #[ ] 如果checkpointing_steps是数字表示按step保存
            if completed_steps >= args.max_train_steps:
                break
    ```

- 在每个epoch后保存一次checkpoint：
    ```py
    if args.checkpointing_steps == "epoch":
        output_dir = f"epoch_{epoch}"
        if args.output_dir is not None:
            output_dir = os.path.join(args.output_dir, output_dir)
        accelerator.save_state(output_dir)
    ```

- 在训练结束最后保存一次model：
    ```py
    accelerator.wait_for_everyone() #保存模型之前需要等待同步
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained( #TODO is_main_process参数的作用
        args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
    ) #等价于`accelerator.save(unwrapped_model.state_dict(), args.output_dir)`
    if accelerator.is_main_process: #只在主进程保存一次
        tokenizer.save_pretrained(args.output_dir)
    ```


## 关于DataLoader的collate_fn
#TODO




## 其他
- 要注意的是：如果你的模型有 tied weight （比如语言模型将 embedding matrix 的权重与 decoder 的权重绑定），将这个模型移动到 TPU （无论是你自己移动、还是由 prepare() 移动）会破坏绑定。你将需要在之后重新绑定权重。

- [accelerate用法](https://www.huaxiaozhuan.com/%E5%B7%A5%E5%85%B7/huggingface_transformer/chapters/7_accelerate.html)
- [datasets用法](https://www.huaxiaozhuan.com/%E5%B7%A5%E5%85%B7/huggingface_transformer/chapters/2_datasets.html)

- [Accelerate文档](https://hugging-face.cn/docs/accelerate/package_reference/torch_wrappers)还值得再梳理一遍