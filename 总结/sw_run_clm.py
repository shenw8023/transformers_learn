import os


"""
#TODO 
- load_dataset
- raw_dataset.map

"""



@dataclass
class ModelArguments:
    model_name_or_path:
    model_type: #用于train from scratch
    config_overrides: #用于从头训练时，修改模型默认设置，例如：n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index
    
    config_name: #Pretrained config name or path if not the same as model_name
    tokenizer_name: #这两个参数用于单独指定这两个对象，而不是跟着模型name走

    cache_dir: #模型缓存路径
    use_fast_tokenizer:
    token: #模型文件访问令牌
    trust_remote_code:
    torch_dtype: #修改模型加载格式 "choices": ["auto", "bfloat16", "float16", "float32"]
    low_cpu_mem_usage:


@dataclass
class DataTrainingArguments:
    dataset_name: #用于下载公共数据集
    dataset_config_name: #[ ]

    train_file: #仅限于以下格式：["csv", "json", "txt"]
    validation_file:

    max_train_samples: #用于debug或快速验证，会截断训练数据量为这个值
    max_eval_samples:
    streaming: #[ ]  流式下载数据集，相应也会影响后续的数据批量处理函数
    block_size: #样本切分长度单位
    overwrite_cache: #Overwrite the cached training and evaluation sets
    validation_split_percentage: #当没有验证集的时候，会从训练集取这个比例的数据作为验证集
    preprocessing_num_workers: #[ ]预处理的进程数量
    keep_linebreaks: #处理txt数据的时候，是否保留换行符



def main():
    #===========参数解析===============
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1])) #[ ]
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model_args.token = model_args.use_auth_token


    #===========logging设置===============



    # Set seed before initializing model.
    set_seed(training_args.seed)



    #===========数据集加载===============
        # 可以提供自己的CSV/JSON/TXT文件，或者是公共数据集的name
        # 对于CSV/JSON，本脚本会使用文件中的'text'列 or the first column
        # 分布式训练中 `load_dataset` function guarantee that only one local process can concurrently download the dataset.
        # TODO 参考 https://huggingface.co/docs/datasets/loading_datasets.

    # 如果设定了数据集名，直接加载，然后看是否包含验证集，如果没有，则重新加载训练集，并且切分为训练和验证
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir, #数据集合模型缓存在一起
            token=model_args.token,
            streaming=data_args.streaming, #[ ]
        )
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
                token=model_args.token,
                streaming=data_args.streaming,
            )
            raw_datasets["train"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
                token=model_args.token,
                streaming=data_args.streaming,
            )
    else:
        # 事先不知道用户可能传了哪些文件，所以先收集data_files（包括train_file和validation_file），然后读取所有传入的文件；此时再检查是否包含验证集，如果不包含，则说明用户没有传validation_file，所以重新加载train_file，部分作为训练集，部分作为验证集。
        # TODO可不可以直接根据用户是否传validation_file决定采用哪种加载方式？如果有val_file，我就加载获得训练集和验证集，如果没有val_file，我就直接把训练集切分加载。

        data_files = {}
        dataset_args = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
        extension = (
            data_args.train_file.split(".")[-1]
            if data_args.train_file is not None
            else data_args.validation_file.split(".")[-1]
        )
        if extension == "txt":
            extension = "text"
            dataset_args["keep_linebreaks"] = data_args.keep_linebreaks
        raw_datasets = load_dataset( #TODO
            extension,
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
            **dataset_args,
        )
        # If no validation data is there, validation_split_percentage will be used to divide the dataset.
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
                token=model_args.token,
                **dataset_args,
            )
            raw_datasets["train"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
                token=model_args.token,
                **dataset_args,
            )



    #===========加载Config对象===============
        # [ ] Config加载逻辑：1.根据有没有明确的config_name；2.根据model_name_or_path加载指定预训练模型；3.根据model_type从头训练
        # Distributed training: The .from_pretrained methods guarantee that only one local process can concurrently download model & vocab.
        
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "token": model_args.token,
        "trust_remote_code": model_args.trust_remote_code,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)
            logger.info(f"New config: {config}")

    

    #===========加载Tokenizer对象===============
        # 逻辑跟Config加载类似，但是tokenizer不支持从头训练，必须有所指定
    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "token": model_args.token,
        "trust_remote_code": model_args.trust_remote_code,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script. "
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    

    #===========加载Model对象===============
        # [ ] model加载逻辑：1.根据model_name_or_path加载预训练模型；2.从头训练，根据上面加载的config对象
    if model_args.model_name_or_path:
        torch_dtype = (
            model_args.torch_dtype
            if model_args.torch_dtype in ["auto", None]
            else getattr(torch, model_args.torch_dtype)
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=model_args.low_cpu_mem_usage,
        )
    else:
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=model_args.trust_remote_code)  #实例化config的第一个用处
        n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
        logger.info(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")


    # [ ] 如果从头训练，确保vocab和embedding维度一致
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))




    #===========数据处理===============
        # 先tokenize文本
        # 然后拼接为长文本，再按照block_size切分为chunks
        # 准备训练集
    
    #[ ]确定训练数据所在的列名
    if training_args.do_train:
        column_names = list(raw_datasets["train"].features)
    else:
        column_names = list(raw_datasets["validation"].features)
    text_column_name = "text" if "text" in column_names else column_names[0]

    #[ ]tokenize函数
    def tokenize_function(examples):
        output = tokenizer(examples[text_column_name])
        return output

    #[ ] 数据集做tokenize，注意只在主进程处理
    with training_args.main_process_first(desc="dataset map tokenization"):
        if not data_args.streaming:
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True, #[ ] 默认一次处理1000条
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names, #[ ]数据可以删了，只需要保留分词后的id序列就够了
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on dataset",
            )
        else:
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                remove_columns=column_names,
            )
    
    #[ ] 确定max_pos_embeddings
    if hasattr(config, "max_position_embeddings"): #实例化config的第二个用处
        max_pos_embeddings = config.max_position_embeddings
    else:
        # Define a default value if the attribute is missing in the config.
        max_pos_embeddings = 1024

    #[ ] 进而确定 block_size取值
    if data_args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > max_pos_embeddings:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                f"Using block_size={min(1024, max_pos_embeddings)} instead. You can change that default value by passing --block_size xxx."
            )
            block_size = min(1024, max_pos_embeddings) #[ ]block_size 不能超过max_pos_embedding
    else:
        if data_args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({data_args.block_size}) is larger than the maximum length for the model "
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(data_args.block_size, tokenizer.model_max_length)


    #[ ] 数据拼接再切分的处理方式
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()} #[ ] 每个key对应的 list[list] 拼接为一个长的list
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
        # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy() #[ ] 创建clm的labels
        return result


    #[ ] 拼接所有样例，再按照block_size切分为chunks（可能预训练可以这么做，但是sft不应该）
    with training_args.main_process_first(desc="grouping texts together"):
        if not data_args.streaming:
            lm_datasets = tokenized_datasets.map(
                group_texts,
                batched=True, #[ ]默认每1000条做一次处理，1000又会被block_size整除，余数部分会被丢掉
                num_proc=data_args.preprocessing_num_workers, #[ ]非流式时，多进程处理
                load_from_cache_file=not data_args.overwrite_cache,
                desc=f"Grouping texts in chunks of {block_size}",
            )
        else:
            lm_datasets = tokenized_datasets.map(
                group_texts,
                batched=True,
            )

    #[ ] 准备好最终的训练集（按要求做采样，用于debug）
    if training_args.do_train:
        if "train" not in tokenized_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = lm_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))  #[ ] Dataset对象支持采样

    #[ ] 准备好最终的验证集，包括logits后处理函数，metrics计算函数
    if training_args.do_eval:
        if "validation" not in tokenized_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = lm_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

        def preprocess_logits_for_metrics(logits, labels):
            if isinstance(logits, tuple):
                # Depending on the model and config, logits may contain extra tensors,
                # like past_key_values, but logits always come first
                logits = logits[0]
            return logits.argmax(dim=-1)

        metric = evaluate.load("accuracy.py")

        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            # we need to shift the labels
            labels = labels[:, 1:].reshape(-1) #上面的labels是直接从input_ids做copy的，所以真正的labels要往后移一位
            preds = preds[:, :-1].reshape(-1)
            return metric.compute(predictions=preds, references=labels)

            """ #[ ]理解shift原理
            input_ids: 0,1,2,3,4
            labels: 0,1,2,3,4    ——> 1,2,3,4
            preds:  1,2,3,4,eos  ——> 1,2,3,4
            """


    #===========确定last_checkpoint===============
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir: #不覆盖，并且有output_dir
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        # 如果output_dir不为空，并且没检测到last_checkpoint，则报异常，建议设定--overwrite_output_dir
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        # 如果有last_checkpoint，并且没有设定断点续训，则提醒一下会执行断点续训
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )



    #===========开始训练和验证===============
    #[ ]实例化Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        # Data collator will default to DataCollatorWithPadding, so we change it.
        data_collator=default_data_collator, #[ ]
        compute_metrics=compute_metrics if training_args.do_eval and not is_torch_tpu_available() else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
        if training_args.do_eval and not is_torch_tpu_available()
        else None,
    )

    #[ ]Training
    if training_args.do_train:
        checkpoint = None
        #[ ]1.看是否设置了resume_from_checkpoint，2.看是否有last_checkpoint，3.否则为None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint) #TODO返回内容
        trainer.save_model()  #TODO saves only the tokenizer with the model

        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state() #TODO

    #[ ]Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    
