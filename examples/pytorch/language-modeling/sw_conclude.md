

## 注意
- v4.36.0版本，在分布式训练的时候会出bug，trainer._save_checkpoint方法对文件多次修改，有问题，v4.36.1就修复了



## resume_from_checkpoint
- TrainingArguments包含该参数，但是不直接使用，是给trainer.train占位的
- Trainer.train(resume_from_checkpoint="")
    - resume_from_checkpoint (`str` or `bool`, *optional*):
        - If a `str`, local path to a saved checkpoint as saved by a previous instance of [`Trainer`]. If a `bool` and equals `True`, load the last checkpoint in *args.output_dir* as saved by a previous instance of [`Trainer`]. If present, training will resume from the model/optimizer/scheduler states loaded here.

    - 使用示例（参考run_clm.py）：
        ```python
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if training_args.do_train:
            checkpoint = None
            if training_args.resume_from_checkpoint is not None:
                checkpoint = training_args.resume_from_checkpoint
            elif last_checkpoint is not None:
                checkpoint = last_checkpoint
            train_result = trainer.train(resume_from_checkpoint=checkpoint)

        ```

## eval_steps
- 


## Trainer的参数梳理
### data_collator
    -  (DataCollator, optional) — The function to use to form a batch from a list of elements of train_dataset or eval_dataset. Will default to `default_data_collator()` if no tokenizer is provided, an instance of `DataCollatorWithPadding` otherwise.
    - 其中 `default_data_collator()` 是transformers库自己实现的最基本的collate方式，重点是输出了`labels`这个量；如果是使用`DataCollatorWithPadding`，就是调用tokenizer的pad方法进行padding后再打包，返回BatchEncoding对象。

- `transformers.default_data_collator`
    - (features: List[InputDataClass], return_tensors="pt") -> Dict[str, Any]:
    - 根据return_tensors类型不同，可以处理pt, tf, np类型
    - 主要处理dict-like objects，将features中的每个dict对象，按key组合在一起（value可以是tensor，ndarray，int），也就是会增加一个batch维度，形成一个batch的数据；输出一个dict，其中每个key是一个向量了
    - 如果有`label`这个键，打包成一维张量；如果有`label_ids`这个键，stack成二维张量。然后以`labels`作为key存到输出对象中
    - 等价于`class DefaultDataCollator`

- `class transformers.DataCollatorWithPadding`
    - Parameters
        - tokenizer (PreTrainedTokenizer or PreTrainedTokenizerFast) 
        - padding (bool, str or PaddingStrategy, optional, defaults to True)
            - True or 'longest' (default)
            - 'max_length'
            - False or 'do_not_pad'
        - max_length (int, optional) 
        - pad_to_multiple_of (int, optional)
        - return_tensors (str, optional, defaults to "pt")
    - 调用 self.tokenizer.pad()完成

- `PreTrainedTokenizerBase.pad()`
    - Pad a single encoded input or a batch of encoded inputs up to predefined length or to the max sequence length in the batch.
    ```python
    # If we have a list of dicts, let's convert it in a dict of lists
    if isinstance(encoded_inputs, (list, tuple)) and isinstance(encoded_inputs[0], Mapping):
        encoded_inputs = {key: [example[key] for example in encoded_inputs] for key in encoded_inputs[0].keys()}
    ```
    - 继续调用self._pad()完成


