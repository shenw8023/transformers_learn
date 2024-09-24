
class Trainer:

    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    ):
        if args is None:
            output_dir = "tmp_trainer"
            logger.info(f"No `TrainingArguments` passed, using `output_dir={output_dir}`.")
            args = TrainingArguments(output_dir=output_dir)
        self.args = args

        self.create_accelerator_and_postprocess() #[ ]

        # set the correct log level depending on the node
        log_level = args.get_process_log_level()
        logging.set_verbosity(log_level)

        # force device and distributed setup init explicitly
        args._setup_devices #[ ]

        self.is_fsdp_xla_enabled = args.fsdp_config["xla"]
        if len(args.fsdp) > 0:
            if self.is_deepspeed_enabled:
                raise ValueError(
                    "Using --fsdp xxx together with --deepspeed is not possible, deactivate one of those flags."
                )
            if not args.fsdp_config["xla"] and args.parallel_mode != ParallelMode.DISTRIBUTED:
                raise ValueError("Using fsdp only works in distributed training.")
        

        default_collator = default_data_collator if tokenizer is None else DataCollatorWithPadding(tokenizer)
        self.data_collator = data_collator if data_collator is not None else default_collator
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer

        self.compute_metrics = compute_metrics
        self.preprocess_logits_for_metrics = preprocess_logits_for_metrics
        self.optimizer, self.lr_scheduler = optimizers

        if (self.is_deepspeed_enabled or self.is_fsdp_xla_enabled or self.is_fsdp_enabled) and (
            self.optimizer is not None or self.lr_scheduler is not None
        ):
            raise RuntimeError(
                "Passing `optimizers` is not allowed if Deepspeed or PyTorch FSDP is enabled. "
                "You should subclass `Trainer` and override the `create_optimizer_and_scheduler` method."
            )
        
        #[ ] 设置callback
        default_callbacks = DEFAULT_CALLBACKS + get_reporting_integration_callbacks(self.args.report_to)
        callbacks = default_callbacks if callbacks is None else default_callbacks + callbacks
        self.callback_handler = CallbackHandler(
            callbacks, self.model, self.tokenizer, self.optimizer, self.lr_scheduler
        )
        self.add_callback(PrinterCallback if self.args.disable_tqdm else DEFAULT_PROGRESS_CALLBACK)

        if self.args.should_save:
            os.makedirs(self.args.output_dir, exist_ok=True)


        #[ ] TrainerState, TrainerControl
        self.state = TrainerState(
            is_local_process_zero=self.is_local_process_zero(),
            is_world_process_zero=self.is_world_process_zero(),
        )
        self.control = TrainerControl()
        self.control = self.callback_handler.on_init_end(self.args, self.state, self.control)


    def compute_loss(self, model, inputs, return_outputs=False):
        """
        真实模型forward计算输出，然后结合labels计算loss        
        """

        labels = inputs.pop("labels")
        outputs = model(**inputs)
        if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
            loss = self.label_smoother(outputs, labels, shift_labels=True)
        else:
            loss = self.label_smoother(outputs, labels)
        return (loss, outputs) if return_outputs else loss


    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        创建optimizer和scheduler
        """
        self.create_optimizer()
        optimizer = self.optimizer
        self.create_scheduler(num_training_steps=num_training_steps, optimizer=optimizer)


    def create_optimizer(self):
        """
        根据TrainingArguments中传入的优化器相关参数来创建optimizer
        """
        if self.optimizer is None:
            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
        return self.optimizer


    def get_optimizer_cls_and_kwargs(args: TrainingArguments) -> Tuple[Any, Any]:
        """
        根据TrainingArguments中传入的参数 `optim`和`optim_args`解析确定Optimizer类和必要的参数
        """

        # parse args.optim_args
        optim_args = {}
        if args.optim_args:
            for mapping in args.optim_args.replace(" ", "").split(","):
                key, value = mapping.split("=")
                optim_args[key] = value

        optimizer_kwargs = {"lr": args.learning_rate}

        adam_kwargs = {
            "betas": (args.adam_beta1, args.adam_beta2),
            "eps": args.adam_epsilon,
        }
        if args.optim == OptimizerNames.ADAFACTOR:
            optimizer_cls = Adafactor
            optimizer_kwargs.update({"scale_parameter": False, "relative_step": False})
        elif args.optim == OptimizerNames.ADAMW_HF:
            
            ...
        return optimizer_cls, optimizer_kwargs
        
    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        """
        根据TrainingArguments中的 `lr_scheduler_type`参数创建scheduler
        包含可选的`get_warmup_steps`, `lr_scheduler_kwargs`
        """
        if self.lr_scheduler is None:
            self.lr_scheduler = get_scheduler(  #[ ] ge_scheduler函数写的很好
                self.args.lr_scheduler_type,
                optimizer=self.optimizer if optimizer is None else optimizer,
                num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                num_training_steps=num_training_steps,
                scheduler_specific_kwargs=self.args.lr_scheduler_kwargs,
            )
            self._created_lr_scheduler = True
        return self.lr_scheduler


    def get_train_dataloader(self) -> DataLoader:
        """
        根据self.train_dataset创建dataloader
        包括删除用不到的数据列
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training") #[ ]根据self.model的入参签名确定不需要的列
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training") #TODO

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params)) #[ ] 使用accelerator.prepare处理



    def _inner_training_loop(self, ):

        #[ ]创建TrainerState来记录训练过程中状态，记录训练到多少轮多少步了
        self.state = TrainerState()
        self.state.train_batch_size = self._train_batch_size
        # Compute absolute values for logging, eval, and save if given as ratio
        if args.logging_steps is not None:
            if args.logging_steps < 1:
                self.state.logging_steps = math.ceil(max_steps * args.logging_steps)
            else:
                self.state.logging_steps = args.logging_steps
        if args.eval_steps is not None:
            if args.eval_steps < 1:
                self.state.eval_steps = math.ceil(max_steps * args.eval_steps)
            else:
                self.state.eval_steps = args.eval_steps
        if args.save_steps is not None:
            if args.save_steps < 1:
                self.state.save_steps = math.ceil(max_steps * args.save_steps)
            else:
                self.state.save_steps = args.save_steps


        #[ ] Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples:,}")
        logger.info(f"  Num Epochs = {num_train_epochs:,}")
        logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}")
        if self.args.per_device_train_batch_size != self._train_batch_size:
            logger.info(f"  Training with DataParallel so batch size has been adjusted to: {self._train_batch_size:,}") #[ ]单卡batch_size
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps:,}")
        logger.info(f"  Number of trainable parameters = {get_model_param_count(model, trainable_only=True):,}")

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None


        # 如果是断点续训，加载上次结束时相关状态
        if resume_from_checkpoint is not None and os.path.isfile(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            epochs_trained = self.state.global_step // num_update_steps_per_epoch #[ ] global_step实际上是global_update_step
            if not args.ignore_data_skip: 
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps #[ ] 换算成要跳过多少个batch
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first"
                    f" {steps_trained_in_current_epoch} batches in the first epoch."
                )


        #[ ] 更新callback的指向
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        #[ ]防止继续训练中的相关参数被修改了，这里重新加载这几个参数，注意，只是部分
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        #[ ] 变量记录loss
        tr_loss = torch.tensor(0.0).to(args.device)#TODO为什么不用self记录
        self._total_loss_scalar = 0.0 
        self._globalstep_last_logged = self.state.global_step  #[ ] global_step是全局update的step数，这里用来记录上次log时的global_step
        model.zero_grad()


        self.control = self.callback_handler.on_train_begin(args, self.state, self.control) #[ ] callback.on_train_begin
        
        #[ ]执行epochs_trained轮，确保dataloader能执行到特定的random_state

        total_batched_samples = 0 #[ ]用来记录走了多少个batch，给gradient_accumulation_steps做判断用的
        for epoch in range(epochs_trained, num_train_epochs):
            epoch_iterator = train_dataloader

            steps_in_epoch = (  #[ ]train_dataloader总共包含的batch数量，也就是step数量
                len(epoch_iterator)
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control) #[ ] callback.on_epoch_begin

            steps_skipped = 0
            if steps_trained_in_current_epoch > 0:
                epoch_iterator = skip_first_batches(epoch_iterator, steps_trained_in_current_epoch)
                steps_skipped = steps_trained_in_current_epoch
                steps_trained_in_current_epoch = 0

            for step, inputs in enumerate(epoch_iterator): #[ ]这里内部有很多冗余判断
                total_batched_samples += 1

                if step % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(args, self.state, self.control) #[ ]callback.on_step_begin，指的是update_step
                
                #[ ] 应用梯度累积，执行一步loss计算，会进行loss.backward，但是优化器还不一定更新
                with self.accelerator.accumulate(model): 
                    tr_loss_step = self.training_step(model, inputs)
                
                #[ ] loss汇总：如果loss消失或爆炸： simply add the average of previous logged losses
                if (args.logging_nan_inf_filter and not is_torch_tpu_available() and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))):
                    tr_loss += tr_loss / (1 + self.state.global_step - self._globalstep_last_logged) #这里每次log后tr_loss会被置为0，所以要从上次log的step到现在看总共step了多少次
                else: #[ ] 梯度累积期间汇总loss
                    tr_loss += tr_loss_step

                #[ ]有可能存在step数量小于grad_acc数量的情况，此时也相当于要进行一次update了
                is_last_step_and_steps_less_than_grad_acc = (
                    steps_in_epoch <= args.gradient_accumulation_steps and (step + 1) == steps_in_epoch
                )
                
                #[ ]判断需要进行一次update_step的时候了
                if (
                    total_batched_samples % args.gradient_accumulation_steps == 0
                    or
                    is_last_step_and_steps_less_than_grad_acc # last step in epoch but step is always smaller than gradient_accumulation_steps
                ):
                    #[ ] 如果不够grad_acc数量，直接手动同步梯度
                    if is_last_step_and_steps_less_than_grad_acc:
                        self.accelerator.gradient_state._set_sync_gradients(True) 

                    #[ ] 梯度裁剪
                    if args.max_grad_norm is not None and args.max_grad_norm > 0:
                        self.accelerator.clip_grad_norm_(
                            model.parameters(),
                            args.max_grad_norm,
                        )

                    #[ ]优化器更新（多次backward已经在self.training_step进行过了）
                    self.optimizer.step()
                    optimizer_was_run = not self.accelerator.optimizer_step_was_skipped
                    if optimizer_was_run: #TODO
                        # Delay optimizer scheduling until metrics are generated
                        if not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                            self.lr_scheduler.step()  #[ ]scheduler更新

                    model.zero_grad()
                    self.state.global_step += 1 #[ ]更新state.global_step
                    self.state.epoch = epoch + (step + 1 + steps_skipped) / steps_in_epoch  #[ ]更新state.epoch
                    self.control = self.callback_handler.on_step_end(args, self.state, self.control) #[ ] callback.on_step_end

                    self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval) #TODO
                else:
                    #[ ]否则进行其他普通step的特别操作
                    self.control = self.callback_handler.on_substep_end(args, self.state, self.control) #[ ] 非update_step的其他step

                if self.control.should_epoch_stop or self.control.should_training_stop: #should_traing_stop有两个地方会发生: state.global_step >= state.max_steps 以及 early_stop判断的时候
                    break

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control) #[ ]callback.on_epoch_end
            self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)
            if self.control.should_training_stop:
                break
            
        #[ ]训练结束后根据情况是否要加载最优模型
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None: #[ ]在_save_checkpoint时会记录self.state.best_model_checkpoint
            # Wait for everyone to get here so we are sure the model has been saved by process 0.
            self._load_best_model()

        # add remaining tr_loss #TODO
        self._total_loss_scalar += tr_loss.item()
        train_loss = self._total_loss_scalar / self.state.global_step #[ ] 后面记录到metrics，这是平均update_step的loss
        
        metrics["train_loss"] = train_loss
        self.log(metrics)

        run_dir = self._get_output_dir(trial)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=run_dir)

        # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint and process allowed to save.
        if self.args.should_save and self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:
            for checkpoint in checkpoints_sorted:
                if not os.path.samefile(checkpoint, self.state.best_model_checkpoint):
                    logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
                    shutil.rmtree(checkpoint)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)
        return TrainOutput(self.state.global_step, train_loss, metrics)
    
    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):
        if self.control.should_log: #TODO 根据control相关属性的变化进行log，在基础流程DefaultFlowCallback中会在各个特殊节点调用callback时，对control变量相关属性进行修改，而在control的_new_step会重新设定should_log为False
            logs: Dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()
            # reset tr_loss to zero #TODO
            tr_loss -= tr_loss
            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4) #[ ]记录多个update_step之间的平均loss
            logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.log(logs)

        metrics = None
        if self.control.should_evaluate:
             metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)