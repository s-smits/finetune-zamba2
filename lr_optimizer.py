import optuna
import numpy as np
import torch
from datasets import load_dataset, Dataset, DatasetDict
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TrainerCallback,
    DataCollatorForLanguageModeling
)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
from scipy.stats import norm
import matplotlib.pyplot as plt
import json
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

class DatasetHandler:
    def __init__(
        self,
        dataset_name=None,
        model_name=None,
        dataset=None,
        tokenizer=None,
        context_window=2048,
        max_samples=None
    ):
        self.context_window = context_window
        self.max_samples = max_samples
        
        # Initialize tokenizer
        self.tokenizer = tokenizer if tokenizer else AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        
        # Initialize dataset
        self.dataset = self._initialize_dataset(dataset, dataset_name)
        
    def _initialize_dataset(self, dataset, dataset_name):
        """Initialize and prepare dataset"""
        if dataset is not None:
            if not isinstance(dataset, (Dataset, DatasetDict)):
                raise ValueError("Dataset must be a HuggingFace Dataset or DatasetDict")
            raw_dataset = dataset
        elif dataset_name is not None:
            raw_dataset = load_dataset(dataset_name)
        else:
            raise ValueError("Either dataset or dataset_name must be provided")
        
        # Handle DatasetDict
        if isinstance(raw_dataset, DatasetDict):
            if 'train' in raw_dataset:
                raw_dataset = raw_dataset['train']
            else:
                raw_dataset = list(raw_dataset.values())[0]
        
        # Limit dataset size if specified
        if self.max_samples and len(raw_dataset) > self.max_samples:
            raw_dataset = raw_dataset.select(range(self.max_samples))
        
        return raw_dataset
    
    def prepare_chat_format(self, examples):
        """Prepare chat format for training"""
        if 'messages' in examples:  # Chat format
            chats = []
            for messages in examples['messages']:
                try:
                    chat = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=True,
                        max_length=self.context_window,
                        truncation=True,
                        return_tensors=None
                    )
                    chats.append(chat)
                except Exception as e:
                    print(f"Error applying chat template: {e}")
                    # Fallback format
                    text = ""
                    for message in messages:
                        role = message["role"]
                        content = message["content"]
                        text += f"<|{role}|>\n{content}</s>\n"
                    
                    chat = self.tokenizer(
                        text,
                        max_length=self.context_window,
                        truncation=True,
                        return_tensors=None
                    )["input_ids"]
                    chats.append(chat)
            return {"input_ids": chats}
        else:  # Regular text format
            return self.tokenizer(
                examples['text'] if 'text' in examples else examples[examples.column_names[0]],
                max_length=self.context_window,
                truncation=True
            )
    
    def get_tokenized_dataset(self):
        """Get tokenized dataset ready for training"""
        return self.dataset.map(
            self.prepare_chat_format,
            batched=True,
            remove_columns=self.dataset.column_names
        )

class LROptimizerCallback(TrainerCallback):
    def __init__(
        self,
        num_trials=10,
        lr_range=(1e-6, 1e-4),
        study_name="learning_rate_optimization",
        save_path="lr_optimization_results.json",
        optimization_steps=100,
        warmup_steps=10
    ):
        self.num_trials = num_trials
        self.lr_range = lr_range
        self.study_name = study_name
        self.save_path = save_path
        self.optimization_steps = optimization_steps
        self.warmup_steps = warmup_steps
        
        # Initialize study
        self.study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=42),
            study_name=study_name
        )
        
        # State tracking
        self.current_trial = 0
        self.current_step = 0
        self.current_losses = []
        self.best_lr = None
        self.best_loss = float('inf')
        self.trial_results = []
        
    def on_train_begin(self, args, state, control, **kwargs):
        """Initialize first trial"""
        self.start_new_trial(args)

    def on_step_end(self, args, state, control, **kwargs):
        """Monitor training progress and switch learning rates"""
        self.current_step += 1
        
        # Check if we should move to next trial
        if self.current_step >= self.optimization_steps and self.current_trial < self.num_trials:
            # Calculate mean loss for current trial
            mean_loss = np.mean(self.current_losses[-self.optimization_steps:])
            self.trial_results.append((args.learning_rate, mean_loss))
            
            # Update best if needed
            if mean_loss < self.best_loss:
                self.best_lr = args.learning_rate
                self.best_loss = mean_loss
            
            # Start new trial if not finished
            self.current_trial += 1
            if self.current_trial < self.num_trials:
                self.start_new_trial(args)
                # Reset step counter
                self.current_step = 0
            else:
                # Finalize optimization
                self.finalize_optimization(args)
                # Set best learning rate
                args.learning_rate = self.best_lr

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Track losses"""
        if logs and "loss" in logs:
            self.current_losses.append(logs["loss"])

    def start_new_trial(self, args):
        """Start a new optimization trial"""
        trial = self.study.ask()
        new_lr = trial.suggest_float("learning_rate", self.lr_range[0], self.lr_range[1], log=True)
        args.learning_rate = new_lr
        print(f"\nStarting trial {self.current_trial + 1}/{self.num_trials} with lr={new_lr:.2e}")

    def optimize_final_lr(self):
        """Run GPR optimization on collected results"""
        try:
            # Extract learning rates and losses
            X = np.array([[lr] for lr, _ in self.trial_results])
            y = np.array([loss for _, loss in self.trial_results])
            
            # Check if we have enough valid results
            valid_mask = np.isfinite(y)
            if not np.any(valid_mask):
                return self._get_default_optimization()
            
            # Filter out infinite values
            X = X[valid_mask]
            y = y[valid_mask]
            
            if len(X) < 2:
                return self._get_default_optimization()
            
            # Transform to log space
            X_log = np.log10(X)
            
            # Normalize y values
            y_mean, y_std = np.mean(y), np.std(y)
            y_std = 1 if y_std == 0 else y_std
            y_normalized = (y - y_mean) / y_std
            
            # Fit GPR
            kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
            gpr = GaussianProcessRegressor(
                kernel=kernel,
                n_restarts_optimizer=10,
                random_state=42,
                normalize_y=False
            )
            
            gpr.fit(X_log, y_normalized)
            
            # Create prediction grid
            X_pred_log = np.linspace(np.log10(X.min()), np.log10(X.max()), 1000).reshape(-1, 1)
            y_pred_normalized, sigma = gpr.predict(X_pred_log, return_std=True)
            
            # Denormalize predictions
            y_pred = y_pred_normalized * y_std + y_mean
            sigma = sigma * y_std
            
            # Find optimal points
            best_idx = np.argmin(y_pred)
            optimal_lr = 10 ** X_pred_log[best_idx, 0]
            
            # Calculate Expected Improvement
            best_f = np.min(y)
            Z = (best_f - y_pred) / (sigma + 1e-9)
            ei = sigma * (Z * norm.cdf(Z) + norm.pdf(Z))
            ei_best_idx = np.argmax(ei)
            ei_optimal_lr = 10 ** X_pred_log[ei_best_idx, 0]
            
            return {
                'gpr_optimal_lr': float(optimal_lr),
                'ei_optimal_lr': float(ei_optimal_lr),
                'predicted_loss': float(y_pred[best_idx]),
                'uncertainty': float(sigma[best_idx])
            }
            
        except Exception as e:
            print(f"GPR optimization failed: {e}")
            return self._get_default_optimization()

    def _get_default_optimization(self):
        """Return default values if optimization fails"""
        return {
            'gpr_optimal_lr': 2e-5,
            'ei_optimal_lr': 2e-5,
            'predicted_loss': float('inf'),
            'uncertainty': float('inf')
        }

    def finalize_optimization(self, args):
        """Run final optimization and save results"""
        final_optimization = self.optimize_final_lr()
        
        results = {
            "best_learning_rate": self.best_lr,
            "best_loss": self.best_loss,
            "all_trials": self.trial_results,
            "gpr_optimal_lr": final_optimization['gpr_optimal_lr'],
            "ei_optimal_lr": final_optimization['ei_optimal_lr'],
            "predicted_loss": final_optimization['predicted_loss'],
            "uncertainty": final_optimization['uncertainty']
        }
        
        # Save results
        with open(self.save_path, "w") as f:
            json.dump(results, f, indent=4)
            
        # Create visualization
        self.plot_results(final_optimization)
        
        print("\nOptimization Results:")
        print(f"Best learning rate: {self.best_lr:.2e}")
        print(f"Best loss: {self.best_loss:.4f}")
        print(f"GPR Optimal learning rate: {final_optimization['gpr_optimal_lr']:.2e}")
        print(f"EI Optimal learning rate: {final_optimization['ei_optimal_lr']:.2e}")

    def plot_results(self, final_optimization):
        """Create visualization of optimization results"""
        X = np.array([[lr] for lr, _ in self.trial_results])
        y = np.array([loss for _, loss in self.trial_results])
        
        plt.figure(figsize=(12, 6))
        plt.semilogx(X, y, 'ko', label='Trials', markersize=8)
        
        if np.isfinite(final_optimization['gpr_optimal_lr']):
            plt.axvline(final_optimization['gpr_optimal_lr'], color='r', 
                       linestyle='--', label='GPR Optimal LR')
        if np.isfinite(final_optimization['ei_optimal_lr']):
            plt.axvline(final_optimization['ei_optimal_lr'], color='g', 
                       linestyle='--', label='EI Optimal LR')
        
        plt.xlabel('Learning Rate')
        plt.ylabel('Loss')
        plt.title('Learning Rate Optimization Results')
        plt.legend()
        plt.grid(True)
        plt.savefig('lr_optimization_plot.png', dpi=300, bbox_inches='tight')
        plt.close()

def setup_training(
    model_name,
    dataset_name=None,
    dataset=None,
    context_window=2048,
    max_samples=None,
    batch_size=4,
    gradient_accumulation_steps=8,
    num_trials=10,
    lr_range=(1e-6, 1e-4),
    output_dir="./output"
):
    """Setup complete training pipeline with LR optimization"""
    
    # Initialize dataset handler
    dataset_handler = DatasetHandler(
        dataset_name=dataset_name,
        model_name=model_name,
        dataset=dataset,
        context_window=context_window,
        max_samples=max_samples
    )
    
    # Get tokenized dataset
    tokenized_dataset = dataset_handler.get_tokenized_dataset()
    
    # Initialize model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.config.pad_token_id = dataset_handler.tokenizer.pad_token_id
    
    # Setup data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=dataset_handler.tokenizer,
        mlm=False
    )
    
    # Calculate steps
    total_steps = len(tokenized_dataset) // (batch_size * gradient_accumulation_steps)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        logging_steps=max(total_steps // 20, 1),
        learning_rate=lr_range[0],  # Initial learning rate
        weight_decay=0.01,
        fp16=False,
        bf16=True,
        warmup_steps=total_steps // 10,
        save_steps=1000000,
        save_total_limit=None,
        report_to="none",
        dataloader_num_workers=4,
        gradient_checkpointing=True,
        max_grad_norm=1.0
    )
    
    # Initialize LR optimizer callback
    lr_optimizer = LROptimizerCallback(
        num_trials=num_trials,
        lr_range=lr_range,
        optimization_steps=total_steps // num_trials
    )
    
    return {
        'model': model,
        'tokenizer': dataset_handler.tokenizer,
        'dataset': tokenized_dataset,
        'data_collator': data_collator,
        'training_args': training_args,
        'callbacks': [lr_optimizer]
    }