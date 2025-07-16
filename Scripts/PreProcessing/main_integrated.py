import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import threading
import os
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import json

# Import preprocessing module
from preprocessing import DataPreprocessor

# Import RL modules (with error handling for missing dependencies)
try:
    from train import train_sappo_agent
    from evaluate import evaluate_sappo_model
    RL_AVAILABLE = True
except ImportError as e:
    RL_AVAILABLE = False
    RL_IMPORT_ERROR = str(e)

class SappoIntegratedGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Sappo Trading Bot - Complete ML Pipeline")
        self.root.geometry("1200x800")
        
        # Data storage
        self.loaded_files = {}
        self.processed_data = None
        self.preprocessor = DataPreprocessor(window_size=24)
        self.training_results = {}
        self.evaluation_results = {}
        
        # Setup main interface
        self.setup_ui()
        
        # Check RL availability
        if not RL_AVAILABLE:
            self.log_message(f"⚠️ RL modules not available: {RL_IMPORT_ERROR}")
            self.log_message("Please install: pip install gymnasium stable-baselines3 torch tensorboard")
    
    def setup_ui(self):
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(0, weight=1)
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Tab 1: Data Preprocessing
        self.setup_preprocessing_tab()
        
        # Tab 2: Training & Evaluation
        self.setup_training_tab()
    
    def setup_preprocessing_tab(self):
        """Setup the data preprocessing tab"""
        self.preprocess_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.preprocess_frame, text="Data Preprocessing")
        
        # Configure grid
        self.preprocess_frame.columnconfigure(1, weight=1)
        self.preprocess_frame.rowconfigure(2, weight=1)
        
        # Controls Section
        controls_frame = ttk.LabelFrame(self.preprocess_frame, text="Data Controls", padding="10")
        controls_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.select_files_btn = ttk.Button(
            controls_frame, 
            text="Select CSV Data Files", 
            command=self.select_files
        )
        self.select_files_btn.grid(row=0, column=0, padx=(0, 10))
        
        self.start_processing_btn = ttk.Button(
            controls_frame, 
            text="Start Preprocessing", 
            command=self.start_preprocessing,
            state="disabled"
        )
        self.start_processing_btn.grid(row=0, column=1)
        
        # Progress Section
        progress_frame = ttk.LabelFrame(self.preprocess_frame, text="Progress", padding="10")
        progress_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        progress_frame.columnconfigure(0, weight=1)
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            progress_frame, 
            variable=self.progress_var, 
            maximum=100
        )
        self.progress_bar.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        self.progress_label = ttk.Label(progress_frame, text="Ready")
        self.progress_label.grid(row=1, column=0, pady=(5, 0))
        
        # Log Section
        log_frame = ttk.LabelFrame(self.preprocess_frame, text="Processing Log", padding="10")
        log_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        
        self.log_text = tk.Text(log_frame, wrap=tk.WORD, state="disabled", height=15)
        log_scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scrollbar.set)
        
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        log_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        self.log_message("Sappo Trading Bot initialized. Ready for data preprocessing.")
    
    def setup_training_tab(self):
        """Setup the RL training & evaluation tab"""
        self.training_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.training_frame, text="RL Training & Evaluation")
        
        # Configure grid
        self.training_frame.columnconfigure(1, weight=1)
        self.training_frame.rowconfigure(3, weight=1)
        
        # Data Selection Section
        data_frame = ttk.LabelFrame(self.training_frame, text="Data Selection", padding="10")
        data_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        data_frame.columnconfigure(1, weight=1)
        
        ttk.Label(data_frame, text="Preprocessed Data:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        self.data_path_var = tk.StringVar()
        self.data_path_entry = ttk.Entry(data_frame, textvariable=self.data_path_var, state="readonly")
        self.data_path_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 10))
        
        self.select_data_btn = ttk.Button(
            data_frame, 
            text="Select .npy File", 
            command=self.select_preprocessed_data
        )
        self.select_data_btn.grid(row=0, column=2)
        
        # Hyperparameters Section
        hyperparam_frame = ttk.LabelFrame(self.training_frame, text="Hyperparameters", padding="10")
        hyperparam_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Create hyperparameter inputs
        self.setup_hyperparameter_inputs(hyperparam_frame)
        
        # Control Buttons Section
        control_frame = ttk.LabelFrame(self.training_frame, text="Training Controls", padding="10")
        control_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.train_btn = ttk.Button(
            control_frame, 
            text="Start Training", 
            command=self.start_training,
            state="disabled"
        )
        self.train_btn.grid(row=0, column=0, padx=(0, 10))
        
        self.evaluate_btn = ttk.Button(
            control_frame, 
            text="Evaluate Best Model", 
            command=self.start_evaluation,
            state="disabled"
        )
        self.evaluate_btn.grid(row=0, column=1, padx=(0, 10))
        
        self.tensorboard_btn = ttk.Button(
            control_frame, 
            text="Open TensorBoard", 
            command=self.open_tensorboard
        )
        self.tensorboard_btn.grid(row=0, column=2)
        
        # Results Section
        results_frame = ttk.LabelFrame(self.training_frame, text="Results & Visualization", padding="10")
        results_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)
        
        # Create notebook for results
        self.results_notebook = ttk.Notebook(results_frame)
        self.results_notebook.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Training log tab
        self.training_log_frame = ttk.Frame(self.results_notebook)
        self.results_notebook.add(self.training_log_frame, text="Training Log")
        
        self.training_log_text = tk.Text(self.training_log_frame, wrap=tk.WORD, state="disabled")
        training_log_scrollbar = ttk.Scrollbar(self.training_log_frame, orient="vertical", command=self.training_log_text.yview)
        self.training_log_text.configure(yscrollcommand=training_log_scrollbar.set)
        
        self.training_log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        training_log_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.training_log_frame.columnconfigure(0, weight=1)
        self.training_log_frame.rowconfigure(0, weight=1)
        
        # Performance chart tab
        self.chart_frame = ttk.Frame(self.results_notebook)
        self.results_notebook.add(self.chart_frame, text="Performance Chart")
        
        # Evaluation progress tab
        self.eval_progress_frame = ttk.Frame(self.results_notebook)
        self.results_notebook.add(self.eval_progress_frame, text="Training Progress")
        self.setup_evaluation_progress_tab()
        
        # Enable RL controls if available
        if not RL_AVAILABLE:
            for widget in [self.train_btn, self.evaluate_btn]:
                widget.configure(state="disabled")
    
    def setup_evaluation_progress_tab(self):
        """Setup the evaluation progress monitoring tab"""
        # Configure grid
        self.eval_progress_frame.columnconfigure(0, weight=1)
        self.eval_progress_frame.rowconfigure(1, weight=1)
        
        # Title and info
        title_frame = ttk.Frame(self.eval_progress_frame)
        title_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=10, pady=5)
        
        ttk.Label(title_frame, text="Validation Evaluations Every 10k Steps", 
                 font=('TkDefaultFont', 10, 'bold')).grid(row=0, column=0, sticky=tk.W)
        
        # Create treeview for evaluation results
        columns = ('Timestep', 'Mean Reward', 'Std Reward', 'Sharpe Ratio', 
                  'Total Return', 'Max Drawdown', 'Final Value', 'Trade Count')
        
        self.eval_tree = ttk.Treeview(self.eval_progress_frame, columns=columns, show='headings', height=15)
        
        # Configure column headings and widths
        column_widths = {'Timestep': 80, 'Mean Reward': 100, 'Std Reward': 100, 
                        'Sharpe Ratio': 100, 'Total Return': 100, 'Max Drawdown': 110,
                        'Final Value': 100, 'Trade Count': 100}
        
        for col in columns:
            self.eval_tree.heading(col, text=col)
            self.eval_tree.column(col, width=column_widths.get(col, 100), minwidth=80)
        
        # Add scrollbars
        eval_v_scrollbar = ttk.Scrollbar(self.eval_progress_frame, orient="vertical", command=self.eval_tree.yview)
        eval_h_scrollbar = ttk.Scrollbar(self.eval_progress_frame, orient="horizontal", command=self.eval_tree.xview)
        self.eval_tree.configure(yscrollcommand=eval_v_scrollbar.set, xscrollcommand=eval_h_scrollbar.set)
        
        # Grid layout
        self.eval_tree.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(10, 0), pady=5)
        eval_v_scrollbar.grid(row=1, column=1, sticky=(tk.N, tk.S), pady=5)
        eval_h_scrollbar.grid(row=2, column=0, sticky=(tk.W, tk.E), padx=(10, 0))
        
        # Summary frame at bottom
        summary_frame = ttk.LabelFrame(self.eval_progress_frame, text="Current Best", padding="10")
        summary_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), padx=10, pady=5)
        
        # Best metrics display
        self.best_sharpe_var = tk.StringVar(value="N/A")
        self.best_return_var = tk.StringVar(value="N/A")
        self.best_timestep_var = tk.StringVar(value="N/A")
        
        ttk.Label(summary_frame, text="Best Sharpe:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        ttk.Label(summary_frame, textvariable=self.best_sharpe_var, font=('TkDefaultFont', 9, 'bold')).grid(row=0, column=1, sticky=tk.W, padx=(0, 20))
        
        ttk.Label(summary_frame, text="Best Return:").grid(row=0, column=2, sticky=tk.W, padx=(0, 5))
        ttk.Label(summary_frame, textvariable=self.best_return_var, font=('TkDefaultFont', 9, 'bold')).grid(row=0, column=3, sticky=tk.W, padx=(0, 20))
        
        ttk.Label(summary_frame, text="At Step:").grid(row=0, column=4, sticky=tk.W, padx=(0, 5))
        ttk.Label(summary_frame, textvariable=self.best_timestep_var, font=('TkDefaultFont', 9, 'bold')).grid(row=0, column=5, sticky=tk.W)
    
    def setup_hyperparameter_inputs(self, parent):
        """Setup hyperparameter input widgets"""
        # Learning rate
        ttk.Label(parent, text="Learning Rate:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.lr_var = tk.StringVar(value="0.0001")
        ttk.Entry(parent, textvariable=self.lr_var, width=10).grid(row=0, column=1, padx=(0, 20))
        
        # Gamma
        ttk.Label(parent, text="Gamma:").grid(row=0, column=2, sticky=tk.W, padx=(0, 5))
        self.gamma_var = tk.StringVar(value="0.99")
        ttk.Entry(parent, textvariable=self.gamma_var, width=10).grid(row=0, column=3, padx=(0, 20))
        
        # Total timesteps
        ttk.Label(parent, text="Training Steps:").grid(row=0, column=4, sticky=tk.W, padx=(0, 5))
        self.timesteps_var = tk.StringVar(value="100000")
        ttk.Entry(parent, textvariable=self.timesteps_var, width=10).grid(row=0, column=5)
        
        # Reward weights
        ttk.Label(parent, text="Reward Weights:").grid(row=1, column=0, sticky=tk.W, pady=(10, 0))
        
        weight_frame = ttk.Frame(parent)
        weight_frame.grid(row=1, column=1, columnspan=5, sticky=tk.W, pady=(10, 0))
        
        # Profit weight
        ttk.Label(weight_frame, text="Profit:").grid(row=0, column=0, padx=(0, 5))
        self.w_profit_var = tk.StringVar(value="1.0")
        ttk.Entry(weight_frame, textvariable=self.w_profit_var, width=8).grid(row=0, column=1, padx=(0, 10))
        
        # Sharpe weight
        ttk.Label(weight_frame, text="Sharpe:").grid(row=0, column=2, padx=(0, 5))
        self.w_sharpe_var = tk.StringVar(value="0.5")
        ttk.Entry(weight_frame, textvariable=self.w_sharpe_var, width=8).grid(row=0, column=3, padx=(0, 10))
        
        # Cost weight
        ttk.Label(weight_frame, text="Cost:").grid(row=0, column=4, padx=(0, 5))
        self.w_cost_var = tk.StringVar(value="1.0")
        ttk.Entry(weight_frame, textvariable=self.w_cost_var, width=8).grid(row=0, column=5, padx=(0, 10))
        
        # MDD weight
        ttk.Label(weight_frame, text="MDD:").grid(row=0, column=6, padx=(0, 5))
        self.w_mdd_var = tk.StringVar(value="0.5")
        ttk.Entry(weight_frame, textvariable=self.w_mdd_var, width=8).grid(row=0, column=7)
    
    # Preprocessing methods (from original GUI)
    def log_message(self, message):
        """Log message to preprocessing tab"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}\n"
        
        self.log_text.config(state="normal")
        self.log_text.insert(tk.END, formatted_message)
        self.log_text.see(tk.END)
        self.log_text.config(state="disabled")
        self.root.update_idletasks()
    
    def log_training_message(self, message):
        """Log message to training tab"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}\n"
        
        self.training_log_text.config(state="normal")
        self.training_log_text.insert(tk.END, formatted_message)
        self.training_log_text.see(tk.END)
        self.training_log_text.config(state="disabled")
        self.root.update_idletasks()
    
    def update_progress(self, value, status=""):
        """Update progress bar"""
        self.progress_var.set(value)
        if status:
            self.progress_label.config(text=status)
        self.root.update_idletasks()
    
    def select_files(self):
        """Select CSV data files for preprocessing"""
        file_paths = filedialog.askopenfilenames(
            title="Select CSV Data Files",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialdir=os.path.join(os.path.dirname(__file__), "..", "..", "Data", "upbit_data")
        )
        
        if file_paths:
            self.loaded_files = {}
            self.log_message(f"Loading {len(file_paths)} files...")
            
            for file_path in file_paths:
                try:
                    filename = os.path.basename(file_path)
                    df = pd.read_csv(file_path)
                    self.loaded_files[filename] = df
                    self.log_message(f"Loaded: {filename} ({len(df)} rows)")
                except Exception as e:
                    self.log_message(f"Error loading {filename}: {str(e)}")
            
            if self.loaded_files:
                self.start_processing_btn.config(state="normal")
                self.log_message(f"Successfully loaded {len(self.loaded_files)} files. Ready for preprocessing.")
            else:
                self.log_message("No files were successfully loaded.")
    
    def start_preprocessing(self):
        """Start preprocessing in separate thread"""
        if not self.loaded_files:
            messagebox.showwarning("Warning", "Please select data files first.")
            return
        
        self.start_processing_btn.config(state="disabled")
        self.select_files_btn.config(state="disabled")
        
        thread = threading.Thread(target=self.run_preprocessing)
        thread.daemon = True
        thread.start()
    
    def run_preprocessing(self):
        """Run preprocessing pipeline"""
        try:
            self.log_message("Starting preprocessing pipeline...")
            self.update_progress(0, "Initializing...")
            
            def progress_callback(progress, status=""):
                self.update_progress(progress, status)
                if status:
                    self.log_message(status)
            
            # Run preprocessing
            self.processed_data = self.preprocessor.preprocess_pipeline(
                self.loaded_files, 
                progress_callback
            )
            
            # Save results
            self.log_message("Saving unified dataset...")
            output_path = self.save_results(self.processed_data)
            
            self.update_progress(100, "Complete!")
            self.log_message(f"Multi-coin preprocessing completed successfully!")
            self.log_message(f"Output saved to: {output_path}")
            self.log_message(f"Final unified dataset shape: {self.processed_data.shape}")
            self.log_message(f"Ready for RL training with {self.processed_data.shape[0]} samples")
            
            # Auto-populate training data path
            self.data_path_var.set(output_path)
            if RL_AVAILABLE:
                self.train_btn.config(state="normal")
            
        except Exception as e:
            self.log_message(f"Error during preprocessing: {str(e)}")
            self.update_progress(0, "Error occurred")
        
        finally:
            self.start_processing_btn.config(state="normal")
            self.select_files_btn.config(state="normal")
    
    def save_results(self, data):
        """Save preprocessing results"""
        results_dir = "results"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
            self.log_message(f"Created results directory: {results_dir}")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"preprocessed_data_{timestamp}.npy"
        output_path = os.path.join(results_dir, output_filename)
        
        np.save(output_path, data)
        return output_path
    
    # RL Training methods
    def select_preprocessed_data(self):
        """Select preprocessed .npy file"""
        file_path = filedialog.askopenfilename(
            title="Select Preprocessed Data File",
            filetypes=[("NumPy files", "*.npy"), ("All files", "*.*")],
            initialdir="results"
        )
        
        if file_path:
            self.data_path_var.set(file_path)
            if RL_AVAILABLE:
                self.train_btn.config(state="normal")
            self.log_training_message(f"Selected data file: {os.path.basename(file_path)}")
    
    def get_hyperparameters(self):
        """Get hyperparameters from GUI"""
        try:
            hyperparams = {
                'learning_rate': float(self.lr_var.get()),
                'gamma': float(self.gamma_var.get())
            }
            
            reward_weights = {
                'profit': float(self.w_profit_var.get()),
                'sharpe': float(self.w_sharpe_var.get()),
                'cost': float(self.w_cost_var.get()),
                'mdd': float(self.w_mdd_var.get())
            }
            
            total_timesteps = int(self.timesteps_var.get())
            
            return hyperparams, reward_weights, total_timesteps
            
        except ValueError as e:
            messagebox.showerror("Invalid Parameters", f"Please check your hyperparameters: {str(e)}")
            return None, None, None
    
    def start_training(self):
        """Start RL training"""
        if not RL_AVAILABLE:
            messagebox.showerror("RL Not Available", f"RL modules not installed: {RL_IMPORT_ERROR}")
            return
        
        if not self.data_path_var.get():
            messagebox.showwarning("Warning", "Please select preprocessed data file first.")
            return
        
        hyperparams, reward_weights, total_timesteps = self.get_hyperparameters()
        if hyperparams is None:
            return
        
        self.train_btn.config(state="disabled")
        self.evaluate_btn.config(state="disabled")
        
        self.log_training_message("Starting RL training...")
        
        # Clear previous evaluation results
        self.clear_evaluation_progress()
        
        thread = threading.Thread(target=self.run_training, 
                                args=(hyperparams, reward_weights, total_timesteps))
        thread.daemon = True
        thread.start()
    
    def run_training(self, hyperparams, reward_weights, total_timesteps):
        """Run RL training in separate thread"""
        try:
            results = train_sappo_agent(
                data_path=self.data_path_var.get(),
                hyperparameters=hyperparams,
                reward_weights=reward_weights,
                total_timesteps=total_timesteps,
                model_save_dir="models",
                log_callback=self.log_training_message,
                progress_callback=self.update_evaluation_progress
            )
            
            self.training_results = results
            
            if results['success']:
                self.log_training_message("🎉 Training completed successfully!")
                self.log_training_message(f"Best model saved: {results['best_model_path']}")
                self.log_training_message(f"Best Sharpe ratio: {results['best_sharpe_ratio']:.4f}")
                self.evaluate_btn.config(state="normal")
            else:
                self.log_training_message(f"❌ Training failed: {results['error']}")
        
        except Exception as e:
            self.log_training_message(f"❌ Training error: {str(e)}")
        
        finally:
            self.train_btn.config(state="normal")
    
    def clear_evaluation_progress(self):
        """Clear the evaluation progress table"""
        for item in self.eval_tree.get_children():
            self.eval_tree.delete(item)
        self.best_sharpe_var.set("N/A")
        self.best_return_var.set("N/A")
        self.best_timestep_var.set("N/A")
    
    def update_evaluation_progress(self, evaluation_data):
        """Update the evaluation progress table with new data"""
        if not evaluation_data:
            return
        
        # This method is called from a different thread, so we need to schedule GUI updates
        self.root.after(0, self._update_evaluation_progress_gui, evaluation_data)
    
    def _update_evaluation_progress_gui(self, evaluation_data):
        """Update evaluation progress in the main thread"""
        try:
            # Extract metrics
            timestep = evaluation_data.get('timestep', 0)
            mean_reward = evaluation_data.get('mean_reward', 0)
            std_reward = evaluation_data.get('std_reward', 0)
            sharpe_ratio = evaluation_data.get('sharpe_ratio', 0)
            total_return = evaluation_data.get('total_return', 0)
            max_drawdown = evaluation_data.get('max_drawdown', 0)
            
            portfolio_stats = evaluation_data.get('portfolio_stats', {})
            final_value = portfolio_stats.get('final_value', 0)
            trade_count = portfolio_stats.get('trade_count', 0)
            
            # Format values for display  
            values = (
                f"{timestep:,}",
                f"{mean_reward:.4f}",
                f"{std_reward:.4f}", 
                f"{sharpe_ratio:.4f}",
                f"{total_return*100:.2f}%",  # Convert to percentage
                f"{max_drawdown*100:.2f}%",  # Convert to percentage
                f"${final_value:,.0f}",
                f"{trade_count:.0f}"
            )
            
            # Insert new row
            item_id = self.eval_tree.insert('', 'end', values=values)
            
            # Highlight if this is a new best Sharpe ratio
            try:
                current_best = float(self.best_sharpe_var.get()) if self.best_sharpe_var.get() != "N/A" else -999
                if sharpe_ratio > current_best:
                    self.eval_tree.set(item_id, 'Sharpe Ratio', f"{sharpe_ratio:.4f} ⭐")
                    self.best_sharpe_var.set(f"{sharpe_ratio:.4f}")
                    self.best_return_var.set(f"{total_return*100:.2f}%")
                    self.best_timestep_var.set(f"{timestep:,}")
                    
                    # Log the new best
                    self.log_training_message(f"🌟 New best Sharpe ratio: {sharpe_ratio:.4f} at step {timestep:,}")
            except:
                # First entry
                self.best_sharpe_var.set(f"{sharpe_ratio:.4f}")
                self.best_return_var.set(f"{total_return*100:.2f}%")
                self.best_timestep_var.set(f"{timestep:,}")
            
            # Auto-scroll to bottom
            self.eval_tree.see(item_id)
            
            # Switch to Training Progress tab if not already visible
            current_tab = self.results_notebook.index(self.results_notebook.select())
            if current_tab != 2:  # Training Progress tab is index 2
                self.results_notebook.select(2)
                
        except Exception as e:
            self.log_training_message(f"Error updating progress table: {str(e)}")
    
    
    def start_evaluation(self):
        """Start model evaluation"""
        if not self.training_results.get('success'):
            messagebox.showwarning("Warning", "No successful training results found.")
            return
        
        self.evaluate_btn.config(state="disabled")
        self.log_training_message("Starting model evaluation...")
        
        thread = threading.Thread(target=self.run_evaluation)
        thread.daemon = True
        thread.start()
    
    def run_evaluation(self):
        """Run model evaluation in separate thread"""
        try:
            # Load test data (last 15% of original data)
            data = np.load(self.data_path_var.get())
            test_start = int(0.85 * data.shape[0])
            test_data = data[test_start:]
            
            results = evaluate_sappo_model(
                model_path=self.training_results['best_model_path'],
                test_data=test_data,
                reward_weights={
                    'profit': float(self.w_profit_var.get()),
                    'sharpe': float(self.w_sharpe_var.get()),
                    'cost': float(self.w_cost_var.get()),
                    'mdd': float(self.w_mdd_var.get())
                },
                save_dir="evaluation_results",
                log_callback=self.log_training_message
            )
            
            self.evaluation_results = results
            
            # Display results
            enhanced = results.get('enhanced_metrics', {})
            self.log_training_message("📊 EVALUATION RESULTS:")
            self.log_training_message(f"Total Return: {enhanced.get('total_return_pct', 0):.2f}%")
            self.log_training_message(f"Sharpe Ratio: {enhanced.get('sharpe_ratio', 0):.3f}")
            self.log_training_message(f"Max Drawdown: {enhanced.get('max_drawdown_pct', 0):.2f}%")
            self.log_training_message(f"Win Rate: {enhanced.get('win_rate', 0):.1f}%")
            
            # Show performance chart
            self.show_performance_chart(results)
            
        except Exception as e:
            self.log_training_message(f"❌ Evaluation error: {str(e)}")
        
        finally:
            self.evaluate_btn.config(state="normal")
    
    def show_performance_chart(self, results):
        """Display performance chart in GUI"""
        try:
            # Clear previous chart
            for widget in self.chart_frame.winfo_children():
                widget.destroy()
            
            enhanced = results.get('enhanced_metrics', {})
            portfolio_values = enhanced.get('portfolio_progression', [])
            
            if not portfolio_values:
                return
            
            # Create matplotlib figure
            fig, ax = plt.subplots(figsize=(10, 6))
            
            time_steps = range(len(portfolio_values))
            ax.plot(time_steps, portfolio_values, 'b-', linewidth=2, label='Agent Portfolio')
            
            # Add benchmark if available
            benchmark = results.get('benchmark', {})
            if benchmark.get('total_return_pct'):
                initial_value = portfolio_values[0]
                final_benchmark = initial_value * (1 + benchmark['total_return_pct'] / 100)
                benchmark_line = np.linspace(initial_value, final_benchmark, len(portfolio_values))
                ax.plot(time_steps, benchmark_line, 'r--', linewidth=2, label='Buy & Hold')
            
            ax.set_title('Portfolio Performance', fontsize=14, fontweight='bold')
            ax.set_xlabel('Time Steps')
            ax.set_ylabel('Portfolio Value ($)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Embed in tkinter
            canvas = FigureCanvasTkAgg(fig, self.chart_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Switch to chart tab
            self.results_notebook.select(1)
            
        except Exception as e:
            self.log_training_message(f"Chart display error: {str(e)}")
    
    def open_tensorboard(self):
        """Open TensorBoard (placeholder)"""
        self.log_training_message("To view TensorBoard:")
        self.log_training_message("1. Open terminal/command prompt")
        self.log_training_message("2. Run: tensorboard --logdir=tensorboard_logs")
        self.log_training_message("3. Open browser to http://localhost:6006")

def main():
    root = tk.Tk()
    app = SappoIntegratedGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()