import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import threading
import os
from datetime import datetime
from preprocessing import DataPreprocessor
from raw_preprocessing import RawDataPreprocessor

class SappoPreprocessingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Sappo Trading Bot - Data Preprocessing Tool")
        self.root.geometry("800x600")
        
        self.loaded_files = {}
        self.processed_data = None
        self.preprocessor = DataPreprocessor(window_size=24)
        self.raw_preprocessor = RawDataPreprocessor(window_size=24)
        self.preprocessing_mode = tk.StringVar(value="standard")
        
        self.setup_ui()
        
    def setup_ui(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
        # Controls Section
        controls_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        controls_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # File selection
        self.select_files_btn = ttk.Button(
            controls_frame, 
            text="Select Data Files", 
            command=self.select_files
        )
        self.select_files_btn.grid(row=0, column=0, padx=(0, 10))
        
        # Preprocessing mode selection
        mode_frame = ttk.Frame(controls_frame)
        mode_frame.grid(row=0, column=1, padx=(10, 10))
        
        ttk.Label(mode_frame, text="Mode:").grid(row=0, column=0, padx=(0, 5))
        
        self.standard_radio = ttk.Radiobutton(
            mode_frame, 
            text="Standard (Engineered Features)", 
            variable=self.preprocessing_mode, 
            value="standard",
            command=self.on_mode_change
        )
        self.standard_radio.grid(row=0, column=1, padx=(0, 10))
        
        self.raw_radio = ttk.Radiobutton(
            mode_frame, 
            text="Raw Data (AI Self-Learning)", 
            variable=self.preprocessing_mode, 
            value="raw",
            command=self.on_mode_change
        )
        self.raw_radio.grid(row=0, column=2)
        
        # Start processing button
        self.start_processing_btn = ttk.Button(
            controls_frame, 
            text="Start Preprocessing", 
            command=self.start_preprocessing,
            state="disabled"
        )
        self.start_processing_btn.grid(row=1, column=0, columnspan=3, pady=(10, 0))
        
        # Progress Section
        progress_frame = ttk.LabelFrame(main_frame, text="Progress", padding="10")
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
        
        # Log View Section
        log_frame = ttk.LabelFrame(main_frame, text="Status & Results", padding="10")
        log_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        
        self.log_text = tk.Text(log_frame, wrap=tk.WORD, state="disabled")
        scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        self.log_message("Application initialized. Ready to process data.")
        self.log_message("Standard mode: Creates engineered features (returns, volatility, correlations)")
        self.log_message("Raw mode: Minimal processing - AI learns from pure OHLCV + time data")
        
    def log_message(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}\n"
        
        self.log_text.config(state="normal")
        self.log_text.insert(tk.END, formatted_message)
        self.log_text.see(tk.END)
        self.log_text.config(state="disabled")
        self.root.update_idletasks()
        
    def update_progress(self, value, status=""):
        self.progress_var.set(value)
        if status:
            self.progress_label.config(text=status)
        self.root.update_idletasks()
    
    def on_mode_change(self):
        """Handle preprocessing mode change"""
        mode = self.preprocessing_mode.get()
        if mode == "standard":
            self.log_message("Mode: Standard preprocessing with engineered features")
        elif mode == "raw":
            self.log_message("Mode: Raw data preprocessing for AI self-learning")
            self.log_message("⚠️  Raw mode provides minimal features - AI must discover patterns itself")
        
    def select_files(self):
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
        if not self.loaded_files:
            messagebox.showwarning("Warning", "Please select data files first.")
            return
            
        self.start_processing_btn.config(state="disabled")
        self.select_files_btn.config(state="disabled")
        
        # Run preprocessing in separate thread to keep GUI responsive
        thread = threading.Thread(target=self.run_preprocessing)
        thread.daemon = True
        thread.start()
        
    def run_preprocessing(self):
        try:
            mode = self.preprocessing_mode.get()
            self.log_message(f"Starting {mode} preprocessing pipeline...")
            self.update_progress(0, "Initializing...")
            
            def progress_callback(progress, status=""):
                self.update_progress(progress, status)
                if status:
                    self.log_message(status)
            
            # Run the appropriate preprocessing pipeline
            if mode == "raw":
                self.log_message("🔄 Using RAW data approach - AI will learn everything from scratch")
                self.processed_data = self.raw_preprocessor.raw_preprocess_pipeline(
                    self.loaded_files, 
                    progress_callback
                )
                mode_suffix = "raw"
            else:
                self.log_message("🔄 Using STANDARD approach with engineered features")
                self.processed_data = self.preprocessor.preprocess_pipeline(
                    self.loaded_files, 
                    progress_callback
                )
                mode_suffix = "standard"
            
            # Save results
            self.log_message("Saving unified dataset...")
            output_path = self.save_results(self.processed_data, mode_suffix)
            
            self.update_progress(100, "Complete!")
            self.log_message(f"✅ {mode.capitalize()} preprocessing completed successfully!")
            self.log_message(f"Output saved to: {output_path}")
            self.log_message(f"Final unified dataset shape: {self.processed_data.shape}")
            self.log_message(f"Ready for RL training with {self.processed_data.shape[0]} samples")
            
            if mode == "raw":
                self.log_message(f"🤖 Each sample: {self.processed_data.shape[2]} RAW features (OHLCV + time)")
                self.log_message("🧠 AI must discover: price patterns, volatility, correlations, cycles")
            else:
                self.log_message(f"📊 Each sample: {self.processed_data.shape[2]} engineered features")
            
        except Exception as e:
            self.log_message(f"❌ Error during preprocessing: {str(e)}")
            self.update_progress(0, "Error occurred")
        
        finally:
            self.start_processing_btn.config(state="normal")
            self.select_files_btn.config(state="normal")
    
    
    def save_results(self, data, mode_suffix="standard"):
        # Create results folder if it doesn't exist
        results_dir = "results"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
            self.log_message(f"Created results directory: {results_dir}")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"preprocessed_data_{mode_suffix}_{timestamp}.npy"
        output_path = os.path.join(results_dir, output_filename)
        
        np.save(output_path, data)
        return output_path

def main():
    root = tk.Tk()
    app = SappoPreprocessingGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()