import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.patches import Rectangle
from pathlib import Path
from scipy import stats
import numpy as np


class CTDAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CTD Sensor Depth Analysis Tool (3 Sensors)")
        self.root.geometry("1200x800")
        
        self.df = None
        self.df_smooth = None
        self.selection_start_idx = None
        self.selection_end_idx = None
        self.selecting = False
        self.drag_start_x = None
        # table for saved statistics
        self.stats_df = pd.DataFrame(columns=[
            "range", "n_points",
            "B_minus_A_mean", "B_minus_C_mean", "C_minus_A_mean",
            "mean_depth_all_sensors",
            "Sensor_A_mean", "Sensor_B_mean", "Sensor_C_mean",
            "Sensor_A", "Sensor_B", "Sensor_C", "source_file"
        ])
        self.last_stats = None
        # pre-selection controls (optional) - filled before processing
        self.pre_idx_start_var = tk.StringVar(value="")
        self.pre_idx_end_var = tk.StringVar(value="")
        self.pre_time_start_var = tk.StringVar(value="")
        self.pre_time_end_var = tk.StringVar(value="")
         
        self.setup_ui()
        
    def setup_ui(self):
        # Main container with two columns
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)
        
        # Left panel for controls
        left_panel = ttk.Frame(main_frame)
        left_panel.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # Right panel for plot
        right_panel = ttk.Frame(main_frame)
        right_panel.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        right_panel.columnconfigure(0, weight=1)
        right_panel.rowconfigure(0, weight=1)
        
        self.setup_left_panel(left_panel)
        self.setup_right_panel(right_panel)
        
    def setup_left_panel(self, parent):
        # File selection
        file_frame = ttk.LabelFrame(parent, text="File Selection", padding="5")
        file_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(file_frame, text="Select CSV File", command=self.load_file).pack(fill=tk.X, padx=5, pady=2)
        self.file_label = ttk.Label(file_frame, text="No file selected", wraplength=250)
        self.file_label.pack(fill=tk.X, padx=5, pady=2)
        
        # Parameters
        param_frame = ttk.LabelFrame(parent, text="Parameters", padding="5")
        param_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(param_frame, text="Skip Rows:").grid(row=0, column=0, sticky=tk.W, padx=2, pady=2)
        self.skip_rows_var = tk.IntVar(value=16)
        ttk.Entry(param_frame, textvariable=self.skip_rows_var, width=10).grid(row=0, column=1, padx=2, pady=2)
        
        ttk.Label(param_frame, text="Min Depth (m):").grid(row=1, column=0, sticky=tk.W, padx=2, pady=2)
        self.min_depth_var = tk.DoubleVar(value=672.0)
        ttk.Entry(param_frame, textvariable=self.min_depth_var, width=10).grid(row=1, column=1, padx=2, pady=2)
        
        ttk.Label(param_frame, text="Max Depth (m):").grid(row=2, column=0, sticky=tk.W, padx=2, pady=2)
        self.max_depth_var = tk.DoubleVar(value=9999.0)
        ttk.Entry(param_frame, textvariable=self.max_depth_var, width=10).grid(row=2, column=1, padx=2, pady=2)
        
        ttk.Label(param_frame, text="Trim Rows:").grid(row=3, column=0, sticky=tk.W, padx=2, pady=2)
        self.trim_rows_var = tk.IntVar(value=200)
        ttk.Entry(param_frame, textvariable=self.trim_rows_var, width=10).grid(row=3, column=1, padx=2, pady=2)
        
        ttk.Label(param_frame, text="Smooth Window:").grid(row=4, column=0, sticky=tk.W, padx=2, pady=2)
        self.smooth_window_var = tk.IntVar(value=100)
        ttk.Entry(param_frame, textvariable=self.smooth_window_var, width=10).grid(row=4, column=1, padx=2, pady=2)
        
        # Sensor mapping (3 sensors)
        sensor_frame = ttk.LabelFrame(parent, text="Sensor Patterns", padding="5")
        sensor_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(sensor_frame, text="Datetime Col:").grid(row=0, column=0, sticky=tk.W, padx=2, pady=2)
        self.datetime_col_var = tk.StringVar(value="datetime")
        ttk.Entry(sensor_frame, textvariable=self.datetime_col_var, width=15).grid(row=0, column=1, padx=2, pady=2)
        
        ttk.Label(sensor_frame, text="Sensor A:").grid(row=1, column=0, sticky=tk.W, padx=2, pady=2)
        self.sensor_a_var = tk.StringVar(value="230405")
        ttk.Entry(sensor_frame, textvariable=self.sensor_a_var, width=15).grid(row=1, column=1, padx=2, pady=2)
        
        ttk.Label(sensor_frame, text="Sensor B:").grid(row=2, column=0, sticky=tk.W, padx=2, pady=2)
        self.sensor_b_var = tk.StringVar(value="230406")
        ttk.Entry(sensor_frame, textvariable=self.sensor_b_var, width=15).grid(row=2, column=1, padx=2, pady=2)
        
        ttk.Label(sensor_frame, text="Sensor C:").grid(row=3, column=0, sticky=tk.W, padx=2, pady=2)
        self.sensor_c_var = tk.StringVar(value="236222")
        ttk.Entry(sensor_frame, textvariable=self.sensor_c_var, width=15).grid(row=3, column=1, padx=2, pady=2)
        
        # Selection info
        selection_frame = ttk.LabelFrame(parent, text="Selected Range", padding="5")
        selection_frame.pack(fill=tk.X, pady=5)
        
        # Pre-selection (index/time) - applied during process_data if provided
        preselect_frame = ttk.LabelFrame(parent, text="Pre-selection (optional)", padding="5")
        preselect_frame.pack(fill=tk.X, pady=5)
        ttk.Label(preselect_frame, text="Index Start:").grid(row=0, column=0, sticky=tk.W, padx=2, pady=2)
        ttk.Entry(preselect_frame, textvariable=self.pre_idx_start_var, width=10).grid(row=0, column=1, padx=2, pady=2)
        ttk.Label(preselect_frame, text="Index End:").grid(row=0, column=2, sticky=tk.W, padx=2, pady=2)
        ttk.Entry(preselect_frame, textvariable=self.pre_idx_end_var, width=10).grid(row=0, column=3, padx=2, pady=2)
        ttk.Label(preselect_frame, text="Time Start:").grid(row=1, column=0, sticky=tk.W, padx=2, pady=2)
        ttk.Entry(preselect_frame, textvariable=self.pre_time_start_var, width=18).grid(row=1, column=1, columnspan=3, padx=2, pady=2)
        ttk.Label(preselect_frame, text="Time End:").grid(row=2, column=0, sticky=tk.W, padx=2, pady=2)
        ttk.Entry(preselect_frame, textvariable=self.pre_time_end_var, width=18).grid(row=2, column=1, columnspan=3, padx=2, pady=2)
        ttk.Label(preselect_frame, text="(leave blank to skip)").grid(row=3, column=0, columnspan=4, sticky=tk.W, padx=2, pady=2)
        
        self.selection_label = ttk.Label(selection_frame, text="No selection\nDrag on plot to select", wraplength=250, justify=tk.LEFT)
        self.selection_label.pack(padx=5, pady=5)
        
        ttk.Button(selection_frame, text="Clear Selection", command=self.clear_selection).pack(fill=tk.X, padx=5, pady=2)
        
        # Action buttons
        button_frame = ttk.LabelFrame(parent, text="Actions", padding="5")
        button_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(button_frame, text="Process Data", command=self.process_data).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(button_frame, text="Show Statistics", command=self.show_statistics).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(button_frame, text="Plot Depths", command=lambda: self.plot_data('depths')).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(button_frame, text="Plot Differences", command=lambda: self.plot_data('differences')).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(button_frame, text="Add Stats to Table", command=self.add_stats_to_table).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(button_frame, text="Export Table", command=self.export_stats_table).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(button_frame, text="Predict Offsets", command=self.show_offset_predictor).pack(fill=tk.X, padx=5, pady=2)
         
        # Output text area
        output_frame = ttk.LabelFrame(parent, text="Output Log", padding="5")
        output_frame.pack(fill=tk.BOTH, expand=True, pady=5)
         
        self.output_text = scrolledtext.ScrolledText(output_frame, height=10, wrap=tk.WORD)
        self.output_text.pack(fill=tk.BOTH, expand=True)
        
        # Saved statistics table
        stats_frame = ttk.LabelFrame(parent, text="Saved Statistics", padding="5")
        stats_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        cols = ("range", "n_points", "B_minus_A_mean", "B_minus_C_mean", "C_minus_A_mean",
                "mean_depth_all_sensors", "Sensor_A_mean", "Sensor_B_mean", "Sensor_C_mean",
                "Sensor_A", "Sensor_B", "Sensor_C", "source_file")
        self.stats_tree = ttk.Treeview(stats_frame, columns=cols, show="headings", height=6)
        for c in cols:
            self.stats_tree.heading(c, text=c)
            self.stats_tree.column(c, width=90, anchor=tk.CENTER)
        vsb = ttk.Scrollbar(stats_frame, orient="vertical", command=self.stats_tree.yview)
        self.stats_tree.configure(yscrollcommand=vsb.set)
        self.stats_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        
    def setup_right_panel(self, parent):
        # Plot area
        self.plot_frame = ttk.Frame(parent)
        self.plot_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Initial empty plot
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.ax.text(0.5, 0.5, 'Load and process data to see plot', 
                    ha='center', va='center', transform=self.ax.transAxes, fontsize=14)
        self.ax.set_xlabel('Time')
        self.ax.set_ylabel('Depth (m)')
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.draw()
        # add Matplotlib navigation toolbar (pan/zoom)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.plot_frame)
        self.toolbar.update()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Connect mouse events
        self.canvas.mpl_connect('button_press_event', self.on_mouse_press)
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.canvas.mpl_connect('button_release_event', self.on_mouse_release)
        
    def log(self, message):
        self.output_text.insert(tk.END, message + "\n")
        self.output_text.see(tk.END)
        self.root.update_idletasks()
        
    def load_file(self):
        filename = filedialog.askopenfilename(
            title="Select CTD CSV File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if filename:
            self.file_label.config(text=Path(filename).name)
            self.filepath = filename
            self.log(f"Selected: {Path(filename).name}")
            
    def find_sensor_column(self, df, pattern):
        """Find column matching the sensor pattern"""
        # Look for columns containing both the pattern and 'Depth'
        matches = [col for col in df.columns if pattern in col and 'Depth' in col]
        if matches:
            self.log(f"  Found column for {pattern}: {matches[0]}")
            return matches[0]
        self.log(f"  ERROR: No column found for pattern '{pattern}'")
        self.log(f"  Available columns: {list(df.columns)}")
        return None
        
    def clear_selection(self):
        self.selection_start_idx = None
        self.selection_end_idx = None
        self.selection_label.config(text="No selection\nDrag on plot to select")
        self.log("Selection cleared")
        if hasattr(self, 'current_plot_type'):
            self.plot_data(self.current_plot_type)
        
    def process_data(self):
        if not hasattr(self, 'filepath'):
            messagebox.showerror("Error", "Please select a file first")
            return
            
        try:
            self.log("\n" + "="*50)
            self.log("Processing...")
            
            # Load data
            skip_rows = self.skip_rows_var.get()
            # Read with comment='#' to automatically skip comment lines
            df = pd.read_csv(self.filepath, comment='#')
            self.log(f"Loaded {len(df)} rows")
            self.log(f"Columns found: {len(df.columns)} columns")
            
            # Find sensor columns
            datetime_col = self.datetime_col_var.get()
            sensor_a_col = self.find_sensor_column(df, self.sensor_a_var.get())
            sensor_b_col = self.find_sensor_column(df, self.sensor_b_var.get())
            sensor_c_col = self.find_sensor_column(df, self.sensor_c_var.get())
            
            if not all([sensor_a_col, sensor_b_col, sensor_c_col]):
                raise ValueError("Could not find all three sensor columns")
            
            self.log(f"Found sensors A, B, and C")
            
            # Rename columns
            df = df.rename(columns={
                sensor_a_col: 'Sensor_A_Depth',
                sensor_b_col: 'Sensor_B_Depth',
                sensor_c_col: 'Sensor_C_Depth'
            })
            
            # Select depth columns and handle NaN
            depth_cols = ['Sensor_A_Depth', 'Sensor_B_Depth', 'Sensor_C_Depth']
            df_depth = df[[datetime_col] + depth_cols].copy()
            
            # Drop rows where ALL depth values are NaN
            df_depth = df_depth.dropna(subset=depth_cols, how='all')
            self.log(f"After dropping rows with all NaN depths: {len(df_depth)} rows")
            
            # For filtering, use the first non-NaN sensor
            df_depth['filter_depth'] = df_depth['Sensor_A_Depth'].fillna(
                df_depth['Sensor_B_Depth']).fillna(df_depth['Sensor_C_Depth'])
            
            # Filter by depth range
            min_depth = self.min_depth_var.get()
            max_depth = self.max_depth_var.get()
            
            df_bottom = df_depth[
                (df_depth['filter_depth'] >= min_depth) & 
                (df_depth['filter_depth'] <= max_depth)
            ].copy()
            df_bottom = df_bottom.drop(columns=['filter_depth'])
            self.log(f"Filtered by depth range: {len(df_bottom)} rows")
            
            # Trim edges
            trim_rows = self.trim_rows_var.get()
            if len(df_bottom) > trim_rows * 2:
                df_trimmed = df_bottom.iloc[trim_rows:-trim_rows].copy()
                self.log(f"Trimmed: {len(df_trimmed)} rows")
            else:
                df_trimmed = df_bottom.copy()
            
            # Apply optional pre-selection by index
            idx_start = self.pre_idx_start_var.get().strip()
            idx_end = self.pre_idx_end_var.get().strip()
            if idx_start or idx_end:
                try:
                    s = int(idx_start) if idx_start else 0
                    e = int(idx_end) if idx_end else len(df_trimmed) - 1
                except ValueError:
                    raise ValueError("Invalid index pre-selection values (must be integers)")
                df_trimmed = df_trimmed.reset_index(drop=True).iloc[max(0, s):min(len(df_trimmed), e + 1)].copy()
                self.log(f"Applied index pre-selection: rows {s}-{e} -> {len(df_trimmed)} rows")
            
            # Apply optional pre-selection by time
            time_start = self.pre_time_start_var.get().strip()
            time_end = self.pre_time_end_var.get().strip()
            if time_start or time_end:
                if datetime_col not in df_trimmed.columns:
                    raise ValueError(f"Datetime column '{datetime_col}' not found for time pre-selection")
                df_trimmed[datetime_col] = pd.to_datetime(df_trimmed[datetime_col], errors='coerce')
                if df_trimmed[datetime_col].isnull().any():
                    raise ValueError("Found invalid datetimes")
                ts = pd.to_datetime(time_start) if time_start else df_trimmed[datetime_col].min()
                te = pd.to_datetime(time_end) if time_end else df_trimmed[datetime_col].max()
                mask = (df_trimmed[datetime_col] >= ts) & (df_trimmed[datetime_col] <= te)
                df_trimmed = df_trimmed.loc[mask].reset_index(drop=True)
                self.log(f"Applied time pre-selection: {ts} to {te} -> {len(df_trimmed)} rows")
             
            # Keep raw copy and compute raw differences
            df_raw = df_trimmed.reset_index(drop=True)
            df_raw['B_minus_A_raw'] = df_raw['Sensor_B_Depth'] - df_raw['Sensor_A_Depth']
            df_raw['B_minus_C_raw'] = df_raw['Sensor_B_Depth'] - df_raw['Sensor_C_Depth']
            df_raw['C_minus_A_raw'] = df_raw['Sensor_C_Depth'] - df_raw['Sensor_A_Depth']
             
            # Apply smoothing
            smooth_window = self.smooth_window_var.get()
            df_smooth = df_raw[[datetime_col]].copy()
            df_smooth['Sensor_A_Depth'] = df_raw['Sensor_A_Depth'].rolling(window=smooth_window, min_periods=1).mean()
            df_smooth['Sensor_B_Depth'] = df_raw['Sensor_B_Depth'].rolling(window=smooth_window, min_periods=1).mean()
            df_smooth['Sensor_C_Depth'] = df_raw['Sensor_C_Depth'].rolling(window=smooth_window, min_periods=1).mean()
            
            # Calculate smoothed differences
            df_smooth['B_minus_A'] = df_smooth['Sensor_B_Depth'] - df_smooth['Sensor_A_Depth']
            df_smooth['B_minus_C'] = df_smooth['Sensor_B_Depth'] - df_smooth['Sensor_C_Depth']
            df_smooth['C_minus_A'] = df_smooth['Sensor_C_Depth'] - df_smooth['Sensor_A_Depth']
            
            df_smooth = df_smooth.reset_index(drop=True)
            
            self.df = df_raw
            self.df_smooth = df_smooth
            
            self.log("Complete! Ready to plot.")
            
            # Plot depths by default
            self.plot_data('depths')
            
        except Exception as e:
            self.log(f"ERROR: {str(e)}")
            messagebox.showerror("Error", f"Processing failed: {str(e)}")
    
    def plot_data(self, plot_type):
        if self.df_smooth is None:
            messagebox.showerror("Error", "Please process data first")
            return
        
        self.current_plot_type = plot_type
        datetime_col = self.datetime_col_var.get()
        
        self.ax.clear()
        
        if plot_type == 'depths':
            self.ax.plot(self.df_smooth.index, self.df_smooth['Sensor_A_Depth'], 
                        label='Sensor A', linewidth=1.5, alpha=0.8)
            self.ax.plot(self.df_smooth.index, self.df_smooth['Sensor_B_Depth'], 
                        label='Sensor B', linewidth=1.5, alpha=0.8)
            self.ax.plot(self.df_smooth.index, self.df_smooth['Sensor_C_Depth'], 
                        label='Sensor C', linewidth=1.5, alpha=0.8)
            self.ax.set_ylabel('Depth (m)')
            self.ax.invert_yaxis()
            self.ax.set_title('CTD Depth Sensors (Drag to select range for analysis)')
        else:  # differences
            if self.selection_start_idx is not None and self.selection_end_idx is not None:
                s = self.selection_start_idx
                e = self.selection_end_idx + 1
                df_raw = self.df.iloc[s:e]
                df_sm = self.df_smooth.iloc[s:e]
                self.ax.set_xlim(self.selection_start_idx, self.selection_end_idx)
            else:
                df_raw = self.df
                df_sm = self.df_smooth
            
            # Plot raw (faded) and smoothed (bold) differences for all 3 comparisons
            self.ax.plot(df_raw.index, df_raw['B_minus_A_raw'], linewidth=1.0, alpha=0.3, color='C0')
            self.ax.plot(df_sm.index, df_sm['B_minus_A'], label='B - A', linewidth=2.0, alpha=1.0, color='C0')
            
            self.ax.plot(df_raw.index, df_raw['B_minus_C_raw'], linewidth=1.0, alpha=0.3, color='C1')
            self.ax.plot(df_sm.index, df_sm['B_minus_C'], label='B - C', linewidth=2.0, alpha=1.0, color='C1')
            
            self.ax.plot(df_raw.index, df_raw['C_minus_A_raw'], linewidth=1.0, alpha=0.3, color='C2')
            self.ax.plot(df_sm.index, df_sm['C_minus_A'], label='C - A', linewidth=2.0, alpha=1.0, color='C2')

            self.ax.set_ylabel('Depth Difference (m)')
            self.ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
            self.ax.set_title('Sensor Differences (Drag to select range for analysis)')
         
        self.ax.set_xlabel('Row Index')
        self.ax.legend()
        self.ax.grid(True, alpha=0.3)
         
        # Highlight selection if exists
        if self.selection_start_idx is not None and self.selection_end_idx is not None:
            ylim = self.ax.get_ylim()
            self.ax.axvspan(self.selection_start_idx, self.selection_end_idx, 
                           alpha=0.2, color='red', label='_nolegend_')
        
        self.canvas.draw()
        
    def on_mouse_press(self, event):
        if getattr(self, 'toolbar', None) and getattr(self.toolbar, 'mode', ''):
            return
        if event.inaxes != self.ax or self.df_smooth is None:
            return
        self.selecting = True
        self.drag_start_x = event.xdata
         
    def on_mouse_move(self, event):
        if getattr(self, 'toolbar', None) and getattr(self.toolbar, 'mode', ''):
            return
        if not self.selecting or event.inaxes != self.ax or self.df_smooth is None:
            return
        
        self.plot_data(self.current_plot_type if hasattr(self, 'current_plot_type') else 'depths')
        
        if event.xdata is not None:
            ylim = self.ax.get_ylim()
            self.ax.axvspan(min(self.drag_start_x, event.xdata), 
                           max(self.drag_start_x, event.xdata), 
                           alpha=0.3, color='yellow', label='_nolegend_')
            self.canvas.draw()
        
    def on_mouse_release(self, event):
        if getattr(self, 'toolbar', None) and getattr(self.toolbar, 'mode', ''):
            return
        if not self.selecting or self.df_smooth is None:
            return
        
        self.selecting = False
        
        if event.inaxes == self.ax and event.xdata is not None:
            start_x = min(self.drag_start_x, event.xdata)
            end_x = max(self.drag_start_x, event.xdata)
            
            self.selection_start_idx = max(0, int(start_x))
            self.selection_end_idx = min(len(self.df_smooth) - 1, int(end_x))
            
            n_points = self.selection_end_idx - self.selection_start_idx + 1
            self.selection_label.config(
                text=f"Selected:\nRows {self.selection_start_idx} to {self.selection_end_idx}\n({n_points} points)"
            )
            self.log(f"Selected rows {self.selection_start_idx} to {self.selection_end_idx}")
            
            self.plot_data(self.current_plot_type if hasattr(self, 'current_plot_type') else 'depths')
            
    def show_statistics(self):
        if self.df_smooth is None:
            messagebox.showerror("Error", "Please process data first")
            return
        
        if self.selection_start_idx is not None and self.selection_end_idx is not None:
            df_calc = self.df_smooth.iloc[self.selection_start_idx:self.selection_end_idx+1]
            range_info = f"Selected range: rows {self.selection_start_idx}-{self.selection_end_idx} ({len(df_calc)} points)"
        else:
            df_calc = self.df_smooth
            range_info = "Full dataset (no selection)"
            
        self.log("\n" + "="*50)
        self.log("STATISTICS")
        self.log(range_info)
        self.log("="*50)
        
        self.log("\nB minus A:")
        self.log(str(df_calc['B_minus_A'].describe()))
        
        self.log("\nB minus C:")
        self.log(str(df_calc['B_minus_C'].describe()))
        
        self.log("\nC minus A:")
        self.log(str(df_calc['C_minus_A'].describe()))
        
        # Mean depth across all three sensors
        mean_depth_all = df_calc[['Sensor_A_Depth', 'Sensor_B_Depth', 'Sensor_C_Depth']].mean(axis=1).mean()
        self.log(f"\nMean depth (all sensors): {mean_depth_all:.6f}")

        # Per-sensor means
        sensor_a_mean = float(df_calc['Sensor_A_Depth'].mean())
        sensor_b_mean = float(df_calc['Sensor_B_Depth'].mean())
        sensor_c_mean = float(df_calc['Sensor_C_Depth'].mean())
        self.log(f"Mean Sensor A: {sensor_a_mean:.6f}")
        self.log(f"Mean Sensor B: {sensor_b_mean:.6f}")
        self.log(f"Mean Sensor C: {sensor_c_mean:.6f}")
 
        # Store last stats
        self.last_stats = {
            "range": range_info,
            "n_points": len(df_calc),
            "B_minus_A_mean": float(df_calc['B_minus_A'].mean()),
            "B_minus_C_mean": float(df_calc['B_minus_C'].mean()),
            "C_minus_A_mean": float(df_calc['C_minus_A'].mean()),
            "mean_depth_all_sensors": float(mean_depth_all),
            "Sensor_A_mean": sensor_a_mean,
            "Sensor_B_mean": sensor_b_mean,
            "Sensor_C_mean": sensor_c_mean
        }
        self.last_stats["Sensor_A"] = self.sensor_a_var.get()
        self.last_stats["Sensor_B"] = self.sensor_b_var.get()
        self.last_stats["Sensor_C"] = self.sensor_c_var.get()
        self.last_stats["source_file"] = Path(getattr(self, "filepath", "")).name
        
    def add_stats_to_table(self):
        if self.last_stats is None:
            messagebox.showerror("Error", "No statistics available. Run 'Show Statistics' first.")
            return
        self.stats_df = pd.concat([self.stats_df, pd.DataFrame([self.last_stats])], ignore_index=True)
        vals = [self.last_stats[c] for c in self.stats_df.columns]
        self.stats_tree.insert("", "end", values=vals)
        self.log("Added current statistics to table")
        
    def export_stats_table(self):
        if self.stats_df.empty:
            messagebox.showerror("Error", "No saved statistics to export")
            return
        fname = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files","*.csv")])
        if fname:
            self.stats_df.to_csv(fname, index=False)
            self.log(f"Exported statistics to {Path(fname).name}")
    
    def show_offset_predictor(self):
        """Show window for offset prediction based on linear regression"""
        if len(self.stats_df) < 2:
            messagebox.showerror("Error", "Need at least 2 saved statistics for regression analysis")
            return
        
        # Create new window
        pred_window = tk.Toplevel(self.root)
        pred_window.title("Offset Predictor (3 Sensors)")
        pred_window.geometry("900x700")
        
        # Main layout
        main_frame = ttk.Frame(pred_window, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left panel for controls and results
        left_panel = ttk.Frame(main_frame)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # Right panel for plot
        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Perform regression for all three comparisons
        depth_data = self.stats_df['mean_depth_all_sensors'].values
        ba_data = self.stats_df['B_minus_A_mean'].values
        bc_data = self.stats_df['B_minus_C_mean'].values
        ca_data = self.stats_df['C_minus_A_mean'].values
        
        # Remove NaN values
        mask = ~(np.isnan(depth_data) | np.isnan(ba_data) | np.isnan(bc_data) | np.isnan(ca_data))
        depth_clean = depth_data[mask]
        ba_clean = ba_data[mask]
        bc_clean = bc_data[mask]
        ca_clean = ca_data[mask]
        
        if len(depth_clean) < 2:
            messagebox.showerror("Error", "Not enough valid data points for regression")
            pred_window.destroy()
            return
        
        # Linear regressions
        res_ba = stats.linregress(depth_clean, ba_clean)
        res_bc = stats.linregress(depth_clean, bc_clean)
        res_ca = stats.linregress(depth_clean, ca_clean)
        r_squared_ba = res_ba.rvalue ** 2
        r_squared_bc = res_bc.rvalue ** 2
        r_squared_ca = res_ca.rvalue ** 2
        
        # Controls frame
        ctrl_frame = ttk.LabelFrame(left_panel, text="Prediction", padding="10")
        ctrl_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(ctrl_frame, text="Target Depth (m):").pack(anchor=tk.W)
        depth_var = tk.DoubleVar(value=700.0)
        depth_entry = ttk.Entry(ctrl_frame, textvariable=depth_var, width=15)
        depth_entry.pack(fill=tk.X, pady=5)
        
        result_text = scrolledtext.ScrolledText(ctrl_frame, height=12, width=35, wrap=tk.WORD)
        result_text.pack(fill=tk.BOTH, expand=True, pady=5)
        
        def predict():
            target = depth_var.get()
            offset_ba = (res_ba.slope * target) + res_ba.intercept
            offset_bc = (res_bc.slope * target) + res_bc.intercept
            offset_ca = (res_ca.slope * target) + res_ca.intercept
            
            result_text.delete(1.0, tk.END)
            result_text.insert(tk.END, f"Predictions for {target}m depth:\n")
            result_text.insert(tk.END, f"="*35 + "\n\n")
            result_text.insert(tk.END, f"B - A Offset: {offset_ba:.6f} m\n")
            result_text.insert(tk.END, f"B - C Offset: {offset_bc:.6f} m\n")
            result_text.insert(tk.END, f"C - A Offset: {offset_ca:.6f} m\n")
        
        ttk.Button(ctrl_frame, text="Calculate", command=predict).pack(fill=tk.X, pady=5)
        
        # Model info frame
        info_frame = ttk.LabelFrame(left_panel, text="Regression Models", padding="10")
        info_frame.pack(fill=tk.X, pady=5)
        
        info_text = scrolledtext.ScrolledText(info_frame, height=12, width=35, wrap=tk.WORD)
        info_text.pack(fill=tk.BOTH, expand=True)
        
        info_text.insert(tk.END, "B - A vs Depth\n")
        info_text.insert(tk.END, f"R²={r_squared_ba:.4f}, slope={res_ba.slope:.6f}\n\n")
        
        info_text.insert(tk.END, "B - C vs Depth\n")
        info_text.insert(tk.END, f"R²={r_squared_bc:.4f}, slope={res_bc.slope:.6f}\n\n")
        
        info_text.insert(tk.END, "C - A vs Depth\n")
        info_text.insert(tk.END, f"R²={r_squared_ca:.4f}, slope={res_ca.slope:.6f}\n\n")
        
        info_text.insert(tk.END, f"Data points: {len(depth_clean)}\n")
        
        # Plot
        fig, ax = plt.subplots(figsize=(6, 5))
        
        # Scatter plots and trendlines for all three
        ax.scatter(depth_clean, ba_clean, color='C0', label='B - A', alpha=0.6, s=50)
        trendline_ba = res_ba.slope * depth_clean + res_ba.intercept
        ax.plot(depth_clean, trendline_ba, color='C0', linewidth=2, linestyle='--')
        
        ax.scatter(depth_clean, bc_clean, color='C1', label='B - C', alpha=0.6, s=50)
        trendline_bc = res_bc.slope * depth_clean + res_bc.intercept
        ax.plot(depth_clean, trendline_bc, color='C1', linewidth=2, linestyle='--')
        
        ax.scatter(depth_clean, ca_clean, color='C2', label='C - A', alpha=0.6, s=50)
        trendline_ca = res_ca.slope * depth_clean + res_ca.intercept
        ax.plot(depth_clean, trendline_ca, color='C2', linewidth=2, linestyle='--')
        
        ax.set_xlabel('Mean Depth (m)', fontsize=11)
        ax.set_ylabel('Sensor Bias (m)', fontsize=11)
        ax.set_title('Sensor Differences vs Depth', fontsize=12, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        
        canvas = FigureCanvasTkAgg(fig, master=right_panel)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Toolbar
        toolbar = NavigationToolbar2Tk(canvas, right_panel)
        toolbar.update()
        
        # Export button
        export_frame = ttk.Frame(left_panel)
        export_frame.pack(fill=tk.X, pady=5)
        
        def export_plot():
            fname = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("PDF files", "*.pdf"), ("All files", "*.*")]
            )
            if fname:
                fig.savefig(fname, dpi=300, bbox_inches='tight')
                messagebox.showinfo("Success", f"Plot saved to {Path(fname).name}")
        
        ttk.Button(export_frame, text="Export Plot", command=export_plot).pack(fill=tk.X)


if __name__ == "__main__":
    root = tk.Tk()
    app = CTDAnalyzerApp(root)
    root.mainloop()