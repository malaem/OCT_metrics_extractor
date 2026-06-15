"""GUI for OCT Longitudinal RNFL Analysis Pipeline.

Graphical interface for the two-stage OCT analysis workflow:
1. Extract metadata from FDA files
2. Perform paired longitudinal analysis

Features:
- Folder/file selection dialogs
- Configurable parameters (workers, pairing mode)
- Real-time progress display
- Automatic sequential execution
- Error handling and reporting

Usage:
    python oct_analysis_gui.py

Author: Marco Miranda
Date: 28 May 2026
"""


import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
import threading
import sys
import multiprocessing
import contextlib
from pathlib import Path
from multiprocessing import cpu_count
import queue
import os

from metadata_extractor import extract_metadata_batch
from data_extractor_paired import run_paired_analysis
from shared_resources.system_awake import keep_system_awake


class _QueueWriter:
    """Redirects sys.stdout writes to a queue so pipeline print() output
    appears in the GUI log widget in real time."""

    def __init__(self, q: queue.Queue):
        self._q = q
        self._buf = ""

    def write(self, text: str):
        self._buf += text
        while '\n' in self._buf:
            line, self._buf = self._buf.split('\n', 1)
            self._q.put(line)

    def flush(self):
        if self._buf:
            self._q.put(self._buf)
            self._buf = ""


class OCTAnalysisGUI:
    """Main GUI application for OCT analysis pipeline."""
    
    def __init__(self, root):
        """Initialize GUI components."""
        self.root = root
        self.root.title("OCT Longitudinal RNFL Analysis")
        self._set_initial_window_size()
        
        # Set application icon
        try:
            icon_path = Path(__file__).parent / "icon.png"
            if icon_path.exists():
                self.root.iconphoto(True, tk.PhotoImage(file=str(icon_path)))
        except Exception:
            pass  # Icon is optional, don't fail if it can't be loaded
        
        # Default values
        self.default_workers = max(1, cpu_count() - 1)
        
        # Process tracking
        self.is_running = False
        self._stop_event = threading.Event()
        self.output_queue = queue.Queue()   # pipeline text → log widget
        self.action_queue = queue.Queue()   # callables to run on the main thread
        
        # Handle window close button
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        
        # Setup GUI
        self.setup_gui()
        
        # Start output monitor
        self.monitor_output()
    
    def setup_gui(self):
        """Create all GUI components."""
        # Outer container
        container = ttk.Frame(self.root)
        container.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Scrollable canvas so the full form remains accessible on smaller monitors
        self.canvas = tk.Canvas(container, highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.canvas_scrollbar = ttk.Scrollbar(container, orient="vertical", command=self.canvas.yview)
        self.canvas_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.canvas.configure(yscrollcommand=self.canvas_scrollbar.set)

        # Configure grid weights for resizing
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        container.columnconfigure(0, weight=1)
        container.rowconfigure(0, weight=1)

        # Main form hosted inside the canvas
        main_frame = ttk.Frame(self.canvas, padding="10")
        self._canvas_window = self.canvas.create_window((0, 0), window=main_frame, anchor="nw")
        self.main_frame = main_frame
        main_frame.columnconfigure(1, weight=1)
        main_frame.bind("<Configure>", self._on_main_frame_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self._bind_mousewheel(self.canvas)
        
        # Title
        title_label = ttk.Label(
            main_frame,
            text="OCT Longitudinal RNFL Analysis Pipeline",
            font=("Arial", 14, "bold")
        )
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 15))
        
        # === STAGE 1: Metadata Extraction ===
        stage1_frame = ttk.LabelFrame(main_frame, text="Stage 1: Metadata (Create New OR Use Existing)", padding="10")
        stage1_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        stage1_frame.columnconfigure(1, weight=1)
        
        # Option A: Create new metadata from FDA directory
        ttk.Label(stage1_frame, text="Option A - Extract from FDA:", font=('Arial', 9, 'bold')).grid(
            row=0, column=0, columnspan=3, sticky=tk.W, pady=(0, 5)
        )
        
        # Input directory (FDA files)
        ttk.Label(stage1_frame, text="  FDA Files Directory:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.fda_dir_var = tk.StringVar()
        ttk.Entry(stage1_frame, textvariable=self.fda_dir_var, width=50).grid(
            row=1, column=1, sticky=(tk.W, tk.E), padx=5
        )
        ttk.Button(stage1_frame, text="Browse...", command=self.browse_fda_dir).grid(
            row=1, column=2, padx=5
        )
        
        # Metadata output file (for Option A)
        ttk.Label(stage1_frame, text="  Save Metadata As:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.metadata_output_var = tk.StringVar(value="metadata.csv")
        ttk.Entry(stage1_frame, textvariable=self.metadata_output_var, width=50).grid(
            row=2, column=1, sticky=(tk.W, tk.E), padx=5
        )
        ttk.Button(stage1_frame, text="Browse...", command=self.browse_metadata_output).grid(
            row=2, column=2, padx=5
        )
        
        # Separator
        ttk.Separator(stage1_frame, orient='horizontal').grid(
            row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10
        )
        
        # Option B: Use existing metadata
        ttk.Label(stage1_frame, text="Option B - Use Existing:", font=('Arial', 9, 'bold')).grid(
            row=4, column=0, columnspan=3, sticky=tk.W, pady=(0, 5)
        )
        
        ttk.Label(stage1_frame, text="  Metadata CSV File:").grid(row=5, column=0, sticky=tk.W, pady=5)
        self.existing_metadata_var = tk.StringVar()
        ttk.Entry(stage1_frame, textvariable=self.existing_metadata_var, width=50).grid(
            row=5, column=1, sticky=(tk.W, tk.E), padx=5
        )
        ttk.Button(stage1_frame, text="Browse...", command=self.browse_existing_metadata).grid(
            row=5, column=2, padx=5
        )
        
        # === STAGE 2: Paired Analysis ===
        stage2_frame = ttk.LabelFrame(main_frame, text="Stage 2: Paired Analysis", padding="10")
        stage2_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        stage2_frame.columnconfigure(1, weight=1)
        
        # Results output directory
        ttk.Label(stage2_frame, text="Results Directory:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.results_dir_var = tk.StringVar(value="results/")
        ttk.Entry(stage2_frame, textvariable=self.results_dir_var, width=50).grid(
            row=0, column=1, sticky=(tk.W, tk.E), padx=5
        )
        ttk.Button(stage2_frame, text="Browse...", command=self.browse_results_dir).grid(
            row=0, column=2, padx=5
        )
        
        # Output base name
        ttk.Label(stage2_frame, text="Output Base Name:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.output_base_name_var = tk.StringVar(value="scan_metrics")
        ttk.Entry(stage2_frame, textvariable=self.output_base_name_var, width=50).grid(
            row=1, column=1, sticky=(tk.W, tk.E), padx=5
        )
        ttk.Label(
            stage2_frame,
            text="Suffixes added: _TSNIT, _Macula6, _ETDRS",
            font=("Arial", 8),
            foreground="gray"
        ).grid(row=2, column=1, sticky=tk.W, padx=5)
        
        # Layer selection
        ttk.Label(stage2_frame, text="Layers to Extract:").grid(row=3, column=0, sticky=tk.W, pady=5)
        layer_frame = ttk.Frame(stage2_frame)
        layer_frame.grid(row=3, column=1, sticky=tk.W, padx=5)
        
        self.layer_cprnfl_var = tk.BooleanVar(value=True)
        self.layer_gclp_var = tk.BooleanVar(value=False)
        self.layer_gclpp_var = tk.BooleanVar(value=False)
        self.layer_retina_var = tk.BooleanVar(value=False)

        self.layer_cprnfl_cb = ttk.Checkbutton(layer_frame, text="cpRNFL", variable=self.layer_cprnfl_var)
        self.layer_gclp_cb = ttk.Checkbutton(layer_frame, text="GCL+", variable=self.layer_gclp_var)
        self.layer_gclpp_cb = ttk.Checkbutton(layer_frame, text="GCL++", variable=self.layer_gclpp_var)
        self.layer_retina_cb = ttk.Checkbutton(layer_frame, text="Retina", variable=self.layer_retina_var)
        self.layer_cprnfl_cb.pack(side=tk.LEFT, padx=5)
        self.layer_gclp_cb.pack(side=tk.LEFT, padx=5)
        self.layer_gclpp_cb.pack(side=tk.LEFT, padx=5)
        self.layer_retina_cb.pack(side=tk.LEFT, padx=5)
        
        # Fixation filter
        # NOTE: 'Disc' is intentionally excluded from this list until disc scan processing
        # has been validated. The backend (pairing_utils.py, LAYER_FIXATION_COMPAT) fully
        # supports Disc; re-enable here by adding "Disc" back to the values list and
        # restoring the Disc branch in _on_fixation_filter_changed below.
        ttk.Label(stage2_frame, text="Fixation Filter:").grid(row=4, column=0, sticky=tk.W, pady=5)
        self.fixation_filter_var = tk.StringVar(value="All")
        fixation_combo = ttk.Combobox(
            stage2_frame,
            textvariable=self.fixation_filter_var,
            values=["All", "3D Wide", "Macula"],
            state="readonly",
            width=20
        )
        fixation_combo.grid(row=4, column=1, sticky=tk.W, padx=5)
        
        # Instrument filter
        ttk.Label(stage2_frame, text="Instrument Filter:").grid(row=5, column=0, sticky=tk.W, pady=5)
        self.instrument_filter_var = tk.StringVar(value="Both")
        instrument_combo = ttk.Combobox(
            stage2_frame,
            textvariable=self.instrument_filter_var,
            values=["Both", "Maestro", "Triton"],
            state="readonly",
            width=20
        )
        instrument_combo.grid(row=5, column=1, sticky=tk.W, padx=5)
        
        # Info about pairing (removed dropdown - always uses first_vs_all)
        pairing_info = ttk.Label(
            stage2_frame,
            text="Pairing: First acquired scan (baseline) vs all follow-ups",
            font=("Arial", 9),
            foreground="#0066cc"
        )
        pairing_info.grid(row=6, column=0, columnspan=3, sticky=tk.W, pady=5)
        
        # === CONFIGURATION ===
        config_frame = ttk.LabelFrame(main_frame, text="Configuration", padding="10")
        config_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Workers
        ttk.Label(config_frame, text="Parallel Workers:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.workers_var = tk.IntVar(value=self.default_workers)
        workers_spinbox = ttk.Spinbox(
            config_frame,
            from_=1,
            to=cpu_count(),
            textvariable=self.workers_var,
            width=10
        )
        workers_spinbox.grid(row=0, column=1, sticky=tk.W, padx=5)
        ttk.Label(
            config_frame,
            text=f"(default: {self.default_workers}, max: {cpu_count()})",
            font=("Arial", 8),
            foreground="gray"
        ).grid(row=0, column=2, sticky=tk.W)
        
        # Resume checkbox
        self.resume_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            config_frame,
            text="Resume from checkpoint (if available)",
            variable=self.resume_var
        ).grid(row=1, column=0, columnspan=3, sticky=tk.W, pady=5)
        
        # Alignment mode dropdown
        ttk.Label(config_frame, text="Alignment Mode:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.alignment_mode_var = tk.StringVar(value="no-aligned")
        self.alignment_combo = ttk.Combobox(
            config_frame,
            textvariable=self.alignment_mode_var,
            values=["no-aligned", "aligned", "both"],
            state="readonly",
            width=20
        )
        self.alignment_combo.grid(row=2, column=1, sticky=tk.W, padx=5)
        
        # Help text for alignment modes
        alignment_help = ttk.Label(
            config_frame,
            text="no-aligned: no pairing (unpaired-only) | aligned: paired Wide + KAZE registration | both: paired Wide aligned + unaligned",
            font=("Arial", 8),
            foreground="gray"
        )
        alignment_help.grid(row=3, column=0, columnspan=3, sticky=tk.W, pady=(0, 5))

        # Disable alignment options and incompatible layers when fixation filter changes.
        # Compatibility:
        #   Wide / All  → all layers allowed; alignment allowed
        #   Macula      → GCL+, GCL++, Retina only (cpRNFL disabled); no alignment
        #   Disc        → cpRNFL only (GCL+/GCL++/Retina disabled); no alignment
        #                 (Disc is currently hidden from the GUI — see fixation combobox above)
        def _on_fixation_filter_changed(*args):
            fixation = self.fixation_filter_var.get()

            # --- alignment combobox ---
            if fixation in ("Macula", "Disc"):
                self.alignment_mode_var.set("no-aligned")
                self.alignment_combo.configure(state="disabled")
            else:
                self.alignment_combo.configure(state="readonly")

            # --- layer checkboxes ---
            if fixation == "Macula":
                # cpRNFL is not available on Macula scans
                self.layer_cprnfl_var.set(False)
                self.layer_cprnfl_cb.configure(state="disabled")
                self.layer_gclp_cb.configure(state="normal")
                self.layer_gclpp_cb.configure(state="normal")
                self.layer_retina_cb.configure(state="normal")
            # NOTE: Disc branch kept here for when it is re-enabled in the GUI.
            # elif fixation == "Disc":
            #     self.layer_gclp_var.set(False)
            #     self.layer_gclpp_var.set(False)
            #     self.layer_retina_var.set(False)
            #     self.layer_cprnfl_cb.configure(state="normal")
            #     self.layer_gclp_cb.configure(state="disabled")
            #     self.layer_gclpp_cb.configure(state="disabled")
            #     self.layer_retina_cb.configure(state="disabled")
            else:  # All or 3D Wide — all layers available
                self.layer_cprnfl_cb.configure(state="normal")
                self.layer_gclp_cb.configure(state="normal")
                self.layer_gclpp_cb.configure(state="normal")
                self.layer_retina_cb.configure(state="normal")

        self.fixation_filter_var.trace_add("write", _on_fixation_filter_changed)
        
        # === CONTROL BUTTONS ===
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=4, column=0, columnspan=3, pady=(0, 10))
        
        self.run_button = ttk.Button(
            button_frame,
            text="▶ Run Complete Pipeline",
            command=self.run_pipeline,
            width=25
        )
        self.run_button.grid(row=0, column=0, padx=5)
        
        self.stop_button = ttk.Button(
            button_frame,
            text="⬛ Stop",
            command=self.stop_pipeline,
            state="disabled",
            width=15
        )
        self.stop_button.grid(row=0, column=1, padx=5)
        
        ttk.Button(
            button_frame,
            text="Clear Log",
            command=self.clear_log,
            width=15
        ).grid(row=0, column=2, padx=5)
        
        # === PROGRESS LOG ===
        log_frame = ttk.LabelFrame(main_frame, text="Progress Log", padding="10")
        log_frame.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        main_frame.rowconfigure(5, weight=1)
        
        # Scrolled text widget for output
        self.log_text = scrolledtext.ScrolledText(
            log_frame,
            wrap=tk.WORD,
            width=80,
            height=15,
            font=("Courier", 9)
        )
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        
        # Progress bar
        self.progress_var = tk.StringVar(value="Ready")
        self.progress_label = ttk.Label(main_frame, textvariable=self.progress_var)
        self.progress_label.grid(row=6, column=0, columnspan=3, sticky=tk.W)
    
    def _set_initial_window_size(self):
        """Start smaller on low-resolution displays; full form remains reachable via scrolling."""
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        width = min(900, max(760, screen_w - 120))
        height = min(900, max(650, screen_h - 120))
        self.root.geometry(f"{width}x{height}")
        self.root.minsize(760, 650)

    def _on_main_frame_configure(self, _event=None):
        """Update the canvas scroll region whenever the form size changes."""
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, event):
        """Keep the embedded form width in sync with the visible canvas width."""
        self.canvas.itemconfigure(self._canvas_window, width=event.width)

    def _bind_mousewheel(self, widget):
        widget.bind_all("<MouseWheel>", self._on_mousewheel)
        widget.bind_all("<Button-4>", self._on_mousewheel)
        widget.bind_all("<Button-5>", self._on_mousewheel)

    def _on_mousewheel(self, event):
        """Cross-platform mouse-wheel scrolling for the outer form canvas."""
        if getattr(event, 'num', None) == 4:
            delta = -1
        elif getattr(event, 'num', None) == 5:
            delta = 1
        else:
            raw_delta = getattr(event, 'delta', 0)
            if raw_delta == 0:
                return
            delta = -1 if raw_delta > 0 else 1
        self.canvas.yview_scroll(delta, "units")

    def _normalise_path(self, path_value: str) -> str:
        """Return a Windows/macOS-safe normalised path string without requiring it to exist."""
        if not path_value:
            return ""
        cleaned = path_value.strip().strip('"').strip("'")
        return os.path.normpath(os.path.expanduser(cleaned))

    @contextlib.contextmanager
    def _safe_keep_system_awake(self, enabled: bool, reason: str):
        """Best-effort wrapper so platform-specific keep-awake failures do not stop the pipeline."""
        try:
            with keep_system_awake(enabled, reason=reason):
                yield
        except Exception as exc:
            self.log(f"Warning: keep_system_awake unavailable ({exc}). Continuing without sleep prevention.")
            yield

    def browse_fda_dir(self):
        """Open dialog to select FDA files directory."""
        directory = filedialog.askdirectory(title="Select FDA Files Directory")
        if directory:
            self.fda_dir_var.set(self._normalise_path(directory))
            # Auto-suggest a fully qualified metadata output path so the app does not
            # try to write beside the executable (common Windows permission issue).
            selected_dir = Path(directory)
            dir_name = selected_dir.name
            suggested_name = f"metadata_{dir_name}.csv"
            self.metadata_output_var.set(str(selected_dir / suggested_name))

            # Also pre-fill a sensible results directory if the field is still empty.
            if not self.results_dir_var.get():
                self.results_dir_var.set(str(selected_dir / "results"))
    
    def browse_metadata_output(self):
        """Open dialog to select metadata output file (Option A)."""
        current_value = self._normalise_path(self.metadata_output_var.get())
        initial_dir = str(Path(current_value).parent) if current_value else self._normalise_path(self.fda_dir_var.get()) or os.getcwd()
        initial_file = Path(current_value).name if current_value else "metadata.csv"
        filename = filedialog.asksaveasfilename(
            title="Save Metadata CSV As",
            initialdir=initial_dir,
            initialfile=initial_file,
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            self.metadata_output_var.set(self._normalise_path(filename))
    
    def browse_existing_metadata(self):
        """Open dialog to select existing metadata file (Option B)."""
        filename = filedialog.askopenfilename(
            title="Select Existing Metadata CSV",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            self.existing_metadata_var.set(self._normalise_path(filename))
    
    def browse_results_dir(self):
        """Open dialog to select results output directory."""
        directory = filedialog.askdirectory(title="Select Results Output Directory")
        if directory:
            self.results_dir_var.set(self._normalise_path(directory))
    
    def log(self, message):
        """Add message to log window."""
        if threading.current_thread() is not threading.main_thread():
            self.output_queue.put(str(message))
            return
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.update_idletasks()

    def set_progress(self, text: str):
        """Set progress label text safely from any thread."""
        if threading.current_thread() is threading.main_thread():
            self.progress_var.set(text)
        else:
            # Never call root.after() from a background thread — it is a Tcl
            # call and not thread-safe.  Post a callable to action_queue instead;
            # monitor_output() drains it on the main thread.
            self.action_queue.put(lambda t=text: self.progress_var.set(t))
    
    def clear_log(self):
        """Clear the log window."""
        self.log_text.delete(1.0, tk.END)
    
    def validate_inputs(self):
        """Validate user inputs before running."""
        # Normalise path-like fields first for cross-platform consistency.
        self.fda_dir_var.set(self._normalise_path(self.fda_dir_var.get()))
        self.metadata_output_var.set(self._normalise_path(self.metadata_output_var.get()))
        self.existing_metadata_var.set(self._normalise_path(self.existing_metadata_var.get()))
        self.results_dir_var.set(self._normalise_path(self.results_dir_var.get()))

        # Check metadata source: need EITHER Option A (FDA dir) OR Option B (existing CSV)
        fda_dir = self.fda_dir_var.get()
        existing_metadata = self.existing_metadata_var.get()
        
        if not fda_dir and not existing_metadata:
            messagebox.showerror(
                "Error",
                "Please provide metadata source:\n"
                "Option A: Select FDA Files Directory to extract metadata\n"
                "OR\n"
                "Option B: Select an existing Metadata CSV file"
            )
            return False
        
        if fda_dir and existing_metadata:
            messagebox.showwarning(
                "Warning",
                "Both Option A and Option B are filled.\n"
                "Will use Option B (existing metadata) and skip extraction."
            )
        
        # If using Option A, validate FDA directory
        if fda_dir and not existing_metadata:
            if not Path(fda_dir).exists() or not Path(fda_dir).is_dir():
                messagebox.showerror("Error", f"FDA directory does not exist:\n{fda_dir}")
                return False
            
            # Check metadata output file
            metadata_output = self.metadata_output_var.get()
            if not metadata_output:
                messagebox.showerror("Error", "Please specify where to save extracted metadata (Option A)")
                return False

            try:
                Path(metadata_output).parent.mkdir(parents=True, exist_ok=True)
            except Exception as exc:
                messagebox.showerror("Error", f"Cannot create metadata output folder:\n{Path(metadata_output).parent}\n\n{exc}")
                return False
        
        # If using Option B, validate existing metadata
        if existing_metadata:
            if not Path(existing_metadata).exists():
                messagebox.showerror("Error", f"Metadata file does not exist:\n{existing_metadata}")
                return False
        
        # Check results directory
        results_dir = self.results_dir_var.get()
        if not results_dir:
            messagebox.showerror("Error", "Please specify Results Directory")
            return False

        try:
            Path(results_dir).mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            messagebox.showerror("Error", f"Cannot create results directory:\n{results_dir}\n\n{exc}")
            return False
        
        # Check layer selection - at least one layer must be selected
        if not any([
            self.layer_cprnfl_var.get(),
            self.layer_gclp_var.get(),
            self.layer_gclpp_var.get(),
            self.layer_retina_var.get()
        ]):
            messagebox.showerror("Error", "Please select at least one layer to extract")
            return False
        
        return True
    
    def run_pipeline(self):
        """Run the complete two-stage pipeline."""
        if not self.validate_inputs():
            return

        # Snapshot ALL Tk variable values here on the main thread before spawning
        # the background worker. Tk variables are backed by the Tcl interpreter which
        # is NOT thread-safe; even a .get() call from a background thread can race
        # and cause an intermittent Tcl/Tk crash.
        layers = [
            layer for layer, var in [
                ('cpRNFL', self.layer_cprnfl_var),
                ('GCL+',   self.layer_gclp_var),
                ('GCL++',  self.layer_gclpp_var),
                ('Retina', self.layer_retina_var),
            ] if var.get()
        ]
        params = dict(
            fda_dir=self.fda_dir_var.get(),
            existing_metadata=self.existing_metadata_var.get(),
            metadata_output=self.metadata_output_var.get(),
            results_dir=self.results_dir_var.get(),
            workers=self.workers_var.get(),
            resume=self.resume_var.get(),
            alignment_mode=self.alignment_mode_var.get(),
            output_base_name=self.output_base_name_var.get(),
            fixation_filter=self.fixation_filter_var.get(),
            instrument_filter=self.instrument_filter_var.get(),
            layers=layers,
        )

        # Disable run button, enable stop button
        self._stop_event.clear()
        self.is_running = True
        self.run_button.config(state="disabled")
        self.stop_button.config(state="normal")
        self.set_progress("Running...")

        # Run in separate thread to keep GUI responsive
        thread = threading.Thread(target=self.run_pipeline_thread, args=(params,), daemon=True)
        thread.start()
    
    def run_pipeline_thread(self, params: dict):
        """Execute pipeline in background thread.

        All Tk variable values are passed in via *params* (snapshotted on the
        main thread by run_pipeline) so this method never touches Tk objects.
        """
        try:
            # Unpack pre-snapshotted plain-Python values — no Tk calls here.
            fda_dir           = params['fda_dir']
            existing_metadata = params['existing_metadata']
            metadata_output   = params['metadata_output']
            results_dir       = params['results_dir']
            workers           = params['workers']
            resume            = params['resume']
            alignment_mode    = params['alignment_mode']
            output_base_name  = params['output_base_name']
            fixation_filter   = params['fixation_filter']
            instrument_filter = params['instrument_filter']
            layers            = params['layers']
            layers_str        = ",".join(layers)

            # Determine which metadata file to use for Stage 2
            metadata_file_for_stage2 = None
            
            # === STAGE 1: Metadata Extraction (if needed) ===
            if existing_metadata:
                # Option B: Use existing metadata, skip extraction
                self.log("="*80)
                self.log("STAGE 1: USING EXISTING METADATA")
                self.log("="*80)
                self.log(f"Using existing file: {existing_metadata}")
                self.set_progress("Stage 1: Using existing metadata...")
                metadata_file_for_stage2 = self._normalise_path(existing_metadata)
                self.log("\n✓ Stage 1 complete (skipped extraction)")
            
            elif fda_dir:
                # Option A: Extract metadata from FDA directory
                self.log("="*80)
                self.log("STAGE 1: METADATA EXTRACTION")
                self.log("="*80)
                self.set_progress("Stage 1: Extracting metadata...")
                
                stage1_cmd = [
                    "extract_metadata_batch",
                    fda_dir, "→", metadata_output,
                    f"{workers} workers"
                ]
                self.log(f"Running: extract_metadata_batch({fda_dir!r}, {metadata_output!r}, workers={workers})\n")

                # Run Stage 1 — direct function call with stdout captured to log
                with self._safe_keep_system_awake(True, reason="metadata extraction"):
                    success = self._run_with_stdout_capture(
                        extract_metadata_batch,
                        input_dir=fda_dir,
                        output_csv=metadata_output,
                        n_workers=workers,
                        stop_event=self._stop_event,
                    )
                
                if not success:
                    self.log("\n❌ Stage 1 failed. Pipeline stopped.")
                    self.set_progress("Failed at Stage 1")
                    self.finish_pipeline()
                    return

                # Check stop before file-existence test: if Stage 1 was stopped early
                # it returns normally (no exception) but deliberately skips writing the
                # CSV. Without this check, a stale metadata file from a prior run would
                # pass the existence test below and Stage 2 would run against old data.
                if self._stop_event.is_set():
                    self.log("\n⚠ Stopped during Stage 1. Metadata CSV was not written.")
                    self.set_progress("⚠ Stopped")
                    self.finish_pipeline()
                    return

                # Check if metadata file was created
                if not Path(metadata_output).exists():
                    self.log(f"\n❌ Metadata file not created: {metadata_output}")
                    self.set_progress("Failed: No metadata file")
                    self.finish_pipeline()
                    return
                
                metadata_file_for_stage2 = self._normalise_path(metadata_output)
                self.log("\n✓ Stage 1 complete")
            
            # Check stop_event before starting Stage 2 — the user may have clicked
            # Stop while Stage 1 was running; the stop is honoured here so Stage 2
            # is skipped cleanly rather than submitting all pairs before noticing.
            if self._stop_event.is_set():
                self.log("\n⚠ Stopped after Stage 1. Stage 2 will not run.")
                self.set_progress("Stopped after Stage 1")
                self.finish_pipeline()
                return

            # === STAGE 2: Paired Analysis ===
            self.log("\n" + "="*80)
            self.log("STAGE 2: PAIRED ANALYSIS")
            self.log("="*80)
            self.log("Pairing mode: first_vs_all (baseline vs all follow-ups)")
            self.log(f"Layers: {layers_str}")
            self.log(f"Fixation filter: {fixation_filter}")
            self.log(f"Instrument filter: {instrument_filter}")
            self.set_progress("Stage 2: Processing pairs...")

            self.log(f"Running: run_paired_analysis(layers={layers}, alignment={alignment_mode!r}, resume={resume})\n")

            # Run Stage 2 — direct function call with stdout captured to log
            with self._safe_keep_system_awake(True, reason="paired analysis"):
                success = self._run_with_stdout_capture(
                    run_paired_analysis,
                    metadata_csv=metadata_file_for_stage2,
                    output_dir=results_dir,
                    output_base_name=output_base_name,
                    pairing_mode='first_vs_all',
                    alignment_mode=alignment_mode,
                    layers_to_extract=layers,
                    fixation_filter=fixation_filter,
                    instrument_filter=instrument_filter,
                    n_workers=workers,
                    resume=resume,
                    stop_event=self._stop_event,
                )
            
            if not success:
                self.log("\n❌ Stage 2 failed.")
                self.set_progress("Failed at Stage 2")
                self.finish_pipeline()
                return
            
            # If the user clicked Stop during Stage 2, run_paired_analysis returns
            # normally (early return after printing "Stopped early"). Detect this
            # here so the GUI reports "Stopped" rather than "Complete".
            if self._stop_event.is_set():
                self.log("\n⚠ Stage 2 stopped early by user request.")
                self.set_progress("⚠ Stopped")
                self.finish_pipeline()
                return

            self.log("\n✓ Stage 2 complete")
            
            # === COMPLETE ===
            self.log("\n" + "="*80)
            self.log("✅ PIPELINE COMPLETE")
            self.log("="*80)
            self.log(f"\nOutput files:")
            self.log(f"  Metadata: {metadata_file_for_stage2}")
            self.log(f"  Results: {results_dir}/")
            self.set_progress("✅ Complete!")
            
            # Show success message
            success_msg = f"Pipeline completed successfully!\n\nResults saved to:\n{results_dir}"
            self.action_queue.put(lambda m=success_msg: messagebox.showinfo("Success", m))
        
        except Exception as e:
            self.log(f"\n❌ Unexpected error: {e}")
            import traceback
            self.log(traceback.format_exc())
            self.set_progress("Error")
            err_str = str(e)
            self.action_queue.put(lambda m=err_str: messagebox.showerror("Error", f"Unexpected error:\n{m}"))
        
        finally:
            self.finish_pipeline()
    
    def _run_with_stdout_capture(self, fn, *args, **kwargs) -> bool:
        """Call *fn* with sys.stdout redirected to the log queue.

        Returns True on success, False if the function raises or exits non-zero.
        Restores sys.stdout unconditionally via finally.
        """
        writer = _QueueWriter(self.output_queue)
        old_stdout = sys.stdout
        sys.stdout = writer
        try:
            fn(*args, **kwargs)
            return True
        except SystemExit as e:
            writer.flush()
            return (e.code == 0 or e.code is None)
        except Exception as e:
            writer.flush()
            import traceback
            self.output_queue.put(f"\nError: {type(e).__name__}: {e}")
            self.output_queue.put(traceback.format_exc())
            return False
        finally:
            sys.stdout = old_stdout
    
    def monitor_output(self):
        """Drain the output and action queues; runs exclusively on the main thread."""
        # Drain log messages from pipeline workers.
        try:
            while True:
                line = self.output_queue.get_nowait()
                self.log(line)
        except queue.Empty:
            pass

        # Drain deferred main-thread actions posted by background threads.
        # Using a queue instead of root.after() from worker threads is the only
        # truly thread-safe way to call Tk methods from non-main threads.
        try:
            while True:
                fn = self.action_queue.get_nowait()
                fn()
        except queue.Empty:
            pass

        # Schedule next check
        self.root.after(100, self.monitor_output)
    
    def stop_pipeline(self):
        """Stop the running pipeline."""
        if self.is_running:
            response = messagebox.askyesno(
                "Confirm Stop",
                "Are you sure you want to stop the pipeline?\n"
                "Progress will be saved to checkpoint.\n\n"
                "Note: the current scan/batch will finish before halting."
            )

            if response:
                # Only signal the stop; do NOT set is_running=False here.
                # is_running stays True so on_close keeps the "still running"
                # warning active until finish_pipeline() is called by the thread.
                self._stop_event.set()
                self.log("\n⚠ Stop requested — waiting for current task to finish...")
                self.set_progress("Stopping...")
    
    def on_close(self):
        """Handle window close button."""
        if self.is_running:
            response = messagebox.askyesno(
                "Pipeline Running",
                "The pipeline is still running.\n\n"
                "Force quit now? Checkpoints are saved up to the last completed pair."
            )
            if response:
                os._exit(0)
            # else: leave window open
        else:
            self.root.destroy()
    
    def finish_pipeline(self):
        """Re-enable buttons after pipeline completes (must run on main thread)."""
        if threading.current_thread() is not threading.main_thread():
            # Post to action_queue so the main thread runs this — never call
            # root.after() directly from a background thread.
            self.action_queue.put(self.finish_pipeline)
            return
        self.is_running = False
        self.run_button.config(state="normal")
        self.stop_button.config(state="disabled")


def main():
    """Launch the GUI application."""
    root = tk.Tk()
    app = OCTAnalysisGUI(root)
    
    # Bring window to front on macOS
    root.lift()
    root.attributes('-topmost', True)
    root.after(100, lambda: root.attributes('-topmost', False))
    root.focus_force()
    
    root.mainloop()


if __name__ == '__main__':
    multiprocessing.freeze_support()  # Required for PyInstaller + multiprocessing on macOS
    main()
