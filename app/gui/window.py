"""Tk-based desktop window for controlling the project without a console."""

from __future__ import annotations

import os
import sys
import time
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, ttk

from app.core.logging import configure_logging, get_logger
from app.core.runtime import get_config_file, get_log_file, get_runtime_dir
from app.core.version import VERSION, VERSION_TAG
from app.gui.config import GuiConfig, GuiConfigStore
from app.gui.i18n import translate
from app.gui.tasks import (
    ApiServerController,
    BackgroundTaskRunner,
    TaskResult,
    run_evaluation,
    run_inference,
    run_prepare_dataset,
    run_training,
)
from app.models.task import TaskType

LOGGER = get_logger(__name__)


def parse_positive_int(value: str, fallback: int) -> int:
    """Safely parse a positive integer while the user edits text fields."""
    try:
        return max(int(value.strip() or str(fallback)), 1)
    except (TypeError, ValueError):
        return fallback


class LLMStudioWindow(tk.Tk):
    """Main desktop window that exposes the project in a GUI-first workflow."""

    def __init__(self) -> None:
        # Configure file logging before the UI starts so startup issues are captured.
        configure_logging(include_console=False)
        super().__init__()

        self.store = GuiConfigStore()
        self.gui_config = self.store.load()
        self.task_runner = BackgroundTaskRunner()
        self.server_controller = ApiServerController()
        self._geometry_save_job: str | None = None
        self._closing_after_task = False
        self._last_activity = time.monotonic()

        self._build_state()
        self._configure_window()
        self._configure_style()
        self._build_menu()
        self._build_layout()
        self._bind_events()
        self._apply_translations()
        self._restore_geometry()
        self._refresh_log_view()
        self._sync_backend_badge()

        # Poll background work and countdown from the Tk event loop.
        self.after(200, self._poll_background_results)
        self.after(1000, self._tick_auto_close)

        if self.auto_start_var.get():
            self.after(300, self.start_api)

    def _build_state(self) -> None:
        """Create Tk variables so the GUI can auto-save state changes."""
        self.language_var = tk.StringVar(value=self.gui_config.language)
        self.auto_start_var = tk.BooleanVar(value=self.gui_config.auto_start_backend)
        self.auto_close_var = tk.BooleanVar(value=self.gui_config.auto_close_enabled)
        self.auto_close_seconds_var = tk.StringVar(value=str(self.gui_config.auto_close_seconds))
        self.host_var = tk.StringVar(value=self.gui_config.host)
        self.port_var = tk.StringVar(value=str(self.gui_config.port))
        self.dataset_input_var = tk.StringVar(value=self.gui_config.dataset_input_path)
        self.dataset_output_var = tk.StringVar(value=self.gui_config.dataset_output_path)
        self.training_config_var = tk.StringVar(value=self.gui_config.training_config_path)
        self.evaluation_config_var = tk.StringVar(value=self.gui_config.evaluation_config_path)
        self.inference_task_var = tk.StringVar(value=self.gui_config.inference_task)
        self.inference_language_var = tk.StringVar(value=self.gui_config.inference_language)
        self.inference_context_file_var = tk.StringVar(value=self.gui_config.inference_context_file)
        self.status_var = tk.StringVar(value="")
        self.countdown_var = tk.StringVar(value="")
        self.backend_state_var = tk.StringVar(value="")
        self.runtime_dir_var = tk.StringVar(value=str(get_runtime_dir()))
        self.config_path_var = tk.StringVar(value=str(get_config_file()))
        self.log_path_var = tk.StringVar(value=str(get_log_file()))
        self.version_var = tk.StringVar(value=VERSION_TAG)

    def _configure_window(self) -> None:
        """Configure top-level window metadata and close behavior."""
        self.title("DeepSeek Coder Studio")
        self.minsize(1100, 760)
        self.protocol("WM_DELETE_WINDOW", self.request_close)

    def _configure_style(self) -> None:
        """Create a custom ttk look that feels like a real desktop control center."""
        style = ttk.Style(self)
        style.theme_use("clam")

        self.configure(background="#f4efe6")
        style.configure(".", font=("Segoe UI", 10), background="#f4efe6", foreground="#1f2937")
        style.configure("Card.TFrame", background="#fffdf8", relief="flat")
        style.configure("Section.TLabelframe", background="#fffdf8", bordercolor="#d9cdb8")
        style.configure("Section.TLabelframe.Label", background="#fffdf8", foreground="#345c72", font=("Segoe UI Semibold", 10))
        style.configure("Title.TLabel", background="#f4efe6", foreground="#0f3557", font=("Segoe UI Semibold", 24))
        style.configure("Subtitle.TLabel", background="#f4efe6", foreground="#5b6470", font=("Segoe UI", 10))
        style.configure("Accent.TButton", background="#1f6f8b", foreground="#ffffff", padding=(14, 10), borderwidth=0)
        style.map("Accent.TButton", background=[("active", "#16596f")])
        style.configure("Secondary.TButton", background="#dfe7ec", foreground="#153243", padding=(12, 10), borderwidth=0)
        style.map("Secondary.TButton", background=[("active", "#ced8df")])
        style.configure("Status.Horizontal.TProgressbar", troughcolor="#e7e1d6", background="#1f6f8b")
        style.configure("TNotebook", background="#f4efe6", borderwidth=0)
        style.configure("TNotebook.Tab", padding=(18, 10), background="#dfd5c4", foreground="#334155")
        style.map("TNotebook.Tab", background=[("selected", "#fffdf8")])

    def _build_menu(self) -> None:
        """Create a Windows-friendly menu with accelerators and About dialog access."""
        self.menu_bar = tk.Menu(self)
        self.file_menu = tk.Menu(self.menu_bar, tearoff=False)
        self.language_menu = tk.Menu(self.menu_bar, tearoff=False)
        self.help_menu = tk.Menu(self.menu_bar, tearoff=False)

        self.file_menu.add_command(label=self._t("menu_exit"), accelerator="Ctrl+Q", command=self.request_close)
        self.language_menu.add_radiobutton(label=self._t("language_es"), value="es", variable=self.language_var, command=self._on_language_changed)
        self.language_menu.add_radiobutton(label=self._t("language_en"), value="en", variable=self.language_var, command=self._on_language_changed)
        self.help_menu.add_command(label=self._t("menu_about"), command=self.open_about_dialog)

        self.menu_bar.add_cascade(label=self._t("menu_file"), menu=self.file_menu)
        self.menu_bar.add_cascade(label=self._t("menu_language"), menu=self.language_menu)
        self.menu_bar.add_cascade(label=self._t("menu_help"), menu=self.help_menu)
        self.config(menu=self.menu_bar)

    def _build_layout(self) -> None:
        """Build header, toolbar, notebook and status bar."""
        root = ttk.Frame(self, padding=(18, 18, 18, 12))
        root.pack(fill="both", expand=True)

        self.header_title = ttk.Label(root, style="Title.TLabel")
        self.header_title.pack(anchor="w")
        self.header_subtitle = ttk.Label(root, style="Subtitle.TLabel")
        self.header_subtitle.pack(anchor="w", pady=(4, 14))

        toolbar = ttk.Frame(root)
        toolbar.pack(fill="x", pady=(0, 12))

        self.start_api_button = ttk.Button(toolbar, style="Accent.TButton", command=self.start_api)
        self.start_api_button.pack(side="left", padx=(0, 8))
        self.stop_api_button = ttk.Button(toolbar, style="Secondary.TButton", command=self.stop_api)
        self.stop_api_button.pack(side="left", padx=(0, 8))
        self.health_button = ttk.Button(toolbar, style="Secondary.TButton", command=self.check_api_health)
        self.health_button.pack(side="left", padx=(0, 8))
        self.open_log_button = ttk.Button(toolbar, style="Secondary.TButton", command=self.open_log_file)
        self.open_log_button.pack(side="left", padx=(0, 8))
        self.open_config_button = ttk.Button(toolbar, style="Secondary.TButton", command=self.open_config_file)
        self.open_config_button.pack(side="left", padx=(0, 8))
        self.exit_button = ttk.Button(toolbar, style="Secondary.TButton", command=self.request_close)
        self.exit_button.pack(side="right")

        preference_card = ttk.LabelFrame(root, style="Section.TLabelframe", padding=14)
        preference_card.pack(fill="x", pady=(0, 12))
        self.preference_card = preference_card

        ttk.Checkbutton(preference_card, variable=self.auto_start_var).grid(row=0, column=0, sticky="w", padx=(0, 16))
        self.auto_start_label = ttk.Label(preference_card)
        self.auto_start_label.grid(row=0, column=1, sticky="w")

        ttk.Checkbutton(preference_card, variable=self.auto_close_var).grid(row=0, column=2, sticky="w", padx=(24, 16))
        self.auto_close_label = ttk.Label(preference_card)
        self.auto_close_label.grid(row=0, column=3, sticky="w")

        self.auto_close_seconds_label = ttk.Label(preference_card)
        self.auto_close_seconds_label.grid(row=0, column=4, sticky="w", padx=(24, 10))
        ttk.Entry(preference_card, width=8, textvariable=self.auto_close_seconds_var).grid(row=0, column=5, sticky="w")

        self.host_label = ttk.Label(preference_card)
        self.host_label.grid(row=1, column=0, sticky="w", pady=(10, 0))
        ttk.Entry(preference_card, width=22, textvariable=self.host_var).grid(row=1, column=1, sticky="w", pady=(10, 0))
        self.port_label = ttk.Label(preference_card)
        self.port_label.grid(row=1, column=2, sticky="w", padx=(24, 10), pady=(10, 0))
        ttk.Entry(preference_card, width=10, textvariable=self.port_var).grid(row=1, column=3, sticky="w", pady=(10, 0))

        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill="both", expand=True)

        self.dashboard_tab = ttk.Frame(self.notebook, padding=14)
        self.operations_tab = ttk.Frame(self.notebook, padding=14)
        self.assistant_tab = ttk.Frame(self.notebook, padding=14)
        self.logs_tab = ttk.Frame(self.notebook, padding=14)
        self.notebook.add(self.dashboard_tab, text="")
        self.notebook.add(self.operations_tab, text="")
        self.notebook.add(self.assistant_tab, text="")
        self.notebook.add(self.logs_tab, text="")

        self._build_dashboard_tab()
        self._build_operations_tab()
        self._build_assistant_tab()
        self._build_logs_tab()

        status_bar = ttk.Frame(root)
        status_bar.pack(fill="x", pady=(10, 0))
        self.status_label = ttk.Label(status_bar, textvariable=self.status_var)
        self.status_label.pack(side="left", fill="x", expand=True)
        self.progress = ttk.Progressbar(status_bar, style="Status.Horizontal.TProgressbar", mode="indeterminate", length=150)
        self.progress.pack(side="left", padx=10)
        self.countdown_label = ttk.Label(status_bar, textvariable=self.countdown_var)
        self.countdown_label.pack(side="right")

        self._select_saved_tab()

    def _build_dashboard_tab(self) -> None:
        """Create a quick runtime summary with the most important app details."""
        runtime_card = ttk.LabelFrame(self.dashboard_tab, style="Section.TLabelframe", padding=16)
        runtime_card.pack(fill="x", pady=(0, 12))
        self.runtime_card = runtime_card

        self.runtime_dir_label = ttk.Label(runtime_card)
        self.runtime_dir_label.grid(row=0, column=0, sticky="w")
        ttk.Label(runtime_card, textvariable=self.runtime_dir_var).grid(row=0, column=1, sticky="w", padx=(16, 0))

        self.config_path_label = ttk.Label(runtime_card)
        self.config_path_label.grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Label(runtime_card, textvariable=self.config_path_var).grid(row=1, column=1, sticky="w", padx=(16, 0), pady=(8, 0))

        self.log_path_label = ttk.Label(runtime_card)
        self.log_path_label.grid(row=2, column=0, sticky="w", pady=(8, 0))
        ttk.Label(runtime_card, textvariable=self.log_path_var).grid(row=2, column=1, sticky="w", padx=(16, 0), pady=(8, 0))

        self.version_label = ttk.Label(runtime_card)
        self.version_label.grid(row=3, column=0, sticky="w", pady=(8, 0))
        ttk.Label(runtime_card, textvariable=self.version_var).grid(row=3, column=1, sticky="w", padx=(16, 0), pady=(8, 0))

        backend_card = ttk.LabelFrame(self.dashboard_tab, style="Section.TLabelframe", padding=16)
        backend_card.pack(fill="x")
        self.backend_card = backend_card
        self.backend_state_label = ttk.Label(backend_card)
        self.backend_state_label.grid(row=0, column=0, sticky="w")
        ttk.Label(backend_card, textvariable=self.backend_state_var).grid(row=0, column=1, sticky="w", padx=(16, 0))

    def _build_operations_tab(self) -> None:
        """Create controls for dataset, training and evaluation workflows."""
        dataset_card = ttk.LabelFrame(self.operations_tab, style="Section.TLabelframe", padding=16)
        dataset_card.pack(fill="x", pady=(0, 12))
        self.dataset_card = dataset_card

        self.dataset_input_label = ttk.Label(dataset_card)
        self.dataset_input_label.grid(row=0, column=0, sticky="w")
        ttk.Entry(dataset_card, textvariable=self.dataset_input_var, width=82).grid(row=0, column=1, padx=10, sticky="ew")
        self.dataset_input_browse_button = ttk.Button(dataset_card, style="Secondary.TButton", command=self.browse_dataset_input)
        self.dataset_input_browse_button.grid(row=0, column=2, sticky="e")

        self.dataset_output_label = ttk.Label(dataset_card)
        self.dataset_output_label.grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(dataset_card, textvariable=self.dataset_output_var, width=82).grid(row=1, column=1, padx=10, pady=(8, 0), sticky="ew")
        self.dataset_output_browse_button = ttk.Button(dataset_card, style="Secondary.TButton", command=self.browse_dataset_output)
        self.dataset_output_browse_button.grid(row=1, column=2, pady=(8, 0), sticky="e")

        self.prepare_dataset_button = ttk.Button(dataset_card, style="Accent.TButton", command=self.prepare_dataset_job)
        self.prepare_dataset_button.grid(row=2, column=1, pady=(12, 0), sticky="w")

        training_card = ttk.LabelFrame(self.operations_tab, style="Section.TLabelframe", padding=16)
        training_card.pack(fill="x", pady=(0, 12))
        self.training_card = training_card

        self.training_config_label = ttk.Label(training_card)
        self.training_config_label.grid(row=0, column=0, sticky="w")
        ttk.Entry(training_card, textvariable=self.training_config_var, width=82).grid(row=0, column=1, padx=10, sticky="ew")
        self.training_config_browse_button = ttk.Button(training_card, style="Secondary.TButton", command=self.browse_training_config)
        self.training_config_browse_button.grid(row=0, column=2, sticky="e")

        self.training_button = ttk.Button(training_card, style="Accent.TButton", command=self.train_model_job)
        self.training_button.grid(row=1, column=1, pady=(12, 0), sticky="w")

        evaluation_card = ttk.LabelFrame(self.operations_tab, style="Section.TLabelframe", padding=16)
        evaluation_card.pack(fill="x")
        self.evaluation_card = evaluation_card

        self.evaluation_config_label = ttk.Label(evaluation_card)
        self.evaluation_config_label.grid(row=0, column=0, sticky="w")
        ttk.Entry(evaluation_card, textvariable=self.evaluation_config_var, width=82).grid(row=0, column=1, padx=10, sticky="ew")
        self.evaluation_config_browse_button = ttk.Button(evaluation_card, style="Secondary.TButton", command=self.browse_evaluation_config)
        self.evaluation_config_browse_button.grid(row=0, column=2, sticky="e")

        self.evaluation_button = ttk.Button(evaluation_card, style="Accent.TButton", command=self.evaluate_model_job)
        self.evaluation_button.grid(row=1, column=1, pady=(12, 0), sticky="w")

    def _build_assistant_tab(self) -> None:
        """Create local inference controls and result viewers."""
        assistant_card = ttk.LabelFrame(self.assistant_tab, style="Section.TLabelframe", padding=16)
        assistant_card.pack(fill="both", expand=True)
        self.assistant_card = assistant_card

        self.inference_task_label = ttk.Label(assistant_card)
        self.inference_task_label.grid(row=0, column=0, sticky="w")
        ttk.Combobox(
            assistant_card,
            textvariable=self.inference_task_var,
            values=[task.value for task in TaskType],
            state="readonly",
            width=24,
        ).grid(row=0, column=1, sticky="w", padx=(10, 20))

        self.inference_language_label = ttk.Label(assistant_card)
        self.inference_language_label.grid(row=0, column=2, sticky="w")
        ttk.Entry(assistant_card, textvariable=self.inference_language_var, width=18).grid(row=0, column=3, sticky="w", padx=(10, 0))

        self.inference_context_label = ttk.Label(assistant_card)
        self.inference_context_label.grid(row=1, column=0, sticky="w", pady=(12, 0))
        ttk.Entry(assistant_card, textvariable=self.inference_context_file_var, width=82).grid(row=1, column=1, columnspan=2, padx=(10, 10), pady=(12, 0), sticky="ew")
        self.inference_context_browse_button = ttk.Button(assistant_card, style="Secondary.TButton", command=self.browse_inference_context)
        self.inference_context_browse_button.grid(row=1, column=3, pady=(12, 0), sticky="e")

        self.prompt_label = ttk.Label(assistant_card)
        self.prompt_label.grid(row=2, column=0, sticky="nw", pady=(14, 6))
        self.prompt_text = tk.Text(
            assistant_card,
            wrap="word",
            height=12,
            font=("Consolas", 11),
            bg="#fffdf8",
            fg="#1f2937",
            insertbackground="#1f2937",
            relief="flat",
        )
        self.prompt_text.grid(row=2, column=1, columnspan=3, sticky="nsew", pady=(14, 6))
        self.prompt_text.insert("1.0", self.gui_config.inference_prompt)

        button_row = ttk.Frame(assistant_card)
        button_row.grid(row=3, column=1, columnspan=3, sticky="w", pady=(0, 12))
        self.run_assistant_button = ttk.Button(button_row, style="Accent.TButton", command=self.run_assistant_job)
        self.run_assistant_button.pack(side="left")
        self.clear_prompt_button = ttk.Button(button_row, style="Secondary.TButton", command=self.clear_prompt)
        self.clear_prompt_button.pack(side="left", padx=(8, 8))
        self.copy_output_button = ttk.Button(button_row, style="Secondary.TButton", command=self.copy_output)
        self.copy_output_button.pack(side="left")

        self.output_label = ttk.Label(assistant_card)
        self.output_label.grid(row=4, column=0, sticky="nw")
        self.output_text = tk.Text(
            assistant_card,
            wrap="word",
            height=12,
            font=("Consolas", 11),
            bg="#fdf7ed",
            fg="#1f2937",
            insertbackground="#1f2937",
            relief="flat",
        )
        self.output_text.grid(row=4, column=1, columnspan=3, sticky="nsew")
        self.output_text.configure(state="disabled")

        assistant_card.columnconfigure(1, weight=1)
        assistant_card.rowconfigure(2, weight=1)
        assistant_card.rowconfigure(4, weight=1)

    def _build_logs_tab(self) -> None:
        """Create a lightweight log tail viewer for non-intrusive diagnostics."""
        logs_card = ttk.LabelFrame(self.logs_tab, style="Section.TLabelframe", padding=16)
        logs_card.pack(fill="both", expand=True)
        self.logs_card = logs_card

        toolbar = ttk.Frame(logs_card)
        toolbar.pack(fill="x", pady=(0, 10))
        self.refresh_logs_button = ttk.Button(toolbar, style="Secondary.TButton", command=self._refresh_log_view)
        self.refresh_logs_button.pack(side="left")

        self.log_text = tk.Text(
            logs_card,
            wrap="none",
            font=("Consolas", 10),
            bg="#0f172a",
            fg="#e2e8f0",
            insertbackground="#e2e8f0",
            relief="flat",
        )
        self.log_text.pack(fill="both", expand=True)
        self.log_text.configure(state="disabled")

    def _bind_events(self) -> None:
        """Bind autosave, accelerators and interaction tracking."""
        tracked_vars = [
            self.language_var,
            self.auto_start_var,
            self.auto_close_var,
            self.auto_close_seconds_var,
            self.host_var,
            self.port_var,
            self.dataset_input_var,
            self.dataset_output_var,
            self.training_config_var,
            self.evaluation_config_var,
            self.inference_task_var,
            self.inference_language_var,
            self.inference_context_file_var,
        ]
        for variable in tracked_vars:
            variable.trace_add("write", self._on_variable_changed)

        # Save prompt changes without forcing the user to click a save button.
        self.prompt_text.bind("<KeyRelease>", self._on_prompt_changed)
        self.bind("<Configure>", self._on_window_configure)
        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)

        # Windows-style shortcuts for frequent actions.
        self.bind_all("<Control-q>", lambda _: self.request_close())
        self.bind_all("<F5>", lambda _: self.start_api())
        self.bind_all("<Shift-F5>", lambda _: self.stop_api())
        self.bind_all("<Control-Return>", lambda _: self.run_assistant_job())

        # Any interaction resets the inactivity timer used by auto-close.
        self.bind_all("<Any-KeyPress>", self._record_activity, add="+")
        self.bind_all("<Any-ButtonPress>", self._record_activity, add="+")

    def _apply_translations(self) -> None:
        """Refresh all user-facing labels after a language change."""
        self.title(self._t("app_title"))
        self.header_title.configure(text=self._t("app_title"))
        self.header_subtitle.configure(text=self._t("subtitle"))
        self._build_menu()

        self.start_api_button.configure(text=self._t("toolbar_start_api"))
        self.stop_api_button.configure(text=self._t("toolbar_stop_api"))
        self.health_button.configure(text=self._t("toolbar_health"))
        self.open_log_button.configure(text=self._t("toolbar_open_log"))
        self.open_config_button.configure(text=self._t("toolbar_open_config"))
        self.exit_button.configure(text=self._t("toolbar_exit"))

        self.preference_card.configure(text=self._t("settings_title"))
        self.auto_start_label.configure(text=self._t("auto_start_backend"))
        self.auto_close_label.configure(text=self._t("auto_close_enabled"))
        self.auto_close_seconds_label.configure(text=self._t("auto_close_seconds"))
        self.host_label.configure(text=self._t("host"))
        self.port_label.configure(text=self._t("port"))

        self.notebook.tab(self.dashboard_tab, text=self._t("tab_dashboard"))
        self.notebook.tab(self.operations_tab, text=self._t("tab_operations"))
        self.notebook.tab(self.assistant_tab, text=self._t("tab_assistant"))
        self.notebook.tab(self.logs_tab, text=self._t("tab_logs"))

        self.runtime_card.configure(text=self._t("card_runtime"))
        self.runtime_dir_label.configure(text=self._t("runtime_dir"))
        self.config_path_label.configure(text=self._t("config_path"))
        self.log_path_label.configure(text=self._t("log_path"))
        self.version_label.configure(text=self._t("version"))

        self.backend_card.configure(text=self._t("card_backend"))
        self.backend_state_label.configure(text=self._t("backend_state"))

        self.dataset_card.configure(text=self._t("dataset_group"))
        self.dataset_input_label.configure(text=self._t("input_path"))
        self.dataset_output_label.configure(text=self._t("output_path"))
        self.dataset_input_browse_button.configure(text=self._t("browse_file"))
        self.dataset_output_browse_button.configure(text=self._t("browse_dir"))
        self.prepare_dataset_button.configure(text=self._t("run_prepare_dataset"))

        self.training_card.configure(text=self._t("training_group"))
        self.training_config_label.configure(text=self._t("training_config"))
        self.training_config_browse_button.configure(text=self._t("browse_file"))
        self.training_button.configure(text=self._t("run_training"))

        self.evaluation_card.configure(text=self._t("evaluation_group"))
        self.evaluation_config_label.configure(text=self._t("evaluation_config"))
        self.evaluation_config_browse_button.configure(text=self._t("browse_file"))
        self.evaluation_button.configure(text=self._t("run_evaluation"))

        self.assistant_card.configure(text=self._t("assistant_group"))
        self.inference_task_label.configure(text=self._t("task"))
        self.inference_language_label.configure(text=self._t("language"))
        self.inference_context_label.configure(text=self._t("context_file"))
        self.prompt_label.configure(text=self._t("prompt"))
        self.run_assistant_button.configure(text=self._t("run_assistant"))
        self.clear_prompt_button.configure(text=self._t("clear_prompt"))
        self.copy_output_button.configure(text=self._t("copy_output"))
        self.output_label.configure(text=self._t("assistant_output"))

        self.logs_card.configure(text=self._t("logs_group"))
        self.refresh_logs_button.configure(text=self._t("refresh_logs"))

        self._sync_backend_badge()
        self._update_countdown_label()
        if not self.status_var.get():
            self._set_status(self._t("status_ready"))

    def _restore_geometry(self) -> None:
        """Restore the last persisted window size and position."""
        try:
            self.geometry(self.gui_config.window_geometry)
        except tk.TclError:
            LOGGER.warning("Invalid saved geometry: %s", self.gui_config.window_geometry)

    def _select_saved_tab(self) -> None:
        """Restore the last active notebook tab."""
        tabs = {
            "dashboard": self.dashboard_tab,
            "operations": self.operations_tab,
            "assistant": self.assistant_tab,
            "logs": self.logs_tab,
        }
        target = tabs.get(self.gui_config.selected_tab, self.dashboard_tab)
        self.notebook.select(target)

    def _on_variable_changed(self, *_: object) -> None:
        """Persist config changes and react to language changes."""
        self._record_activity()
        self._save_config()

    def _on_language_changed(self) -> None:
        """Repaint labels after the selected language changes."""
        self._apply_translations()
        self._save_config()

    def _on_prompt_changed(self, _: tk.Event[tk.Text]) -> None:
        """Persist prompt text as the user types."""
        self._record_activity()
        self._save_config()

    def _on_window_configure(self, event: tk.Event[tk.Misc]) -> None:
        """Debounce geometry persistence while the user resizes the window."""
        if event.widget is not self:
            return
        if self._geometry_save_job is not None:
            self.after_cancel(self._geometry_save_job)
        self._geometry_save_job = self.after(250, self._save_config)

    def _on_tab_changed(self, _: tk.Event[ttk.Notebook]) -> None:
        """Persist the selected tab to reopen the same section next time."""
        self._save_config()

    def _collect_config(self) -> GuiConfig:
        """Collect all persisted GUI settings from Tk variables and widgets."""
        return GuiConfig(
            window_geometry=self.geometry(),
            language=self.language_var.get(),
            auto_start_backend=self.auto_start_var.get(),
            auto_close_enabled=self.auto_close_var.get(),
            # Preserve autosave even while the user is midway through editing numbers.
            auto_close_seconds=parse_positive_int(
                self.auto_close_seconds_var.get(),
                self.gui_config.auto_close_seconds,
            ),
            host=self.host_var.get().strip() or "127.0.0.1",
            port=parse_positive_int(self.port_var.get(), self.gui_config.port),
            dataset_input_path=self.dataset_input_var.get().strip(),
            dataset_output_path=self.dataset_output_var.get().strip(),
            training_config_path=self.training_config_var.get().strip(),
            evaluation_config_path=self.evaluation_config_var.get().strip(),
            inference_task=self.inference_task_var.get().strip(),
            inference_language=self.inference_language_var.get().strip(),
            inference_prompt=self.prompt_text.get("1.0", "end").strip(),
            inference_context_file=self.inference_context_file_var.get().strip(),
            selected_tab=self._current_tab_key(),
        )

    def _save_config(self) -> None:
        """Persist the full GUI state to config.json."""
        try:
            self.gui_config = self._collect_config()
            self.store.save(self.gui_config)
        except Exception:  # noqa: BLE001
            LOGGER.exception("Failed to persist GUI config")

    def _current_tab_key(self) -> str:
        """Map the selected notebook page to its persistence key."""
        selected = self.notebook.select()
        mapping = {
            str(self.dashboard_tab): "dashboard",
            str(self.operations_tab): "operations",
            str(self.assistant_tab): "assistant",
            str(self.logs_tab): "logs",
        }
        return mapping.get(selected, "dashboard")

    def _record_activity(self, *_: object) -> None:
        """Reset the inactivity timer used by auto-close."""
        self._last_activity = time.monotonic()

    def _tick_auto_close(self) -> None:
        """Update the visible countdown and close silently after inactivity."""
        if self.auto_close_var.get():
            timeout_seconds = parse_positive_int(
                self.auto_close_seconds_var.get(),
                self.gui_config.auto_close_seconds,
            )
            remaining = timeout_seconds - int(time.monotonic() - self._last_activity)
            if remaining <= 0:
                if self.task_runner.has_active_tasks():
                    # Avoid killing long-running training or evaluation jobs mid-flight.
                    self._record_activity()
                    self._set_status(self._t("status_busy"), level="warning")
                else:
                    self._closing_after_task = True
                    self.request_close()
                    return
            self.countdown_var.set(self._t("countdown_active", seconds=max(remaining, 0)))
        else:
            self.countdown_var.set(self._t("countdown_disabled"))
        self.after(1000, self._tick_auto_close)

    def _poll_background_results(self) -> None:
        """Consume worker results and update the GUI safely from the main thread."""
        while not self.task_runner.results.empty():
            result = self.task_runner.results.get_nowait()
            self._handle_background_result(result)

        if self.task_runner.has_active_tasks():
            self.progress.start(8)
        else:
            self.progress.stop()
        self.after(200, self._poll_background_results)

    def _handle_background_result(self, result: TaskResult) -> None:
        """Render the result of a background action."""
        if not result.success:
            self._set_status(self._t("status_error", message=result.error_message or "Unknown error"), level="error")
            return

        if result.name == "start_api":
            url = result.payload["url"]
            message_key = "status_api_external" if result.payload["status"] == "external" else "status_api_running"
            self._set_status(self._t(message_key, url=url))
            self._sync_backend_badge()
            return

        if result.name == "stop_api":
            self._set_status(self._t("status_api_stopped"))
            self._sync_backend_badge()
            if self._closing_after_task:
                self.destroy()
            return

        if result.name == "check_api":
            if result.payload["reachable"]:
                self._set_status(self._t("status_api_running", url=result.payload["url"]))
            else:
                self._set_status(self._t("status_api_unavailable"), level="warning")
            self._sync_backend_badge()
            return

        if result.name == "prepare_dataset":
            splits = result.payload["splits"]
            self._set_status(
                self._t(
                    "status_prepare_success",
                    train=splits["train"],
                    validation=splits["validation"],
                    test=splits["test"],
                )
            )
            return

        if result.name == "train_model":
            self._set_status(self._t("status_training_success", path=result.payload["adapter_output_dir"]))
            return

        if result.name == "evaluate_model":
            score = result.payload["summary"]["average_score"]
            self._set_status(self._t("status_evaluation_success", score=score))
            return

        if result.name == "run_inference":
            self._set_output_text(result.payload["output_text"])
            self._set_status(self._t("status_inference_success", latency=result.payload["latency_ms"]))
            return

    def _set_output_text(self, value: str) -> None:
        """Replace the assistant output text safely."""
        self.output_text.configure(state="normal")
        self.output_text.delete("1.0", "end")
        self.output_text.insert("1.0", value)
        self.output_text.configure(state="disabled")

    def _set_status(self, message: str, level: str = "info") -> None:
        """Update the status bar and log the same message with an appropriate level."""
        self.status_var.set(message)
        foreground = {"info": "#1f2937", "warning": "#7c4f00", "error": "#8b1e3f"}.get(level, "#1f2937")
        self.status_label.configure(foreground=foreground)
        if level == "error":
            LOGGER.error(message)
        elif level == "warning":
            LOGGER.warning(message)
        else:
            LOGGER.info(message)

    def _sync_backend_badge(self) -> None:
        """Refresh the visible backend state in the dashboard."""
        port = parse_positive_int(self.port_var.get(), self.gui_config.port)
        health = self.server_controller.health(self.host_var.get().strip() or "127.0.0.1", port)
        if health["reachable"]:
            self.backend_state_var.set(health["url"])
        else:
            self.backend_state_var.set(self._t("backend_offline"))

    def _update_countdown_label(self) -> None:
        """Refresh the countdown label after translations change."""
        if self.auto_close_var.get():
            self.countdown_var.set(
                self._t(
                    "countdown_active",
                    seconds=parse_positive_int(
                        self.auto_close_seconds_var.get(),
                        self.gui_config.auto_close_seconds,
                    ),
                )
            )
        else:
            self.countdown_var.set(self._t("countdown_disabled"))

    def _refresh_log_view(self) -> None:
        """Load the latest log tail into the log tab."""
        log_file = get_log_file()
        if log_file.exists():
            lines = log_file.read_text(encoding="utf-8", errors="replace").splitlines()[-400:]
            content = "\n".join(lines)
        else:
            content = ""
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.insert("1.0", content)
        self.log_text.configure(state="disabled")

    def browse_dataset_input(self) -> None:
        """Pick a source dataset file."""
        path = filedialog.askopenfilename(title=self._t("open_file_title"))
        if path:
            self.dataset_input_var.set(path)

    def browse_dataset_output(self) -> None:
        """Pick the folder where processed dataset files should be written."""
        path = filedialog.askdirectory(title=self._t("open_dir_title"))
        if path:
            self.dataset_output_var.set(path)

    def browse_training_config(self) -> None:
        """Pick the training YAML file."""
        path = filedialog.askopenfilename(title=self._t("open_file_title"), filetypes=[("YAML", "*.yaml *.yml"), ("All files", "*.*")])
        if path:
            self.training_config_var.set(path)

    def browse_evaluation_config(self) -> None:
        """Pick the evaluation YAML file."""
        path = filedialog.askopenfilename(title=self._t("open_file_title"), filetypes=[("YAML", "*.yaml *.yml"), ("All files", "*.*")])
        if path:
            self.evaluation_config_var.set(path)

    def browse_inference_context(self) -> None:
        """Pick an optional context file used during inference."""
        path = filedialog.askopenfilename(title=self._t("open_file_title"))
        if path:
            self.inference_context_file_var.set(path)

    def prepare_dataset_job(self) -> None:
        """Prepare the dataset in a background thread."""
        self._set_status(self._t("status_busy"))
        self.task_runner.submit("prepare_dataset", run_prepare_dataset, self.dataset_input_var.get(), self.dataset_output_var.get())

    def train_model_job(self) -> None:
        """Launch model training in a background thread."""
        self._set_status(self._t("status_busy"))
        self.task_runner.submit("train_model", run_training, self.training_config_var.get())

    def evaluate_model_job(self) -> None:
        """Run benchmark evaluation in a background thread."""
        self._set_status(self._t("status_busy"))
        self.task_runner.submit("evaluate_model", run_evaluation, self.evaluation_config_var.get())

    def run_assistant_job(self) -> None:
        """Run local inference from the assistant tab."""
        prompt = self.prompt_text.get("1.0", "end").strip()
        if not prompt:
            self._set_status(self._t("status_prompt_required"), level="warning")
            return
        self._set_status(self._t("status_busy"))
        self.task_runner.submit(
            "run_inference",
            run_inference,
            self.inference_task_var.get(),
            prompt,
            self.inference_language_var.get().strip() or None,
            self.inference_context_file_var.get().strip() or None,
        )

    def clear_prompt(self) -> None:
        """Clear the prompt editor without touching the result pane."""
        self.prompt_text.delete("1.0", "end")
        self._save_config()

    def copy_output(self) -> None:
        """Copy assistant output to the clipboard silently."""
        value = self.output_text.get("1.0", "end").strip()
        self.clipboard_clear()
        self.clipboard_append(value)
        self._set_status(self._t("status_copy_done"))

    def start_api(self) -> None:
        """Start the backend API silently in a background worker."""
        host = self.host_var.get().strip() or "127.0.0.1"
        port = parse_positive_int(self.port_var.get(), -1)
        if port < 1:
            self._set_status(self._t("status_port_invalid"), level="warning")
            return
        self._set_status(self._t("status_busy"))
        self.task_runner.submit("start_api", self.server_controller.start, host, port)

    def stop_api(self) -> None:
        """Stop the managed backend API silently."""
        self._set_status(self._t("status_busy"))
        self.task_runner.submit("stop_api", self.server_controller.stop)

    def check_api_health(self) -> None:
        """Check whether the configured API endpoint is responding."""
        host = self.host_var.get().strip() or "127.0.0.1"
        port = parse_positive_int(self.port_var.get(), -1)
        if port < 1:
            self._set_status(self._t("status_port_invalid"), level="warning")
            return
        self._set_status(self._t("status_busy"))
        self.task_runner.submit("check_api", self.server_controller.health, host, port)

    def request_close(self) -> None:
        """Close the app without disrupting active work or leaving the API behind."""
        if self.task_runner.has_active_tasks():
            self._set_status(self._t("status_close_blocked"), level="warning")
            return

        self._save_config()
        if self.server_controller.is_running():
            self._closing_after_task = True
            self.stop_api()
            return
        self.destroy()

    def open_log_file(self) -> None:
        """Open the runtime log file with the default OS handler."""
        log_file = get_log_file()
        log_file.touch(exist_ok=True)
        self._open_path(log_file)

    def open_config_file(self) -> None:
        """Open the runtime config file with the default OS handler."""
        self._save_config()
        config_file = get_config_file()
        config_file.touch(exist_ok=True)
        self._open_path(config_file)

    def open_about_dialog(self) -> None:
        """Show a lightweight About dialog without using message boxes."""
        dialog = tk.Toplevel(self)
        dialog.title(self._t("menu_about"))
        dialog.transient(self)
        dialog.resizable(False, False)
        dialog.configure(background="#fffdf8")
        body = ttk.Frame(dialog, padding=18)
        body.pack(fill="both", expand=True)
        ttk.Label(body, text=self._t("app_title"), style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            body,
            text=self._t(
                "about_body",
                name=self._t("app_title"),
                version=VERSION_TAG,
                year=datetime.now().year,
            ),
            justify="left",
        ).pack(anchor="w", pady=(12, 0))
        ttk.Button(body, text=self._t("about_ok"), style="Accent.TButton", command=dialog.destroy).pack(anchor="e", pady=(18, 0))

    def _open_path(self, path: Path) -> None:
        """Open a file using the local operating system integration."""
        try:
            if sys.platform.startswith("win"):
                os.startfile(path)  # type: ignore[attr-defined]
            else:
                LOGGER.info("Path ready at %s", path)
        except Exception:  # noqa: BLE001
            LOGGER.exception("Failed to open %s", path)
            self._set_status(self._t("status_error", message=str(path)), level="warning")

    def _t(self, key: str, **kwargs: object) -> str:
        """Shortcut for the current GUI language."""
        return translate(self.language_var.get(), key, **kwargs)


def run() -> None:
    """Launch the desktop GUI main loop."""
    window = LLMStudioWindow()
    window.mainloop()
