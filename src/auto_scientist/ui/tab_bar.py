"""TabBar widget for switching between Home and experiment screens."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Button


_STATUS_INDICATORS = {
    "running": "\u25cf",   # ●
    "completed": "\u2713", # ✓
    "failed": "\u2717",    # ✗
    "paused": "\u25cb",    # ○
}


class TabBar(Widget):
    """Horizontal tab bar for switching between Home and experiment screens."""

    DEFAULT_CSS = """
    TabBar {
        dock: top;
        height: 3;
        background: $boost;
        padding: 0 1;
    }
    TabBar Horizontal {
        height: 3;
        width: 100%;
    }
    TabBar Button {
        min-width: 12;
        margin: 0 0 0 0;
    }
    TabBar .tab-active {
        background: $accent;
        text-style: bold;
    }
    """

    class TabSelected(Message):
        """Posted when a tab is clicked."""

        def __init__(self, sender: Widget, tab_id: str) -> None:
            super().__init__()
            self.tab_id = tab_id

    def __init__(self) -> None:
        super().__init__()
        self._tabs: dict[str, dict] = {
            "home": {"label": "Home", "status": None},
        }
        self._active: str = "home"

    @property
    def tab_ids(self) -> list[str]:
        """Return list of all tab IDs."""
        return list(self._tabs)

    @property
    def active_tab(self) -> str:
        """Return the currently active tab ID."""
        return self._active

    def compose(self) -> ComposeResult:
        with Horizontal():
            yield Button("Home", id="tab-home", classes="tab-active")

    def add_tab(self, label: str, tab_id: str, status: str = "running") -> None:
        """Add a new experiment tab."""
        if tab_id in self._tabs:
            return
        indicator = _STATUS_INDICATORS.get(status, "")
        display_label = f"{label} {indicator}".strip()
        self._tabs[tab_id] = {"label": label, "status": status}
        button = Button(display_label, id=f"tab-{tab_id}")
        self.query_one(Horizontal).mount(button)

    def remove_tab(self, tab_id: str) -> None:
        """Remove a tab. Cannot remove the Home tab."""
        if tab_id == "home" or tab_id not in self._tabs:
            return
        del self._tabs[tab_id]
        try:
            btn = self.query_one(f"#tab-{tab_id}", Button)
            btn.remove()
        except Exception:
            pass
        if self._active == tab_id:
            self.set_active("home")

    def set_active(self, tab_id: str) -> None:
        """Set the active tab and update styling."""
        if tab_id not in self._tabs:
            return
        self._active = tab_id
        for tid in self._tabs:
            try:
                btn = self.query_one(f"#tab-{tid}", Button)
                if tid == tab_id:
                    btn.add_class("tab-active")
                else:
                    btn.remove_class("tab-active")
            except Exception:
                pass

    def set_status(self, tab_id: str, status: str) -> None:
        """Update the status indicator on a tab."""
        if tab_id not in self._tabs:
            return
        self._tabs[tab_id]["status"] = status
        indicator = _STATUS_INDICATORS.get(status, "")
        label = self._tabs[tab_id]["label"]
        display = f"{label} {indicator}".strip()
        try:
            btn = self.query_one(f"#tab-{tab_id}", Button)
            btn.label = display
        except Exception:
            pass

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle tab button clicks."""
        btn_id = event.button.id or ""
        if btn_id.startswith("tab-"):
            tab_id = btn_id[4:]  # strip "tab-" prefix
            if tab_id in self._tabs:
                self.set_active(tab_id)
                self.post_message(self.TabSelected(self, tab_id))
