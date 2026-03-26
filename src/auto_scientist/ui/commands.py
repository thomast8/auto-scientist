"""Command palette provider for pipeline navigation and control."""

from __future__ import annotations

from functools import partial

from textual.command import Hit, Hits, Provider

from auto_scientist.ui.widgets import AgentPanel, IterationContainer


class PipelineCommandProvider(Provider):
    """Command palette provider for pipeline navigation and control."""

    async def search(self, query: str) -> Hits:
        matcher = self.matcher(query)
        app = self.app

        # Import at runtime to avoid circular import
        from auto_scientist.ui.app import PipelineApp

        if not isinstance(app, PipelineApp):
            return

        # Static commands
        commands = [
            ("Expand all panels", app.action_toggle_expand),
            ("Collapse all panels", app.action_toggle_expand),
            ("Go to top", partial(app._scroll_to, "top")),
            ("Go to bottom", partial(app._scroll_to, "bottom")),
            ("Quit", app.action_quit),
        ]

        # Theme switching
        for theme_name in sorted(app.available_themes):
            commands.append((
                f"Switch theme: {theme_name}",
                partial(app._set_theme, theme_name),
            ))

        # Pipeline control
        if hasattr(app._orchestrator, "pause_requested"):
            commands.append((
                "Pause after current iteration",
                partial(app._set_orchestrator_flag, "pause_requested"),
            ))
        if hasattr(app._orchestrator, "skip_to_report"):
            commands.append((
                "Skip to report",
                partial(app._set_orchestrator_flag, "skip_to_report"),
            ))

        # Dynamic: go to iteration N
        for container in app.query(IterationContainer):
            title = container.border_title or "?"
            commands.append((
                f"Go to {title}",
                partial(app._scroll_to_widget, container),
            ))

        # Dynamic: view agent details
        for panel in app.query(AgentPanel):
            commands.append((
                f"View {panel.panel_name} details",
                partial(app._open_agent_detail, panel),
            ))

        # Open experiment directory (macOS)
        if app._orchestrator and hasattr(app._orchestrator, "output_dir"):
            commands.append((
                "Open experiment directory",
                partial(
                    app._open_directory,
                    app._orchestrator.output_dir,
                ),
            ))

        for label, callback in commands:
            score = matcher.match(label)
            if score > 0:
                yield Hit(
                    score, matcher.highlight(label), callback,
                )
