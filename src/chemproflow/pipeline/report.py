import os
from typing import Any, Dict

import jinja2


class Report:
    """Utility class to render a Jinja2 HTML report."""

    def __init__(self, template_path: str) -> None:
        if not os.path.isfile(template_path):
            raise ValueError(f"Template file does not exist: {template_path}")

        # Create a Jinja2 environment
        self.env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(os.path.dirname(template_path) or "."),
            autoescape=jinja2.select_autoescape(["html", "xml"]),
            trim_blocks=True,
            lstrip_blocks=True,
        )

        # Add global functions if needed
        self.env.globals.update(zip=zip, len=len, str=str)

        # Load the template by filename
        self.template = self.env.get_template(os.path.basename(template_path))

    def to_html(self, output_path: str, context: Dict[Any, Any]) -> None:
        """Render the template with context and write the output HTML file.

        Parameters
        ----------
        output_path : str
            Path of the output HTML file.
        context : dict
            Data to inject into the template.
        """
        rendered = self.template.render(context=context)

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fout:
            fout.write(rendered)