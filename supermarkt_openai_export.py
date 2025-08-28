from __future__ import annotations

"""Integration with OpenAI for the Supermarkt challenge.

This module provides a small helper that sends a document image to an
OpenAI vision-enabled model and exports the resulting text to both JSON
and PDF files.  The script is intentionally lightweight so it can be used
as a starting point for the custom language *H* mentioned in the project
scope.
"""

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Optional

from openai import OpenAI
from fpdf import FPDF


@dataclass
class ExportPaths:
    """Convenience container for output file locations."""

    json_path: Path
    pdf_path: Path


class SupermarktOpenAIExport:
    """Wraps the OpenAI API call and export utilities."""

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        self.client = OpenAI()
        self.model = model

    def parse_image(self, image_url: str, outputs: ExportPaths) -> None:
        """Send ``image_url`` to the model and store results.

        Parameters
        ----------
        image_url:
            Public URL pointing to the image to be processed.
        outputs:
            ``ExportPaths`` object describing where JSON and PDF files
            should be written.
        """

        prompt = (
            "Extract all textual content from the image and return a\n"
            "readable transcript.  Avoid commentary and format the\n"
            "answer as plain text."
        )

        response = self.client.responses.create(
            model=self.model,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_image", "image_url": image_url},
                    ],
                }
            ],
            max_output_tokens=2048,
        )

        text = response.output_text
        outputs.json_path.write_text(json.dumps({"text": text}, indent=2), "utf-8")

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        for line in text.splitlines():
            pdf.multi_cell(0, 10, line)
        pdf.output(str(outputs.pdf_path))


def run(url: str, out_dir: Optional[str] = None) -> ExportPaths:
    out_dir = out_dir or os.getcwd()
    out = ExportPaths(
        json_path=Path(out_dir) / "supermarkt_output.json",
        pdf_path=Path(out_dir) / "supermarkt_output.pdf",
    )

    exporter = SupermarktOpenAIExport()
    exporter.parse_image(url, out)
    return out


if __name__ == "__main__":
    # Example usage with the résumé sample provided earlier
    run("https://i.imgur.com/mtLd48I.jpeg")
