from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from patient_agent.domain.state import PatientCaseState


def render_report(
    state: PatientCaseState,
    output_dir: str = "data/reports",
) -> str:
    template_dir = Path("src/patient_agent/reports/templates")

    env = Environment(
        loader=FileSystemLoader(str(template_dir)),
        autoescape=False,
    )

    template = env.get_template("patient_report.md.j2")
    content = template.render(case=state)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    path = out_dir / f"{state.case_id}.md"
    path.write_text(content, encoding="utf-8")

    return str(path)