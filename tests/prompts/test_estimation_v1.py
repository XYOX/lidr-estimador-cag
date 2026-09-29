from app.prompts.loader import render_estimation_prompt
from app.schemas import DetailLevel, EstimationRequest, OutputFormat, ProjectType


def _request(**overrides) -> EstimationRequest:
    defaults = dict(
        description="Necesitamos una app movil para pedir turnos en una clinica dental.",
        project_type=ProjectType.MOBILE_APP,
        detail_level=DetailLevel.MEDIUM,
        output_format=OutputFormat.LINE_ITEMS,
    )
    defaults.update(overrides)
    return EstimationRequest(**defaults)


def test_user_prompt_includes_description_literally():
    request = _request(description="Descripcion muy especifica del proyecto XYZ.")

    _, user = render_estimation_prompt(request)

    assert "<project_description>" in user
    assert "Descripcion muy especifica del proyecto XYZ." in user


def test_system_prompt_reflects_output_format():
    phases_table_request = _request(output_format=OutputFormat.PHASES_TABLE)
    narrative_request = _request(output_format=OutputFormat.NARRATIVE)

    system_phases, _ = render_estimation_prompt(phases_table_request)
    system_narrative, _ = render_estimation_prompt(narrative_request)

    assert "phases_table" in system_phases
    assert "phases_table" not in system_narrative


def test_system_prompt_reflects_detail_level():
    detailed_request = _request(detail_level=DetailLevel.DETAILED)
    summary_request = _request(detail_level=DetailLevel.SUMMARY)

    system_detailed, _ = render_estimation_prompt(detailed_request)
    system_summary, _ = render_estimation_prompt(summary_request)

    assert "asunciones" in system_detailed
    assert "asunciones" not in system_summary
