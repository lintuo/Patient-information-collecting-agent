import argparse
import json
import sys
from pathlib import Path
from typing import Any

import httpx


DEFAULT_API_BASE = "http://127.0.0.1:8000"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def request_json(
    client: httpx.Client,
    method: str,
    url: str,
    **kwargs: Any,
) -> dict[str, Any]:
    response = client.request(method, url, **kwargs)
    response.raise_for_status()
    return response.json()


def contains_text(value: Any, expected: str) -> bool:
    if value is None:
        return False

    if isinstance(value, list):
        return any(expected in str(item) for item in value)

    return expected in str(value)


def list_contains_all(value: Any, expected_items: list[str]) -> bool:
    if not expected_items:
        return True

    if value is None:
        return False

    if not isinstance(value, list):
        value = [value]

    text_values = [str(item) for item in value]

    return all(
        any(expected in item for item in text_values)
        for expected in expected_items
    )


def list_contains_any(value: Any, expected_items: list[str]) -> bool:
    if not expected_items:
        return True

    if value is None:
        return False

    if not isinstance(value, list):
        value = [value]

    text_values = [str(item) for item in value]

    return any(
        expected in item
        for expected in expected_items
        for item in text_values
    )


def assert_condition(errors: list[str], condition: bool, message: str) -> None:
    if not condition:
        errors.append(message)


def validate_state(scenario: dict[str, Any], state: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    expected = scenario.get("expected", {})

    if "status_one_of" in expected:
        assert_condition(
            errors,
            state.get("status") in expected["status_one_of"],
            f"status={state.get('status')} not in {expected['status_one_of']}",
        )

    if "missing_fields_empty" in expected:
        missing_fields = state.get("missing_fields", [])
        should_be_empty = expected["missing_fields_empty"]

        assert_condition(
            errors,
            (not missing_fields) == should_be_empty,
            f"missing_fields={missing_fields}, expected empty={should_be_empty}",
        )

    expected_flags = expected.get("red_flags_contains", [])
    actual_flags = state.get("red_flags", [])

    assert_condition(
        errors,
        list_contains_all(actual_flags, expected_flags),
        f"red_flags={actual_flags}, expected contains {expected_flags}",
    )

    facts = state.get("facts", {})
    expected_facts = expected.get("facts", {})

    for field, expected_value in expected_facts.items():
        if field.endswith("_contains"):
            actual_field = field.removesuffix("_contains")
            actual_value = facts.get(actual_field)

            if isinstance(expected_value, list):
                assert_condition(
                    errors,
                    list_contains_all(actual_value, expected_value),
                    f"facts.{actual_field}={actual_value}, expected contains {expected_value}",
                )
            else:
                assert_condition(
                    errors,
                    contains_text(actual_value, str(expected_value)),
                    f"facts.{actual_field}={actual_value}, expected contains {expected_value}",
                )
        else:
            actual_value = facts.get(field)
            assert_condition(
                errors,
                actual_value == expected_value,
                f"facts.{field}={actual_value}, expected {expected_value}",
            )

    triage_expected = expected.get("triage")
    triage_result = state.get("triage_result")

    if triage_expected:
        if triage_expected.get("should_not_exist"):
            assert_condition(
                errors,
                triage_result is None,
                f"triage_result should not exist, got {triage_result}",
            )
        else:
            assert_condition(
                errors,
                triage_result is not None,
                "triage_result is missing",
            )

            if triage_result:
                if "risk_level_one_of" in triage_expected:
                    assert_condition(
                        errors,
                        triage_result.get("risk_level") in triage_expected["risk_level_one_of"],
                        (
                            f"risk_level={triage_result.get('risk_level')}, "
                            f"expected one of {triage_expected['risk_level_one_of']}"
                        ),
                    )

                departments = triage_result.get("recommended_departments", [])

                if "departments_contains" in triage_expected:
                    assert_condition(
                        errors,
                        list_contains_all(departments, triage_expected["departments_contains"]),
                        (
                            f"departments={departments}, "
                            f"expected contains {triage_expected['departments_contains']}"
                        ),
                    )

                if "departments_contains_any" in triage_expected:
                    assert_condition(
                        errors,
                        list_contains_any(departments, triage_expected["departments_contains_any"]),
                        (
                            f"departments={departments}, "
                            f"expected contains any of {triage_expected['departments_contains_any']}"
                        ),
                    )

    return errors


def run_scenario(
    client: httpx.Client,
    api_base: str,
    scenario_path: Path,
    verbose: bool,
) -> bool:
    scenario = load_json(scenario_path)
    case_id = scenario["case_id"]

    if verbose:
        print(f"\n=== {scenario_path.name}: {scenario.get('description', '')} ===")

    request_json(client, "DELETE", f"{api_base}/cases/{case_id}")

    for turn in scenario.get("turns", []):
        if verbose:
            print(f"USER: {turn}")

        result = request_json(
            client,
            "POST",
            f"{api_base}/cases/{case_id}/turn",
            json={
                "text": turn,
                "auto_triage": False,
                "auto_report": False,
            },
        )

        if verbose:
            print(f"ASSISTANT: {result.get('assistant_message')}")

    if scenario.get("run_triage", False):
        request_json(client, "POST", f"{api_base}/cases/{case_id}/triage")

    if scenario.get("run_report", False):
        request_json(client, "POST", f"{api_base}/cases/{case_id}/report")

    state = request_json(client, "GET", f"{api_base}/cases/{case_id}")

    errors = validate_state(scenario, state)

    if errors:
        print(f"FAIL {scenario_path.name}")
        for error in errors:
            print(f"  - {error}")
        return False

    print(f"PASS {scenario_path.name}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--api-base",
        default=DEFAULT_API_BASE,
        help="FastAPI base URL",
    )
    parser.add_argument(
        "--scenario-dir",
        default="src/patient_agent/tests/scenarios",
        help="Directory containing scenario JSON files",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print each conversation turn",
    )

    args = parser.parse_args()

    scenario_dir = Path(args.scenario_dir)
    scenario_paths = sorted(scenario_dir.glob("*.json"))

    if not scenario_paths:
        print(f"No scenario files found in {scenario_dir}")
        return 1

    passed = 0

    with httpx.Client(timeout=120) as client:
        for scenario_path in scenario_paths:
            ok = run_scenario(
                client=client,
                api_base=args.api_base.rstrip("/"),
                scenario_path=scenario_path,
                verbose=args.verbose,
            )
            if ok:
                passed += 1

    total = len(scenario_paths)

    print(f"\nResult: {passed}/{total} passed")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())