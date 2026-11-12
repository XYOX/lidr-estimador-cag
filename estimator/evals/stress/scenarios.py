"""Synthetic multi-turn conversations with a fact-tracker.

Three profiles, each built to force one behaviour of the CAG memory:

- ``growing``: coherent requirements pile up turn after turn. Measures the cost curve and whether
  the original project name survives to turn 20.
- ``pivot``: turn 5 swaps the stack (React -> Flutter). Measures whether the metadata updates
  cleanly or ``mentioned_technologies`` accumulates both.
- ``contradiction``: turn 3 locks a 30000 EUR budget, turn 8 locks 80000 EUR. Measures which
  figure is preserved, which gets promoted to an anchor and which ends up in the summary.

Every scenario has ``MAX_TURNS`` turns. The N in {1, 3, 6, 10, 20} sweep is read off prefixes of
that single run (``SCENARIO_LENGTHS``), so one 20-turn conversation answers all five lengths.

A ``Fact`` is matched as an exact, case-insensitive substring (``needle`` or any ``alias``) against
summary / anchors / metadata / window, so needles are distinctive tokens, not sentences.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

SCENARIO_LENGTHS = (1, 3, 6, 10, 20)
MAX_TURNS = max(SCENARIO_LENGTHS)


@dataclass(frozen=True)
class Fact:
    label: str
    needle: str
    aliases: tuple[str, ...] = ()


class ScenarioTurn(NamedTuple):
    turn_index: int  # 1-based
    transcript: str
    fact: Fact | None  # the claim later turns should still remember


@dataclass(frozen=True)
class StaleTerm:
    """A claim that stopped being true at ``from_turn``; finding it later is a symptom of drift."""

    label: str
    needle: str
    from_turn: int
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class Scenario:
    name: str
    turns: tuple[ScenarioTurn, ...]
    stale_terms: tuple[StaleTerm, ...] = ()

    def facts_declared_before(self, turn_index: int) -> list[tuple[int, Fact]]:
        """Facts declared at turns strictly before ``turn_index`` (the N > k of the metric)."""
        return [(t.turn_index, t.fact) for t in self.turns if t.fact and t.turn_index < turn_index]


def _turns(rows: list[tuple[str, Fact | None]]) -> tuple[ScenarioTurn, ...]:
    assert len(rows) == MAX_TURNS, f"scenario needs {MAX_TURNS} turns, got {len(rows)}"
    return tuple(ScenarioTurn(i, text, fact) for i, (text, fact) in enumerate(rows, start=1))


def _growing() -> Scenario:
    rows = [
        (
            "We want to build a B2B CRM called Nimbus for a sales team of 40 people. "
            "Stack: React + Postgres. Team of 3 developers.",
            Fact("project name", "Nimbus"),
        ),
        ("Add single sign-on through Okta so reps never manage a separate password.", Fact("sso", "Okta")),
        (
            "The product must be multi-tenant: every customer company gets an isolated workspace.",
            Fact("multi-tenant", "multi-tenant"),
        ),
        ("We also need an audit log of every change made to contacts and deals.", Fact("audit log", "audit log")),
        ("Managers want a CSV export of contacts and deals from any list view.", Fact("csv export", "CSV")),
        ("Billing: monthly subscription plans charged through Stripe.", Fact("billing", "Stripe")),
        (
            "Customers migrating from Salesforce need a one-off import of their accounts and contacts.",
            Fact("salesforce import", "Salesforce"),
        ),
        ("Add email sequences for outreach, sent through SendGrid.", Fact("email", "SendGrid")),
        ("Reps want SMS reminders for follow-ups, delivered through Twilio.", Fact("sms", "Twilio")),
        (
            "Permissions must be role-based: admin, manager and rep, each with different visibility.",
            Fact("rbac", "role-based"),
        ),
        ("Post deal-won notifications to Slack channels chosen by each tenant.", Fact("slack", "Slack")),
        ("Add a dashboard with a pipeline forecast by quarter and by rep.", Fact("forecast", "forecast")),
        (
            "Some tenants are European, so GDPR data retention and deletion requests must be supported.",
            Fact("gdpr", "GDPR"),
        ),
        ("Expose a public API with webhooks so tenants can react to deal changes.", Fact("webhooks", "webhooks")),
        ("Publish a Zapier app on top of that API for no-code automations.", Fact("zapier", "Zapier")),
        ("Provide a dark mode for the whole web interface.", Fact("dark mode", "dark mode")),
        (
            "The interface must be localised in Spanish and Portuguese besides English.",
            Fact("localisation", "Portuguese"),
        ),
        ("Enterprise customers will ask for a SOC 2 audit, so plan the evidence collection.", Fact("soc2", "SOC 2")),
        (
            "Nightly export of all tenant data to BigQuery for the customers' analysts.",
            Fact("bigquery", "BigQuery"),
        ),
        ("Finally, an in-app AI chatbot that answers questions about a tenant's pipeline.", Fact("chatbot", "chatbot")),
    ]
    return Scenario("growing", _turns(rows))


def _pivot() -> Scenario:
    rows = [
        (
            "We are building Helio, a scheduling app for HVAC field technicians. "
            "Frontend in React, backend in Node.js. Team of 2 developers.",
            Fact("project name", "Helio"),
        ),
        ("Dispatchers must see every technician's jobs for the day on a map.", Fact("map", "map")),
        ("Technicians need to attach photos to a job from the van.", Fact("photos", "photos")),
        ("Add push notifications when a job is reassigned.", Fact("push", "push notifications")),
        (
            "Big decision: we drop React entirely and rebuild the frontend in Flutter, "
            "so one codebase ships on iOS and Android.",
            Fact("stack pivot", "Flutter"),
        ),
        ("With Flutter, technicians must be able to work offline and sync when coverage returns.", Fact("offline", "offline")),
        ("Customers sign a digital work order on the technician's phone.", Fact("signature", "work order")),
        ("Integrate parts inventory lookups so technicians can check the van stock.", Fact("inventory", "inventory")),
        ("Add route optimisation to order the day's jobs by travel time.", Fact("routing", "route optimisation")),
        ("Invoices are generated automatically when a job is closed.", Fact("invoices", "invoices")),
        ("Dispatchers need a web admin panel to manage technicians and skills.", Fact("admin panel", "admin panel")),
        ("Add customer-facing SMS with the technician's arrival window.", Fact("sms", "arrival window")),
        ("Each job needs a checklist per equipment type, filled in on the phone.", Fact("checklist", "checklist")),
        ("Provide a weekly report of first-time-fix rate per technician.", Fact("report", "first-time-fix")),
        ("Support recurring maintenance contracts with automatic job creation.", Fact("contracts", "maintenance contracts")),
        ("Add barcode scanning of equipment serial numbers from the camera.", Fact("barcode", "barcode")),
        ("Technicians should see the full service history of the equipment on site.", Fact("history", "service history")),
        ("Integrate with the customer's accounting software for invoice sync.", Fact("accounting", "accounting software")),
        ("Add multi-language support: English and Spanish for the app.", Fact("languages", "Spanish")),
        ("Plan a pilot with ten technicians before the full rollout.", Fact("pilot", "pilot")),
    ]
    return Scenario(
        "pivot",
        _turns(rows),
        # React was dropped at turn 5: it should disappear from any memory that tracks the stack.
        stale_terms=(StaleTerm("react", "React", from_turn=5),),
    )


_BUDGET_30K = ("30000", "30,000", "30.000", "30k", "30 000")
_BUDGET_80K = ("80000", "80,000", "80.000", "80k", "80 000")


def _contradiction() -> Scenario:
    rows = [
        (
            "We need an internal onboarding portal called Atlas for new employees at a "
            "200-person company. Stack: Django + Postgres. Team of 2.",
            Fact("project name", "Atlas"),
        ),
        ("The portal must offer a checklist per role covering the first 30 days.", None),
        (
            "Important: the budget is locked at 30000 EUR for the whole project.",
            Fact("budget 30k", "30000", _BUDGET_30K[1:]),
        ),
        ("Employees upload their ID documents to the portal during onboarding.", None),
        ("Add single sign-on with the company's Google Workspace accounts.", Fact("sso", "Google Workspace")),
        ("HR wants automatic reminders when an onboarding task is overdue.", None),
        ("Managers should approve equipment requests directly from the portal.", None),
        (
            "Finance update: the budget is locked at 80000 EUR now, replacing the previous figure.",
            Fact("budget 80k", "80000", _BUDGET_80K[1:]),
        ),
        ("Add a welcome video library that new hires can browse at their own pace.", None),
        ("Integrate with the payroll provider so new hires are created automatically.", Fact("payroll", "payroll")),
        ("HR needs a dashboard with the onboarding completion rate per department.", None),
        ("Add a buddy system that assigns each new hire an experienced colleague.", Fact("buddy", "buddy system")),
        ("New hires must sign the employee handbook electronically inside the portal.", None),
        ("Provide a Slack bot that answers the most common onboarding questions.", Fact("slack bot", "Slack bot")),
        ("Support onboarding for remote employees in three time zones.", None),
        ("Add a feedback survey at day 30 and day 90 for every new hire.", Fact("survey", "survey")),
        ("IT wants a self-service page to track laptop and account provisioning.", None),
        ("Let HR export the onboarding progress of any cohort to a spreadsheet.", None),
        ("Add a mobile-friendly view so new hires can complete tasks from their phones.", Fact("mobile view", "mobile-friendly")),
        ("Plan the rollout in two waves, engineering first and then the rest of the company.", None),
    ]
    return Scenario(
        "contradiction",
        _turns(rows),
        # The 30000 EUR budget stopped being true at turn 8.
        stale_terms=(StaleTerm("budget 30k", "30000", from_turn=8, aliases=_BUDGET_30K[1:]),),
    )


_BUILDERS = {"growing": _growing, "pivot": _pivot, "contradiction": _contradiction}
SCENARIO_NAMES = tuple(_BUILDERS)


def build_scenario(name: str) -> Scenario:
    try:
        return _BUILDERS[name]()
    except KeyError:
        raise ValueError(f"unknown scenario {name!r}; available: {', '.join(SCENARIO_NAMES)}") from None
