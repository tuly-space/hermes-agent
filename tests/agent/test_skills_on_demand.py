"""Profile-scoped skill-index disclosure; real disk/config/tool paths."""
import json

import pytest

from agent.prompt_builder import build_skills_system_prompt
from hermes_constants import get_hermes_home
from tools.skills_tool import skills_list, skill_view


@pytest.fixture
def profile(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    skill = tmp_path / "skills" / "testing" / "disclosure-probe"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "---\nname: disclosure-probe\ndescription: Unique disclosure description\n"
        "---\n\nUnique skill body for integration verification.\n"
    )
    return tmp_path


def configure(profile, value):
    (profile / "config.yaml").write_text(f"skills:\n  prompt_index: {value}\n")


def test_default_index_is_unchanged(profile):
    prompt = build_skills_system_prompt()
    assert "disclosure-probe" in prompt
    assert "Unique disclosure description" in prompt
    assert "Unique skill body" not in prompt


def test_on_demand_skips_scan_but_tools_still_work(profile, monkeypatch):
    configure(profile, "false")
    def unexpected_scan(*args, **kwargs):
        pytest.fail("on-demand prompt must not scan the skill index")
    monkeypatch.setattr("agent.prompt_builder.get_all_skills_dirs", unexpected_scan)
    prompt = build_skills_system_prompt(available_tools={"skills_list", "skill_view"})
    assert "skills_list" in prompt and "skill_view" in prompt
    assert "disclosure-probe" not in prompt
    assert "<available_skills>" not in prompt
    listing = skills_list(category="testing")
    if isinstance(listing, str):
        listing = json.loads(listing)
    assert "disclosure-probe" in json.dumps(listing)
    viewed = skill_view(name="disclosure-probe")
    if isinstance(viewed, str):
        viewed = json.loads(viewed)
    assert viewed["success"]
    assert "Unique skill body" in viewed["content"]


def test_explicit_profile_does_not_leak_to_ambient_home(profile, tmp_path):
    configure(profile, "true")
    other = tmp_path / "other-profile"
    other.mkdir()
    configure(other, "false")
    original_home = get_hermes_home()
    prompt = build_skills_system_prompt(skills_dir_override=other / "skills")
    assert "Skill discovery is on demand" in prompt
    assert get_hermes_home() == original_home
    assert "disclosure-probe" in build_skills_system_prompt()


@pytest.mark.parametrize("tools", [{"skill_view"}, {"skills_list"}, set()])
def test_missing_discovery_tool_preserves_index(profile, tools):
    configure(profile, "false")
    assert "disclosure-probe" in build_skills_system_prompt(available_tools=tools)


def test_setting_changes_do_not_reuse_wrong_index_cache(profile):
    configure(profile, "true")
    assert "disclosure-probe" in build_skills_system_prompt()
    configure(profile, "false")
    assert "disclosure-probe" not in build_skills_system_prompt()
    configure(profile, "true")
    assert "disclosure-probe" in build_skills_system_prompt()
