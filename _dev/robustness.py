"""Synthetic mutation tests: run the Task 1 pipeline on edited CV texts."""
import asyncio, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import hw2

BASE = {}
for p in sorted(Path('public_test').glob('*.pdf')):
    BASE[p.name] = hw2.cv_text(p)

def sub(text, old, new):
    assert old in text, f"pattern not found: {old!r}"
    return text.replace(old, new, 1)

cases = {}  # name -> (text, expected_label)

# --- mutations that must be caught (label 0) -------------------------------
t = sub(BASE['CV_2.pdf'], 'BSc in ', 'MSc in ')
cases['m_degree_upgrade'] = (t, 0)

t = sub(BASE['CV_2.pdf'], 'Manager,     | BCG', 'Senior Manager, | BCG')
cases['m_title_inflate'] = (t, 0)

t = sub(BASE['CV_2.pdf'], 'Beijing,     | China', 'Shanghai,  | China')
cases['m_wrong_city'] = (t, 0)

t = sub(BASE['CV_2.pdf'], 'The University | of Hong | Kong', 'Kingsbridge College')
cases['m_fake_school'] = (t, 0)

t = sub(BASE['CV_2.pdf'], '2013 – 2017', '2014 – 2017')
cases['m_shift_year'] = (t, 0)

t = sub(BASE['CV_2.pdf'], '• Prototyping', '• Docker, Prototyping')
cases['m_extra_skill'] = (t, 0)

t = sub(BASE['CV_6.pdf'], 'Junior Engineer, Carousell', 'Engineer, Carousell')
cases['m_title_ok_dropjunior'] = (t, 1)   # Profile: Engineer (junior); dropping "Junior" is fine

t = sub(BASE['CV_6.pdf'], 'PolyU', 'Seoul National University')
cases['m_fake_school6'] = (t, 0)

# --- wording-only variants that must stay valid (label 1) ------------------
t = sub(BASE['CV_2.pdf'], 'BSc in ', 'Bachelor of Science in ')
cases['w_degree_words'] = (t, 1)

t = sub(BASE['CV_2.pdf'], '| • UI/UX | Design |', '| • UI/UX |')
cases['w_skill_short'] = (t, 1)

t = sub(BASE['CV_6.pdf'], 'Master of Business Administration in Legal', 'MBA in Legal')
cases['w_mba_words'] = (t, 1)

t = sub(BASE['CV_1.pdf'], 'Supply Chain, Inventory Management, Operations', 'Supply Chain, Operations')
cases['w_fewer_skills'] = (t, 1)

t = sub(BASE['CV_6.pdf'], 'PolyU', 'The Hong Kong Polytechnic University')
cases['w_school_fullname'] = (t, 1)

async def main():
    hw2.load_env_file()
    tools = await hw2.load_mcp_tools()
    agent = hw2.build_agent(tools)
    payload = {name: text for name, (text, _) in cases.items()}
    scores = await hw2.score_cvs(agent, payload)
    ok = 0
    for name, (_, expected) in cases.items():
        score = scores.get(name)
        pred = 1 if (score or 0) > 0.5 else 0
        good = pred == expected
        ok += good
        print(f"{'PASS' if good else 'FAIL'}  {name:24s} score={score}  expected={expected} predicted={pred}")
    print(f"\n{ok}/{len(cases)} mutations judged correctly")

asyncio.run(main())
