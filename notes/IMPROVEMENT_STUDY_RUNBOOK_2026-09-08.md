# Improvement study reproduction

Read the [study](APPLICATION_IMPROVEMENT_STUDY_2026-09-08.html) or its [Markdown source](APPLICATION_IMPROVEMENT_STUDY_2026-09-08.md). The [roadmap CSV](improvement_study_20260908_tables/08_roadmap.csv) contains the ranked implementation program. This study is separate from the earlier correctness review; neither study applies its proposed production fixes.

Use the Sportsbook repository root in PowerShell. The measurements require the local Parquet mirror and the already-used Python packages `numpy`, `pandas`, `pyarrow`, and `requests`, plus the repository's import dependencies. No new environment packages were installed for this study.

```powershell
Set-Location C:\Users\Dwilb\Desktop\Sportsbook
$env:PYTHONIOENCODING = 'utf-8'
python notes\improvement_study_20260908.py inventory
python notes\improvement_study_20260908.py decisions
python notes\improvement_study_20260908.py quotes
python notes\improvement_study_20260908.py performance
python notes\improvement_study_20260908.py nfl
python notes\improvement_study_20260908.py mlb
```

Each command writes only its own `notes/improvement_study_20260908_<command>.json`. The helper blocks outbound requests, including otherwise-free fallbacks, so that implicit paid provider paths cannot run. The study's external research was performed separately through public web pages. No paid APIs, database calls, model saves or promotions are needed by these measurements. The MLB component screen may take several minutes; no output is printed until it completes.

The original baseline was `bd18fd0`; source/data hashes are in the inventory JSON. Later changes to data or code can change the results. Run timing measurements without other heavy jobs, and interpret their ratios as local component observations. Re-running a measurement does not update numbers in the report automatically.

The NFL command exports individual forecast rows to support paired comparisons. It uses projected QB identity and rest, but inherits the documented injury availability and QB-overlap limitations. The MLB command isolates AB opportunity on an empirical H/AB basis and is not the deployed D+xBA pipeline. Neither is a certified profitable strategy or an untouched final test.

To regenerate the standalone HTML and eight CSV tables from the Markdown report:

```powershell
python notes\render_improvement_study.py
python -m compileall -q notes\improvement_study_20260908.py notes\render_improvement_study.py
git diff --check
```

The renderer supports this document's Markdown subset and uses only the standard library. The HTML uses local CSS, contains no remote assets or scripts, and supports browser printing. Rendering was inspected in a local headless Chrome session. Local links, internal anchors, eight tables and 23 linked source notes were checked.

A focused existing test run used these modules under the helper's network guard:

```powershell
@'
import runpy, unittest
runpy.run_path('notes/improvement_study_20260908.py', run_name='study_guard')
names = ['test_bet_selector', 'test_money_math', 'test_distributional',
         'test_bankroll', 'test_ops_telemetry']
result = unittest.TextTestRunner(verbosity=1).run(
    unittest.defaultTestLoader.loadTestsFromNames(names))
raise SystemExit(not result.wasSuccessful())
'@ | python -
```

Observed: 164 tests, 161 pass, three failures. The two `test_ops_telemetry.DbFailureWiringTests` cases for loading prop lines and team market stores pass when `ODI_BACKTEST_MIRROR=0` is set in a fresh process; installed mirror data otherwise bypasses their SQL mocks. `test_distributional.SelectLineMethodsTests.test_deep_bucket_adopts_d` still fails in isolation because it expects two buckets while `LINE_BUCKETS` now contains three. No tests or production code were changed to force a green result. The whole repository suite and deployed app were not certified.
