# CI/CD End-to-End Test

## Data-triggered run (lint-and-test + smoke-train)
- PR: https://github.com/tobni-islam/firewatch-ci/pull/4
- Actions run: https://github.com/tobni-islam/firewatch-ci/actions/runs/27977504402/job/82798963955?pr=4
- Result: PASS - both jobs green
- smoke-train mAP50: 0.1849
- Date: <22/06/2026>

## Model-triggered run (model_promotion.yml)
- Source run: https://github.com/tobni-islam/firewatch-ci/actions/runs/27859844445
- Promoted version: v4
- fire_mAP50: 0.5124
- smoke_mAP50: 0.3753

## MLflow Registry state
- Total registered versions: 4
- Current Production: v4
- Previous versions: Archived

## Links
- Repo: https://github.com/tobni-islam/firewatch-ci
- DagsHub: https://dagshub.com/islam_tb/firewatch-ci
- Streamlit: https://firewatch-ci-n66upqfbncipnrwa4pggct.streamlit.app/
