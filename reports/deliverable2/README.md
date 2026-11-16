# Deliverable 2: Data, Measurement and Uncertainty Characterization

- [Report PDF](Deliverable2_RobotJointTwin.pdf)
- [LaTeX source](main.tex)
- `images/dashboard_prototype.jpg`: cropped screenshot used in the report.
- `images/dashboard_original.jpg`: original viewport capture for provenance.

The report covers the physical and measurement model, acquisition and preprocessing, uncertainty budget, preliminary local dashboard, and planned validation. It reuses saved analysis results from commit `e889be13e25c536dbddf0a0d6340edbf8634f047`. Multi-seed train/test stability tests are future work; the screenshot does not demonstrate live robot deployment or real wear-to-failure validation.

## Build

From this directory, with [Tectonic](https://tectonic-typesetting.github.io/) installed:

```sh
tectonic main.tex
```

This creates `main.pdf`. After checking the rendered pages, copy it to `Deliverable2_RobotJointTwin.pdf` for publication. The LaTeX source and both screenshot files are sufficient to build the report; the raw robot dataset is not needed. The published PDF was compiled with Tectonic 0.17.0 and visually checked on all four pages.
