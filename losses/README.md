# Final loss source map

- Segmentation and TGD reweighting: `teacher_target_search_v2.py`.
- Policy JS and shared-memory evidence cosine losses: `student_search_kd_v2.py`.
- Legacy general loss utilities retained for baseline reproduction:
  `cocd/losses/`.

The project avoids duplicate wrappers here so the paper formula is tied to the
same implementation that generated the retained results.
